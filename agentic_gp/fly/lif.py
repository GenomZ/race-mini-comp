"""Leaky integrate-and-fire (LIF) network with the Shiu et al. (2024) parameters.

This is the "physics" of the fly brain: every neuron is one number (membrane voltage
v) that leaks back to rest, gets pushed up or down by incoming synapses, and fires a
spike when it crosses threshold.  It reproduces the model of

    Shiu, P.K. et al. (2024) A Drosophila computational brain model reveals
    sensorimotor processing.  Nature 634:210-219.  doi:10.1038/s41586-024-07763-9

with these choices (all from the paper's Brian2 code, see ATTRIBUTION.md):

* equations      dv/dt = (v_rest - v + g) / tau_m,   dg/dt = -g / tau_syn
                 integrated EXACTLY over each time step (no Euler error);
* threshold      v > -45 mV  ->  spike, v = -52 mV, g = 0;
* refractory     2.2 ms during which v and g are frozen and incoming spikes are DROPPED
                 (Brian2's "unless refractory"; letting input pile up during the
                 refractory period changes the results a lot);
* synapses       each spike adds  w_syn * (signed synapse count)  mV to the target's
                 g after a 1.8 ms delay (w_syn = 0.275 mV, the model's one free
                 parameter);
* sensory input  Poisson "kicks" of 68.75 mV straight into v, so every input event
                 makes the driven neuron fire; driven neurons have no refractory time.

Each time step follows Brian2's default schedule exactly: (1) integrate the neurons
that are not refractory, (2) threshold, (3) deliver synaptic input and Poisson kicks
that arrive at this step, (4) reset.  Delays and the refractory period are counted in
whole steps the way Brian2 does it (delay = round(1.8 / dt); a neuron is refractory
until int(2.2 / dt) steps after its spike).  So the code is "Brian2's schedule with
exact integration"; the only intended difference from Shiu's run is the random
number stream.

Speed trick (pure numpy, no compiled code needed): the 1.8 ms delay means a spike
emitted now cannot arrive for D = round(1.8 / dt) steps, so spikes are collected for
up to D steps and delivered in ONE vectorised scatter.  The result is exactly the same
as delivering them one step at a time.  If ``numba`` happens to be installed, a
compiled kernel with identical semantics is used instead (``backend="auto"``); it is
never required.

Example::

    net = LIFNetwork(n=3, pre=[0, 1], post=[1, 2], weight_mv=[30.0, 30.0])
    rates = np.zeros(3); rates[0] = 100.0           # drive neuron 0 at 100 Hz
    rates_hz, spikes = net.run(100.0, rates)        # 100 ms of simulated time
    spikes.t_ms, spikes.idx                         # every spike: time and neuron
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = ["LIFParams", "LIFNetwork", "Spikes", "numba_available"]


@dataclass(frozen=True)
class LIFParams:
    """Shiu et al. 2024 parameters (mV and ms)."""

    v_rest: float = -52.0
    v_reset: float = -52.0
    v_threshold: float = -45.0
    tau_m: float = 20.0          # membrane time constant
    tau_syn: float = 5.0         # synaptic (conductance-like) time constant
    t_refractory: float = 2.2
    t_delay: float = 1.8
    w_syn: float = 0.275         # mV per synapse (used by Circuit.weight_mv)
    poisson_kick: float = 68.75  # = w_syn * 250, mV added to v per input event
    dt: float = 0.25             # time step; Brian2's default is 0.1


class Spikes(NamedTuple):
    """All spikes of one ``run``: spike ``i`` was fired by neuron ``idx[i]`` at ``t_ms[i]``."""

    t_ms: np.ndarray
    idx: np.ndarray


def numba_available() -> bool:
    return _numba_kernel() is not None


_KERNEL = None
_KERNEL_TRIED = False


def _numba_kernel():
    """Compile (once) and return the numba kernel, or None when numba is missing."""
    global _KERNEL, _KERNEL_TRIED
    if _KERNEL_TRIED:
        return _KERNEL
    _KERNEL_TRIED = True
    try:
        from numba import njit
    except Exception:  # numba is optional
        return None

    @njit(cache=True)
    def kernel(steps, t0, D, R, u, g, refr_until, buf, indptr, indices, data, driven,
               kick_ptr, kick_idx, kick, em, es, cvg, theta, u_reset, counts, rec_t, rec_i, spk):
        n = u.shape[0]
        nrec = 0
        for s in range(steps):
            t = t0 + s
            slot = t % D
            nspk = 0
            for i in range(n):
                x = buf[slot, i]
                buf[slot, i] = 0.0
                if t <= refr_until[i]:
                    continue                      # refractory: frozen, input dropped
                u[i] = u[i] * em + g[i] * cvg      # (1) integrate
                g[i] = g[i] * es
                if u[i] > theta:                  # (2) threshold
                    spk[nspk] = i
                    nspk += 1
                g[i] += x                         # (3) synaptic input arriving now
            for k in range(kick_ptr[s], kick_ptr[s + 1]):
                u[kick_idx[k]] += kick            # (3) Poisson input (driven cells are never refractory)
            for j in range(nspk):                 # (4) reset
                i = spk[j]
                u[i] = u_reset
                g[i] = 0.0
                counts[i] += 1
                if not driven[i]:
                    refr_until[i] = t + R
                if nrec < rec_t.shape[0]:
                    rec_t[nrec] = t
                    rec_i[nrec] = i
                nrec += 1
                for k in range(indptr[i], indptr[i + 1]):   # arrives at t + D (same slot)
                    buf[slot, indices[k]] += data[k]
        return nrec

    _KERNEL = kernel
    return _KERNEL


class LIFNetwork:
    """A network of LIF neurons connected by delayed, weighted synapses.

    Args:
        n: number of neurons.
        pre, post: edge lists (neuron indices), one entry per connection.
        weight_mv: jump in the target's g (mV) per presynaptic spike; negative = inhibitory.
        params: :class:`LIFParams` (default: Shiu et al. 2024, dt = 0.25 ms).
        seed: seed of the Poisson input generator.
        backend: "numpy", "numba", or "auto" (numba when importable, else numpy).

    State (voltages, synaptic currents, spikes in flight, refractory clocks) is kept
    between :meth:`run` calls, so a driver can call ``run(100, rates)`` once per car tick.
    """

    def __init__(self, n, pre, post, weight_mv, params: LIFParams | None = None,
                 seed: int | None = 0, backend: str = "auto"):
        self.params = p = params or LIFParams()
        self.n = int(n)
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        w = np.asarray(weight_mv, dtype=np.float64)
        if not (pre.shape == post.shape == w.shape):
            raise ValueError("pre, post and weight_mv must have the same length")
        if pre.size and (min(pre.min(), post.min()) < 0 or max(pre.max(), post.max()) >= self.n):
            raise ValueError("edge index out of range")
        # CSR by presynaptic neuron: the outgoing edges of i are indptr[i]:indptr[i+1]
        order = np.argsort(pre, kind="stable")
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=self.n))]).astype(np.int64)
        self.indices = post[order]
        self.data = w[order]

        dt = p.dt
        self.dt = dt
        self.D = max(1, int(round(p.t_delay / dt)))          # delay in steps (Brian2 rounds)
        # Brian2: refractory until timestep(t - lastspike) >= timestep(t_refractory), timestep(x) =
        # int(x / dt + 1e-3).  So input is dropped on the R = that - 1 steps after the spike step.
        self.R = max(0, int(p.t_refractory / dt + 1e-3) - 1)
        self.em = float(np.exp(-dt / p.tau_m))
        self.es = float(np.exp(-dt / p.tau_syn))
        # exact solution of the linear ODE pair over one step (u = v - v_rest)
        self.cvg = float(p.tau_syn / (p.tau_syn - p.tau_m) * (self.es - self.em)) if p.tau_syn != p.tau_m \
            else float(dt / p.tau_m * self.em)
        self.theta = p.v_threshold - p.v_rest
        self.u_reset = p.v_reset - p.v_rest

        if backend not in ("auto", "numpy", "numba"):
            raise ValueError("backend must be 'auto', 'numpy' or 'numba'")
        if backend == "numba" and _numba_kernel() is None:
            raise ImportError("backend='numba' requested but numba is not installed")
        self.backend = "numba" if backend != "numpy" and _numba_kernel() is not None else "numpy"
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Back to rest: no spikes in flight, all voltages at v_rest, clock at 0."""
        n = self.n
        self.u = np.zeros(n)                       # v - v_rest
        self.g = np.zeros(n)
        self.buf = np.zeros((self.D, n))           # ring: input arriving at step t is buf[t % D]
        self.refr_until = np.full(n, -1, np.int64) # input dropped while step <= refr_until
        self.step_count = 0

    @property
    def t_ms(self) -> float:
        """Simulated time since construction / last reset."""
        return self.step_count * self.dt

    @property
    def v_mv(self) -> np.ndarray:
        return self.u + self.params.v_rest

    # -------------------------------------------------------------------- run
    def run(self, ms: float, input_rates_hz=None) -> tuple[np.ndarray, Spikes]:
        """Simulate ``ms`` milliseconds with Poisson input at ``input_rates_hz`` (length n, Hz).

        Returns ``(rates_hz, spikes)``: each neuron's firing rate over this window and
        every spike in it (``spikes.t_ms`` is absolute simulated time).
        """
        steps = int(round(ms / self.dt))
        if steps <= 0:
            return np.zeros(self.n), Spikes(np.zeros(0), np.zeros(0, np.int64))
        rates = np.zeros(self.n) if input_rates_hz is None else np.asarray(input_rates_hz, dtype=np.float64)
        if rates.shape != (self.n,):
            raise ValueError(f"input_rates_hz must have shape ({self.n},)")
        driven = rates > 0
        kick_ptr, kick_idx = self._poisson_schedule(rates, driven, steps)
        if self.backend == "numba":
            counts, st, si = self._run_numba(steps, driven, kick_ptr, kick_idx)
        else:
            counts, st, si = self._run_numpy(steps, driven, kick_ptr, kick_idx)
        window_s = steps * self.dt * 1e-3
        return counts / window_s, Spikes(st * self.dt, si)

    def _poisson_schedule(self, rates, driven, steps):
        """Input events for this window, sorted by step: kick_idx[kick_ptr[s]:kick_ptr[s+1]]."""
        stim = np.flatnonzero(driven)
        if stim.size == 0:
            return np.zeros(steps + 1, np.int64), np.zeros(0, np.int64)
        n_events = self.rng.poisson(rates[stim] * steps * self.dt * 1e-3)
        who = np.repeat(stim, n_events)
        when = self.rng.integers(0, steps, who.size)
        order = np.argsort(when, kind="stable")
        when, who = when[order], who[order]
        return np.searchsorted(when, np.arange(steps + 1)).astype(np.int64), who.astype(np.int64)

    def _run_numpy(self, steps, driven, kick_ptr, kick_idx):
        n, D = self.n, self.D
        u, g, buf, refr_until = self.u, self.g, self.buf, self.refr_until
        em, es, cvg, theta, u_reset, kick = self.em, self.es, self.cvg, self.theta, self.u_reset, self.params.poisson_kick
        t = self.step_count
        # Refractory bookkeeping without a per-step comparison: gate[i] is 1.0 when neuron i
        # listens to input and 0.0 while refractory; `release[step]` lists who re-opens then.
        gate = (refr_until < t).astype(np.float64)
        release: dict[int, list] = {}
        closed = np.flatnonzero(refr_until >= t)
        for step in np.unique(refr_until[closed]):
            release[int(step) + 1] = [closed[refr_until[closed] == step]]
        # refractory length per neuron; driven neurons have none (Shiu: rfc = 0 for stimulated cells)
        r_len = np.where(driven, -1, self.R)
        freeze = u_reset != 0.0             # v_reset == v_rest (default): u stays 0 by itself
        tmp = np.empty(n)
        above = np.empty(n, bool)
        kp = kick_ptr.tolist()
        multiply, greater = np.multiply, np.greater
        all_t, all_i = [], []
        s = 0
        while s < steps:
            block = min(D, steps - s)       # spikes in a block of <= D steps can be delivered together
            blk_t, blk_i = [], []
            for _ in range(block):
                opened = release.pop(t, None)
                if opened is not None:
                    for idx in opened:
                        gate[idx] = 1.0
                # (1) integrate (refractory cells sit at u_reset with g = 0, so this leaves them alone)
                u *= em
                multiply(g, cvg, out=tmp)
                u += tmp
                g *= es
                if freeze:
                    np.copyto(u, u_reset, where=gate == 0.0)
                # (2) threshold
                greater(u, theta, out=above)
                spk = above.nonzero()[0]
                # (3) synaptic input arriving now (dropped while refractory) and Poisson kicks
                arr = buf[t % D]
                multiply(arr, gate, out=arr)
                g += arr
                k0, k1 = kp[s], kp[s + 1]
                if k1 > k0:
                    u[kick_idx[k0:k1]] += kick
                # (4) reset
                if spk.size:
                    u[spk] = u_reset
                    g[spk] = 0.0
                    until = t + r_len[spk]
                    refr_until[spk] = until
                    shut = spk[until >= t + 1]       # driven neurons stay open
                    if shut.size:
                        gate[shut] = 0.0
                        release.setdefault(t + self.R + 1, []).append(shut)
                    blk_t.append(t)
                    blk_i.append(spk)
                t += 1
                s += 1
            if block == D:                           # every slot was consumed: clear them all at once
                buf.fill(0.0)
            else:
                buf[[(t - 1 - j) % D for j in range(block)]] = 0.0
            if blk_i:
                self._deliver(blk_t, blk_i)
                all_t.extend(blk_t)
                all_i.extend(blk_i)
        self.step_count = t
        if all_i:
            si = np.concatenate(all_i)
            st = np.repeat(np.asarray(all_t, np.int64), [len(x) for x in all_i])
        else:
            si = np.zeros(0, np.int64)
            st = np.zeros(0, np.int64)
        return np.bincount(si, minlength=n), st, si

    def _deliver(self, blk_t, blk_i):
        """Scatter the spikes of one block into the delay ring.

        A spike at step t arrives at step t + D, i.e. in ring slot t % D, which this block has
        already consumed, so delivering the whole block at its end is exact."""
        n, D = self.n, self.D
        counts = [len(x) for x in blk_i]
        si = np.concatenate(blk_i) if len(blk_i) > 1 else blk_i[0]
        start = self.indptr[si]
        length = self.indptr[si + 1] - start
        csum = np.cumsum(length)
        total = int(csum[-1])
        if total == 0:
            return
        # edge ids of all outgoing synapses of all spiking neurons, without a Python loop:
        # spike k owns edges start[k] .. start[k] + length[k] - 1
        eidx = np.repeat(start - (csum - length), length)
        eidx += np.arange(total)
        flat = np.repeat(np.repeat(np.asarray(blk_t, np.int64) % D * n, counts), length)
        flat += self.indices[eidx]
        self.buf += np.bincount(flat, weights=self.data[eidx], minlength=D * n).reshape(D, n)

    def _run_numba(self, steps, driven, kick_ptr, kick_idx):
        kernel = _numba_kernel()
        counts = np.zeros(self.n, np.int64)
        cap = steps * self.n // (self.R + 1) + kick_idx.size + 1024
        rec_t = np.empty(cap, np.int64)
        rec_i = np.empty(cap, np.int64)
        nrec = kernel(steps, self.step_count, self.D, self.R, self.u, self.g, self.refr_until, self.buf,
                      self.indptr, self.indices, self.data, driven, kick_ptr, kick_idx,
                      self.params.poisson_kick, self.em, self.es, self.cvg, self.theta, self.u_reset,
                      counts, rec_t, rec_i, np.empty(self.n, np.int64))
        self.step_count += steps
        m = min(nrec, cap)
        return counts, rec_t[:m].copy(), rec_i[:m].copy()
