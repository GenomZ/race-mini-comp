"""A FlyWire-derived circuit: who the neurons are and how they are wired.

A :class:`Circuit` is plain numpy data:

* one row per neuron: FlyWire ``root_id``, ``cell_type`` (e.g. "LC10a"), ``side``
  ("left"/"right"/"center"/"na", the side of the cell body) and ``super_class``;
* one row per connected pair: ``pre`` -> ``post`` (neuron indices) with ``syn``, the
  number of synapses, signed by the presynaptic neuron's transmitter (+ excitatory,
  - inhibitory), exactly as in Shiu et al. 2024.

Neurons are addressed by group name: ``"LC10a_L"`` = left LC10a cells, ``"MDN_R"`` =
right MDN, ``"DNa02"`` = both sides.  Controls for the classroom: :func:`scrambled`
(same neurons, same number of connections in and out of every neuron, random wiring),
:func:`lesioned` (silence groups: they never fire), :func:`mirrored` (swap left and right) and
:func:`shallow` (keep only the direct input -> readout synapses).

The data file is produced by ``python -m agentic_gp.fly.extract`` (see README.md); it
is derived from FlyWire and licensed CC BY-NC 4.0 (see ATTRIBUTION.md).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .lif import LIFParams

__all__ = [
    "Circuit", "CircuitNotFoundError", "DEFAULT_CIRCUIT_PATH", "CIRCUIT_ENV_VAR", "circuit_path",
    "load_circuit", "save_circuit", "content_sha256", "scrambled", "lesioned", "mirrored", "shallow",
    "lc10a_preferred_eccentricity", "SCRAMBLE_STRENGTH_BINS",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CIRCUIT_PATH = REPO_ROOT / "data" / "fly" / "steering_circuit_783.npz"
#: set this environment variable to load the circuit from elsewhere (e.g. with a non-editable install)
CIRCUIT_ENV_VAR = "AGENTIC_GP_FLY_CIRCUIT"

SIDE_CODE = {"L": "left", "R": "right", "C": "center"}
_SIDE_SWAP = {"left": "right", "right": "left"}


@dataclass(frozen=True, eq=False)
class Circuit:
    root_id: np.ndarray          # int64 (n,)   FlyWire v783 root ids
    cell_type: np.ndarray        # str   (n,)   "unknown" when not typed
    side: np.ndarray             # str   (n,)   soma side
    super_class: np.ndarray      # str   (n,)
    pre: np.ndarray              # int32 (m,)   edge source index
    post: np.ndarray             # int32 (m,)   edge target index
    syn: np.ndarray              # int32 (m,)   signed synapse count
    pref_ecc: np.ndarray         # float32 (n,) LC10a preferred eccentricity 0..1 (NaN elsewhere)
    meta: dict = field(default_factory=dict)
    label: str = "real"
    silenced: np.ndarray | None = None   # bool (n,): lesioned cells (no synapses, no input, never fire)

    # ------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return int(self.root_id.size)

    @property
    def n_edges(self) -> int:
        return int(self.pre.size)

    def weight_mv(self, w_syn: float = LIFParams.w_syn) -> np.ndarray:
        """Synaptic weights in mV (signed synapse count x w_syn)."""
        return self.syn.astype(np.float64) * w_syn

    def group(self, name: str) -> np.ndarray:
        """Indices of a group: "LC10a_L", "MDN_R", "DNa02" (both sides), "DNp01_C"."""
        return _group_index(self, name)

    def groups(self, *names: str) -> dict[str, np.ndarray]:
        return {nm: self.group(nm) for nm in names}

    def silenced_index(self) -> np.ndarray:
        """Indices of lesioned (silenced) neurons."""
        return np.zeros(0, np.int64) if self.silenced is None else np.flatnonzero(self.silenced)

    def cell_types(self) -> list[str]:
        return sorted(set(self.cell_type.tolist()))

    def summary(self) -> str:
        inh = float(np.mean(self.syn < 0)) if self.n_edges else 0.0
        return (f"Circuit[{self.label}] {self.n:,} neurons, {self.n_edges:,} connections "
                f"({inh:.1%} inhibitory), {int(np.abs(self.syn).sum()):,} synapses")

    def __repr__(self) -> str:
        return self.summary()


def _type_side_index(c: Circuit) -> dict:
    tab = c.__dict__.get("_type_side_index")
    if tab is None:
        out: dict = {}
        for i, (t, s) in enumerate(zip(c.cell_type.tolist(), c.side.tolist())):
            out.setdefault((t, s), []).append(i)
        tab = {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}
        object.__setattr__(c, "_type_side_index", tab)   # cache on the (frozen) instance
    return tab


def _group_index(c: Circuit, name: str) -> np.ndarray:
    tab = _type_side_index(c)
    base, _, suffix = name.rpartition("_")
    if base and suffix in SIDE_CODE:
        return tab.get((base, SIDE_CODE[suffix]), np.zeros(0, np.int64))
    parts = [v for (t, _s), v in tab.items() if t == name]
    return np.sort(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


# ------------------------------------------------------------------ I/O
class CircuitNotFoundError(FileNotFoundError):
    pass


def circuit_path(path: str | Path | None = None) -> Path:
    """Where the circuit file is looked for: ``path``; else ``$AGENTIC_GP_FLY_CIRCUIT``; else
    ``<repo>/data/fly/steering_circuit_783.npz``; else ``./data/fly/steering_circuit_783.npz``."""
    if path is not None:
        return Path(path)
    env = os.environ.get(CIRCUIT_ENV_VAR)
    if env:
        return Path(env)
    if DEFAULT_CIRCUIT_PATH.exists():
        return DEFAULT_CIRCUIT_PATH
    cwd = Path.cwd() / "data" / "fly" / DEFAULT_CIRCUIT_PATH.name
    return cwd if cwd.exists() else DEFAULT_CIRCUIT_PATH


def load_circuit(path: str | Path | None = None) -> Circuit:
    """Load a circuit file (see :func:`circuit_path` for where it is looked for). Cached per file."""
    p = circuit_path(path)
    if not p.exists():
        raise CircuitNotFoundError(
            f"FlyWire circuit file not found: {p}\n"
            f"Student: ask your instructor for {p.name} and put it in data/fly/ (that one file is enough).\n"
            "Instructor: build it once with\n"
            "    python -m agentic_gp.fly.extract\n"
            "(downloads ~130 MB of FlyWire-derived data into data/flywire/, writes data/fly/), "
            "then hand the .npz + .json to students (non-commercial use only; see agentic_gp/fly/ATTRIBUTION.md).\n"
            f"Or set {CIRCUIT_ENV_VAR} to the path of an existing copy.")
    st = p.stat()
    return _load_cached(str(p.resolve()), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=4)
def _load_cached(path: str, _mtime: int, _size: int) -> Circuit:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        arrays = {k: z[k] for k in ("root_id", "cell_type", "side", "super_class", "pre", "post", "syn")}
        pref = z["pref_ecc"] if "pref_ecc" in z.files else None
    c = Circuit(**arrays, pref_ecc=np.full(arrays["root_id"].size, np.nan, np.float32), meta=meta)
    if pref is None:
        pref = lc10a_preferred_eccentricity(c)
    c = replace(c, pref_ecc=pref)
    for a in (c.root_id, c.cell_type, c.side, c.super_class, c.pre, c.post, c.syn, c.pref_ecc):
        a.setflags(write=False)
    return c


def _arrays(c: Circuit) -> dict[str, np.ndarray]:
    return {"root_id": c.root_id.astype(np.int64), "cell_type": c.cell_type.astype(str),
            "side": c.side.astype(str), "super_class": c.super_class.astype(str),
            "pre": c.pre.astype(np.int32), "post": c.post.astype(np.int32), "syn": c.syn.astype(np.int32),
            "pref_ecc": c.pref_ecc.astype(np.float32)}


def content_sha256(c: Circuit) -> str:
    """Hash of the circuit's arrays (not of the file's bytes): equal circuits, equal hash."""
    h = hashlib.sha256()
    for k, a in _arrays(c).items():
        h.update(f"{k}|{a.dtype.str}|{a.shape}".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def save_circuit(c: Circuit, path: str | Path) -> None:
    """Write a compressed .npz with fixed zip timestamps, so equal circuits give equal bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(_arrays(c), meta_json=np.array(json.dumps(c.meta, sort_keys=True)))
    tmp = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k, a in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(a), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, buf.getvalue())
    tmp.replace(path)


# ------------------------------------------------------- input mapping
def lc10a_preferred_eccentricity(c: Circuit) -> np.ndarray:
    """Retinotopy proxy for each LC10a cell (0 = straight ahead, 1 = 90 deg to the side).

    FlyWire has no receptive-field data, so (following webergithub/fruitfly-lab, re-implemented)
    each LC10a cell's eccentricity is estimated from the connectome: Collie et al. 2026 show
    AOTU019 carries the central visual field and AOTU025 the periphery, so a cell whose output
    goes mostly to the same-side AOTU019 is "central".  ``central = w19 / (w19 + w25)`` and
    ``pref_ecc = 1 - central`` (0.5 if it contacts neither).  Computed once from the REAL wiring
    and carried unchanged into scrambled copies, so controls differ only in wiring.
    """
    pref = np.full(c.n, np.nan, np.float32)
    for s in "LR":
        cells = c.group(f"LC10a_{s}")
        a19, a25 = c.group(f"AOTU019_{s}"), c.group(f"AOTU025_{s}")
        if cells.size == 0:
            continue
        w19 = np.zeros(c.n)
        w25 = np.zeros(c.n)
        m19 = np.isin(c.post, a19)
        m25 = np.isin(c.post, a25)
        np.add.at(w19, c.pre[m19], np.abs(c.syn[m19]))
        np.add.at(w25, c.pre[m25], np.abs(c.syn[m25]))
        tot = w19[cells] + w25[cells]
        central = np.where(tot > 0, w19[cells] / np.maximum(tot, 1), 0.5)
        pref[cells] = 1.0 - central
    return pref


# ----------------------------------------------------------- controls
#: |synapse count| bin edges for ``scrambled(..., method="strength")``: bins 1, 2, 3-4, 5-9, 10-19, 20-49, 50+
SCRAMBLE_STRENGTH_BINS = (2, 3, 5, 10, 20, 50)


def scrambled(c: Circuit, seed: int, method: str = "degree") -> Circuit:
    """Random rewiring that keeps the neurons and their connection counts (the scientific control).

    ``method="degree"`` (default): the list of edge targets is randomly permuted.  Every neuron
    keeps its out-degree, the synapse counts and sign of its outgoing connections (the sign
    belongs to the presynaptic transmitter) and its in-degree.  It does NOT keep how many
    synapses a neuron receives: a cell with a few strong inputs may end up with many weak ones
    (DNa02_L: 5,342 excitatory input synapses for real, about 1,600 scrambled).

    ``method="strength"`` (stricter): targets are permuted only among connections of the same
    sign and similar strength (|synapse count| bins :data:`SCRAMBLE_STRENGTH_BINS`).  On top of
    the above, every neuron then keeps roughly its total excitatory and inhibitory input.

    Either way only WHO connects to WHOM is randomised; self-loops and duplicate pairs can appear
    (as in fruitfly-lab).  Neuron identities and the input mapping are unchanged.
    """
    rng = np.random.default_rng(seed)
    if method == "degree":
        post = c.post[rng.permutation(c.n_edges)]
    elif method == "strength":
        key = np.digitize(np.abs(c.syn), SCRAMBLE_STRENGTH_BINS) * 2 + (c.syn < 0)
        post = c.post.copy()
        for k in np.unique(key):
            i = np.flatnonzero(key == k)
            post[i] = c.post[i[rng.permutation(i.size)]]
    else:
        raise ValueError("method must be 'degree' or 'strength'")
    tag = "" if method == "degree" else f", {method}"
    return replace(c, post=post, label=f"scrambled(seed={seed}{tag})")


def lesioned(c: Circuit, groups) -> Circuit:
    """Silence groups: their neurons lose every incoming and outgoing synapse, and FlyBrain gives
    them no sensory input, so they never fire (and read out as 0 Hz)."""
    names = [groups] if isinstance(groups, str) else list(groups)
    idx = np.concatenate([c.group(g) for g in names]) if names else np.zeros(0, np.int64)
    if names and idx.size == 0:
        raise ValueError(f"no neurons in groups {names}")
    keep = ~(np.isin(c.pre, idx) | np.isin(c.post, idx))
    silenced = np.zeros(c.n, bool) if c.silenced is None else c.silenced.copy()
    silenced[idx] = True
    return replace(c, pre=c.pre[keep], post=c.post[keep], syn=c.syn[keep], silenced=silenced,
                   label=f"{c.label}-lesion({'+'.join(names)})")


def mirrored(c: Circuit, cell_types=None) -> Circuit:
    """Swap left and right labels.

    ``cell_types=None`` (default): every neuron - the mirror-image fly.  The synapses are unchanged
    and the encoder AND the readout both see swapped labels, so this fly MUST still steer toward
    the target: it is a sanity check, and any change in lap time measures only the left/right
    differences of the data (and the per-side readout gains), not a new behaviour.

    ``cell_types=["LC10a", ...]``: swap only those types (e.g. the input neurons: "crossed eyes").
    Then the readout keeps its labels, so a fly with swapped eyes should steer AWAY from targets.
    """
    swap = np.array([_SIDE_SWAP.get(s, s) for s in c.side.tolist()], dtype=c.side.dtype)
    if cell_types is None:
        return replace(c, side=swap, label=f"{c.label}-mirrored")
    names = [cell_types] if isinstance(cell_types, str) else list(cell_types)
    only = np.isin(c.cell_type, names)
    if names and not only.any():
        raise ValueError(f"no neurons of types {names}")
    side = np.where(only, swap, c.side).astype(c.side.dtype)
    return replace(c, side=side, label=f"{c.label}-mirrored({'+'.join(names)})")


def shallow(c: Circuit, inputs, outputs) -> Circuit:
    """Keep ONLY the direct (one-synapse) connections from the ``inputs`` groups to the ``outputs``
    groups and drop every other connection (the other ~4,400 neurons stay but get no input).

    The second classroom control next to :func:`scrambled`: if the shallow fly drives about as well
    as the full one, the skill sits in the direct input -> readout synapses, not in the deep
    circuit.  Groups are names such as "LC10a_L" or "DNa02" (see :meth:`Circuit.group`)."""
    def idx(names):
        names = [names] if isinstance(names, str) else list(names)
        return np.concatenate([c.group(g) for g in names]) if names else np.zeros(0, np.int64)
    keep = np.isin(c.pre, idx(inputs)) & np.isin(c.post, idx(outputs))
    return replace(c, pre=c.pre[keep], post=c.post[keep], syn=c.syn[keep], label=f"{c.label}-shallow")
