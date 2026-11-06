"""Build the classroom steering circuit from the FlyWire v783 connectome (instructor tool).

    python -m agentic_gp.fly.extract            # download (if needed) + extract
    python -m agentic_gp.fly.extract --check    # only verify the inputs' sha256

Inputs (pinned; downloaded into ``data/flywire/`` unless already there):

* ``Connectivity_783.parquet`` from github.com/philshiu/Drosophila_brain_model at commit
  91bdd1e7 (FlyWire v783 synapses summed per neuron pair, with Shiu et al.'s +1/-1
  transmitter sign; 15,091,983 pairs, 100.8 MB);
* ``Supplemental_file1_neuron_annotations.tsv`` from github.com/flyconnectome/
  flywire_annotations at tag v3.2.0 (cell types and soma sides, 31.7 MB).

Recipe (re-implemented from the research notes; neurons selected by ``cell_type``, never
``hemibrain_type``):

* steering core: neurons within 2 hops downstream of LC10a AND within 2 hops upstream of
  the steering DNs (DNa01/02/03/04/11) on edges with >= 5 synapses, plus both end sets
  (this is webergithub/fruitfly-lab's 3,963-neuron rule);
* brake module: neurons on LC16 -> MDN paths of <= 4 hops over edges with >= 10 synapses;
* escape module: LPLC2 + LC4 -> DNp01 (giant fiber) direct connections (>= 5 synapses);
* keep ALL connections among the selected neurons, whatever their strength.

Output: ``data/fly/steering_circuit_783.npz`` (+ ``.json`` sidecar with sizes,
provenance and checksums).  The output is FlyWire-derived data: CC BY-NC 4.0.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np

from .connectome import (Circuit, DEFAULT_CIRCUIT_PATH, REPO_ROOT, content_sha256, lc10a_preferred_eccentricity,
                         save_circuit)

DATA_DIR = REPO_ROOT / "data" / "flywire"
SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOTATIONS_TAG = "v3.2.0"

INPUTS = {
    "connectivity": {
        "file": "Connectivity_783.parquet",
        "url": f"https://github.com/philshiu/Drosophila_brain_model/raw/{SHIU_COMMIT}/Connectivity_783.parquet",
        "sha256": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        "bytes": 100_804_642,
    },
    "annotations": {
        "file": f"Supplemental_file1_neuron_annotations_{ANNOTATIONS_TAG}.tsv",
        "url": ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                f"{ANNOTATIONS_TAG}/supplemental_files/Supplemental_file1_neuron_annotations.tsv"),
        "sha256": "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd",
        "bytes": 31_720_298,
    },
}

RECIPE = {
    "steering_core": {"sources": ["LC10a"], "targets": ["DNa01", "DNa02", "DNa03", "DNa04", "DNa11"],
                      "min_synapses": 5, "max_hops_down": 2, "max_hops_up": 2, "rule": "down AND up"},
    "brake": {"sources": ["LC16"], "targets": ["MDN"], "min_synapses": 10, "max_path_hops": 4,
              "rule": "down + up <= max_path_hops"},
    "escape": {"sources": ["LPLC2", "LC4"], "targets": ["DNp01"], "min_synapses": 5, "max_path_hops": 1,
               "rule": "down + up <= max_path_hops"},
    "edges": "all connections among selected neurons (any synapse count)",
    "sign": "Shiu et al. 2024 'Excitatory' column (+1/-1 per presynaptic neuron)",
}

# groups reported in the sidecar (the ones the brain and the Fly Lab use)
KEY_TYPES = ["LC10a", "AOTU019", "AOTU025", "DNa01", "DNa02", "DNa03", "DNa04", "DNa11",
             "LC16", "MDN", "LPLC2", "LC4", "DNp01"]


# ------------------------------------------------------------- download
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_inputs(data_dir: Path = DATA_DIR, reuse_from: list[Path] | None = None, verify: bool = True) -> dict:
    """Make sure both pinned inputs are in ``data_dir``; returns {name: path}."""
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, spec in INPUTS.items():
        dst = data_dir / spec["file"]
        if not dst.exists():
            for d in reuse_from or []:
                cand = [p for p in Path(d).glob("*") if p.is_file() and p.stat().st_size == spec["bytes"]]
                if cand:
                    print(f"copying {cand[0]} -> {dst}")
                    shutil.copyfile(cand[0], dst)
                    break
        if not dst.exists():
            _download(spec["url"], dst)
        if verify:
            got = sha256(dst)
            if got != spec["sha256"]:
                raise RuntimeError(f"{dst} sha256 {got} != pinned {spec['sha256']}; delete it and re-run")
        out[name] = dst
    return out


def _download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"downloading {url}")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\r  {done / 1e6:.1f} MB in {time.time() - t0:.0f} s" + " " * 20)
    tmp.replace(dst)


# ------------------------------------------------------------ extraction
def _hops(src: np.ndarray, pre: np.ndarray, post: np.ndarray, n: int, max_hops: int) -> np.ndarray:
    """Multi-source BFS distance (in hops) from ``src`` following pre -> post; -1 = not reached."""
    dist = np.full(n, -1, np.int64)
    dist[src] = 0
    frontier = np.zeros(n, bool)
    frontier[src] = True
    for k in range(1, max_hops + 1):
        reached = np.zeros(n, bool)
        reached[post[frontier[pre]]] = True
        new = reached & (dist < 0)
        dist[new] = k
        frontier = new
    return dist


def _module(src, dst, pre, post, n, max_down, max_up=None, max_path=None):
    down = _hops(src, pre, post, n, max_down)
    up = _hops(dst, post, pre, n, max_up if max_up is not None else max_down)
    if max_path is None:
        keep = (down >= 0) & (up >= 0)
    else:
        keep = (down >= 0) & (up >= 0) & (down + up <= max_path)
    keep[src] = True
    keep[dst] = True
    return keep


def extract(conn_path: Path, ann_path: Path) -> tuple[Circuit, dict]:
    import pandas as pd
    import pyarrow.parquet as pq

    t0 = time.time()
    cols = ["Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory"]
    tbl = pq.read_table(conn_path, columns=cols)
    pre = tbl["Presynaptic_Index"].to_numpy().astype(np.int32)
    post = tbl["Postsynaptic_Index"].to_numpy().astype(np.int32)
    syn = tbl["Connectivity"].to_numpy().astype(np.int32)
    sign = tbl["Excitatory"].to_numpy().astype(np.int8)
    n = int(max(pre.max(), post.max())) + 1
    root = np.zeros(n, np.int64)
    root[pre] = tbl["Presynaptic_ID"].to_numpy()
    root[post] = tbl["Postsynaptic_ID"].to_numpy()
    del tbl
    print(f"connectome: {n:,} neurons, {pre.size:,} connections, {int(syn.sum()):,} synapses "
          f"({time.time() - t0:.1f} s)")

    ann = pd.read_csv(ann_path, sep="\t", usecols=["root_id", "cell_type", "side", "super_class"],
                      dtype={"root_id": "int64", "cell_type": str, "side": str, "super_class": str})
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(root)
    cell_type = ann["cell_type"].fillna("unknown").to_numpy(str)
    side = ann["side"].fillna("na").to_numpy(str)
    super_class = ann["super_class"].fillna("unknown").to_numpy(str)

    def ids(types):
        idx = np.flatnonzero(np.isin(cell_type, types))
        if idx.size == 0:
            raise RuntimeError(f"no neurons of type {types} (annotation version mismatch?)")
        return idx

    modules = {}
    for name in ("steering_core", "brake", "escape"):
        r = RECIPE[name]
        strong = syn >= r["min_synapses"]
        p, q = pre[strong], post[strong]
        if name == "steering_core":
            keep = _module(ids(r["sources"]), ids(r["targets"]), p, q, n, r["max_hops_down"], r["max_hops_up"])
        else:
            h = r["max_path_hops"]
            keep = _module(ids(r["sources"]), ids(r["targets"]), p, q, n, h, h, max_path=h)
        modules[name] = keep
        print(f"  module {name:14s} {int(keep.sum()):6,} neurons")
    sel = modules["steering_core"] | modules["brake"] | modules["escape"]
    nodes = np.flatnonzero(sel)
    new_index = np.full(n, -1, np.int64)
    new_index[nodes] = np.arange(nodes.size)
    e = sel[pre] & sel[post]
    c = Circuit(
        root_id=root[nodes], cell_type=cell_type[nodes], side=side[nodes], super_class=super_class[nodes],
        pre=new_index[pre[e]].astype(np.int32), post=new_index[post[e]].astype(np.int32),
        syn=(syn[e] * sign[e]).astype(np.int32), pref_ecc=np.full(nodes.size, np.nan, np.float32))
    c = replace(c, pref_ecc=lc10a_preferred_eccentricity(c))
    stats = {name: int(m.sum()) for name, m in modules.items()}
    print(f"circuit: {c.summary()}  ({time.time() - t0:.1f} s)")
    return c, stats


def build(out: Path = DEFAULT_CIRCUIT_PATH, data_dir: Path = DATA_DIR, reuse_from=None) -> Circuit:
    paths = fetch_inputs(data_dir, reuse_from=reuse_from)
    c, module_sizes = extract(paths["connectivity"], paths["annotations"])
    groups = {}
    for t in KEY_TYPES:
        for s in "LR":
            groups[f"{t}_{s}"] = int(c.group(f"{t}_{s}").size)
    meta = {
        "name": "FlyWire v783 steering circuit (LC10a steering core + LC16 brake + LPLC2/LC4 escape)",
        "n_neurons": c.n, "n_connections": c.n_edges,
        "n_synapses": int(np.abs(c.syn).sum()),
        "inhibitory_connection_fraction": round(float(np.mean(c.syn < 0)), 4),
        "module_sizes": module_sizes,
        "group_sizes": groups,
        "recipe": RECIPE,
        "sources": {k: {"url": v["url"], "sha256": v["sha256"], "bytes": v["bytes"]} for k, v in INPUTS.items()},
        "shiu_commit": SHIU_COMMIT, "annotations_tag": ANNOTATIONS_TAG,
        "flywire_release": "FAFB public release v783 (doi:10.5281/zenodo.10676866)",
        "license": ("CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), per https://flywire.ai/guidelines; "
                    "non-commercial use only"),
        "changes": ("neurons selected by cell type and path length (recipe above); synapses summed per neuron "
                    "pair; counts signed +1/-1 by predicted neurotransmitter as in Shiu et al. 2024; "
                    "LC10a preferred eccentricity derived from LC10a->AOTU019/AOTU025 synapse shares"),
        "attribution": "see agentic_gp/fly/ATTRIBUTION.md (cite Dorkenwald et al. 2024 and Schlegel et al. 2024)",
    }
    c = replace(c, meta=meta)
    save_circuit(c, out)           # deterministic bytes: no timestamp inside the .npz
    sidecar = dict(meta, created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   output={"file": out.name, "sha256": sha256(out), "bytes": out.stat().st_size,
                           "content_sha256": content_sha256(c)})
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.json').name}")
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agentic_gp.fly.extract", description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_CIRCUIT_PATH, help="output .npz")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR, help="where the downloaded inputs live")
    ap.add_argument("--reuse-from", type=Path, action="append", default=[],
                    help="folder with already-downloaded copies (matched by size, verified by sha256)")
    ap.add_argument("--check", action="store_true", help="only download/verify the inputs")
    ap.add_argument("--source", choices=["flywire"], default="flywire",
                    help="connectome source (MaleCNS v1.0, CC BY 4.0, is future work; see README)")
    a = ap.parse_args(argv)
    if a.check:
        for k, p in fetch_inputs(a.data_dir, reuse_from=a.reuse_from).items():
            print(f"ok  {k}: {p}")
        return 0
    build(a.out, a.data_dir, reuse_from=a.reuse_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
