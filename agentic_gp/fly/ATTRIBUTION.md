# Fly brain data and model: attribution and license

## The data file (`data/fly/steering_circuit_783.npz` + `.json`)

The circuit file is **derived from FlyWire**. It is not a FlyWire product and is not endorsed by the
FlyWire Consortium.

> `data/fly/steering_circuit_783.*` is a derived subset of the FlyWire whole-brain connectome (FAFB public
> release v783, Oct-2023 snapshot) by the FlyWire Consortium.
>
> **Sources**
> - `Connectivity_783.parquet` from github.com/philshiu/Drosophila_brain_model (commit
>   `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`). It is derived from the FlyWire v783 edge list
>   (doi:10.5281/zenodo.10676866).
> - Cell-type annotations from github.com/flyconnectome/flywire_annotations (tag `v3.2.0`).
>
> **Changes**
> - Neurons were selected by cell type and path length (recipe in the `.json` sidecar and in
>   `agentic_gp/fly/extract.py`).
> - Synapses were summed per neuron pair.
> - Counts were signed +1/-1 by predicted neurotransmitter, as in Shiu et al. 2024.
> - Each LC10a cell gets a preferred eccentricity, computed from its synapse shares onto AOTU019
>   and AOTU025.
>
> **License:** CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/), per
> https://flywire.ai/guidelines. **Non-commercial use only.** The repository's MIT license covers
> code only, not this data.

Licensing notes (not legal advice):

- The Zenodo deposit is tagged CC BY 4.0, but flywire.ai/guidelines states "FlyWire's public release
  data is made available under license CC BY-NC 4.0". We follow the stricter reading.
- The flywire_annotations repository has no LICENSE file. The same annotations are published with
  Schlegel et al. 2024, a CC BY 4.0 article.
- The Shiu et al. repository is MIT-licensed (© 2023 Philip Shiu and Nico Spiller). That license
  covers its code. The FlyWire-derived files in it keep FlyWire's terms.
- **NonCommercial covers all use, not only sharing.** CC BY-NC 4.0 allows using, adapting and
  sharing the data and anything adapted from it (this circuit file, its plots and the race replays
  of the fly cars) only for purposes "not primarily intended for or directed towards commercial
  advantage or monetary compensation".
- **A free, non-commercial course** (for example a university class or an unpaid community
  bootcamp) may run the Fly Lab and share the extracted file with its students, with this
  attribution.
- **A paid bootcamp, a course sold to a client, or company training run for commercial advantage
  is probably commercial use.** Then do not use FlyWire-derived data at all: do not run the
  connectome part of the chapter, show its plots or hand out the file. Run only the self-contained
  toy fly. The CC BY 4.0 alternative, MaleCNS v1.0 (male-cns.janelia.org, "licensed under
  CC-BY"), is not implemented yet (README.md, "Future work"), so today there is no compliant
  connectome path for a commercial event.
- **The organiser must confirm the event's status before T−7.** The curriculum's T−7 checklist
  line that calls the Zenodo data "CC BY 4.0" is superseded by this file.

## Required citations

FlyWire's rule (from the citation table linked from codex.flywire.ai/about_flywire) is "When using the
FlyWire resource, please co-cite the Dorkenwald et al. and Schlegel et al. manuscripts."

- Dorkenwald, S. et al. (2024) Neuronal wiring diagram of an adult brain. *Nature* 634:124-138.
  doi:10.1038/s41586-024-07558-y
- Schlegel, P. et al. (2024) Whole-brain annotation and multi-connectome cell typing of *Drosophila*.
  *Nature* 634:139-152. doi:10.1038/s41586-024-07686-5

The model (LIF parameters, sign rule and the `Connectivity_783.parquet` file):

- Shiu, P.K. et al. (2024) A *Drosophila* computational brain model reveals sensorimotor processing.
  *Nature* 634:210-219. doi:10.1038/s41586-024-07763-9

Per-aspect citations requested by FlyWire and by flywire_annotations v3.2.0:

- Zheng, Z. et al. (2018) *Cell* 174:730-743. doi:10.1016/j.cell.2018.06.019 (EM volume)
- Buhmann, J. et al. (2021) *Nat Methods* 18:771-774. doi:10.1038/s41592-021-01183-7 (synapses)
- Heinrich, L. et al. (2018) MICCAI. doi:10.1007/978-3-030-00934-2_36 (synapses)
- Eckstein, N. et al. (2024) *Cell* 187:2574-2594. doi:10.1016/j.cell.2024.03.016 (neurotransmitters)
- Matsliah, A. et al. (2024) *Nature* 634:166-180. doi:10.1038/s41586-024-07981-1 (cell types)
- Berg, S. et al. (2026) *Cell* 189:5504-5526. doi:10.1016/j.cell.2026.08.015
- Tastekin, I. et al. (2026) *Cell* 189:5527-5551. doi:10.1016/j.cell.2026.08.016

Science behind the encoders and readout:

- Collie et al. 2026, *Neuron* 114:1833 (AOTU019 central / AOTU025 peripheral pathways)
- Sen et al. 2017, *Curr Biol* 27:766 (LC16 and backing up)
- Bidaye et al. 2014, *Science* 344:97 (moonwalker, MDN)
- Rayshubskiy et al. 2025, *eLife* 13 (DNa02 steering)
- Yang et al. 2024, *Cell* 187:6290 (steering descending neurons)

## The code (`agentic_gp/fly/*.py`)

The code is MIT-licensed with the rest of the repository. It was written for this course.

- The LIF equations and parameters re-implement Shiu et al.'s Brian2 model (`model.py`, MIT). Each
  time step follows Brian2's default schedule (integrate, threshold, deliver synapses, reset) and
  Brian2's refractory semantics; only the random-number stream differs.
- The circuit recipe follows webergithub/fruitfly-lab's 3,963-neuron selection rule, plus brake and
  escape modules from our own research. The same applies to the LC10a eccentricity encoder and the
  "permute post" scramble.
- No code was copied from fruitfly-lab, snuri00/fruit-fly-connectome or mutkuoz/flydoom. They have
  no license, so the ideas were re-implemented from their descriptions.

## Honesty statement (use this wording in slides and the app)

> "The fly brain is **derived from FlyWire**: 4,494 real neurons from the adult fruit-fly connectome
> with their real synapse counts, simulated with the Shiu et al. 2024 spiking model. What the brain
> sees and how its neurons move the car are **engineered by us**:
> - the eyes: bearing → LC10a (dimmed when a wall hides the waypoint), walls → LC16,
>   looming → LPLC2/LC4;
> - which neurons we read and their gains: steering from DNa02 and from AOTU019, a hub one synapse
>   after the LC10a input; brake and back-up from MDN; the emergency stop from the giant fiber
>   DNp01;
> - the 'spinal cord' speed rule.
>
> No neuron in this circuit drives forward speed. The real fly does not drive a car. What the
> connectome contributes is shown by the scrambled control: the same neurons, each with the same
> number of input and output connections, wired at random. The steering response disappears."

What this demo does **not** show:

- that a fly would steer this way in a car;
- that the circuit is complete. The selection is a hand-picked slice: about 3% of the brain's
  neurons (4,494 of 138,639) and 2.7% of its connected pairs (407,139 of 15,091,983);
- that the encoder rates are physiological;
- that the giant fiber's real (electrical) output is modelled. Only its chemical inputs are;
- that every left/right difference is biology. The left side fires about 1.2–1.7× more than the
  right in this circuit, which may be real asymmetry or may come from how the dataset was
  reconstructed and proofread.

Scrambled-control wording: the default scramble (`method="degree"`) keeps each neuron's number of
input and output connections and the synapse count and sign of each outgoing connection, but not
how many synapses each neuron receives. Do not say "same synapse counts" for it.
`scrambled(c, seed, method="strength")` also keeps each neuron's total excitatory and inhibitory
input to within about 5–10% (median). It also removes the steering response: lateralization
index at most 0.10 over 5 seeds, against 0.85 for the real wiring.
