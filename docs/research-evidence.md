# Research evidence

## Documented research use (preprint)

Zanineli, P. H. M., Focassio, B., and Schleder, G. R. (2026). [Cross-Geometry Transferability Assessment of Universal Machine Learning Interatomic Potentials: From Bulk Materials to Atomic Nanowires](https://doi.org/10.48550/arXiv.2608.06662). arXiv:2608.06662.

Section II.C identifies IP-Orch as the orchestration software for model loading, dataset evaluation and metric calculation in the zero-shot benchmark of 26 pretrained MLIPs across five zirconia geometry classes. This documents realized developer research use, not independent adoption or reproduction by the current release. The study does not validate the bundled legacy isolated-atom tables.

## Historical surface notebook

`examples/surfaces_benchmark.ipynb`, cell 13, contains saved IP-Orch execution
logs for seven calculators and bulk/slab/surface comparisons against DFT.
`examples/benchmark/notebook_evidence.json` extracts those outputs without
rerunning or altering numerical values.

| Model | Bulk RMSE (eV/atom) | Slab RMSE (eV/atom) | Surface RMSE (J/m²) |
| --- | ---: | ---: | ---: |
| MACE-MP | 0.060010 | 0.057503 | 0.352056 |
| MACE-MPA-0 | 0.021224 | 0.022068 | 0.096921 |
| ORB-v2 | 0.013204 | 0.069516 | 0.538228 |
| ORB-v2-mptrj | 0.014999 | 0.057647 | 0.276555 |
| ORB-v3 | 0.085764 | 0.073972 | 0.101283 |
| CHGNet | 0.070217 | 0.048725 | 0.538010 |

These values establish recorded developer use, not independent replication or
model ranking under fully controlled conditions. M3GNet reports one skipped
case. The notebook builds a cache with 20 surface and 20 DFT files; those input
files, DFT calculation settings, complete environment locks and checkpoint
hashes are not supplied here. Paths in historical logs are evidence only and
are not installation instructions. No publication specifically associated with this surface notebook has been verified; the zirconia study is a separate application.

IP-Orch's contribution was to select isolated backend interpreters and reuse
one comparison function across models. The new silicon benchmark tests that
execution pattern on a portable input; it cannot reconstruct the surface data.

Related tools: [Janus-Core](https://stfc.github.io/janus-core/) supplies ASE
calculation workflows, while [Rootstock](https://github.com/Garden-AI/rootstock)
hosts MLIPs across environments with an ASE-compatible interface and HPC
integration. IP-Orch supplies local script dispatch and interpreter selection.
Dependency isolation is shared with Rootstock; no exclusive novelty or measured
performance advantage is claimed. See the manuscript for the design trade-offs.
