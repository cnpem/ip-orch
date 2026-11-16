# Public issue backlog prepared for publication

Issue creation through the GitHub connector returned HTTP 403
`Resource not accessible by integration` during this audit. These are concrete
issue drafts, not claims that public issues exist.

| Issue | Evidence needed | Acceptance |
| --- | --- | --- |
| Confirm copyright holders and third-party notices | legal/institutional ownership and source provenance | reviewed MIT attribution with preserved notices |
| Archive surface benchmark inputs | 20 slab/DFT cache files, calculation settings, locks and checkpoint hashes | public rerun reproduces reported RMSE |
| Complete author disclosures | funding, prior AI use, exact tool versions, human review | no manuscript evidence placeholders |
| Establish public development history | visibility/audit history and dated public PRs/issues | more than six months of distributed public development verified |
| Review completed accelerator validation | real CUDA logs and precision-failure evidence in examples/benchmark/results/gpu | collaborator review of scope and memory limitations |
| Regenerate reference energies | immutable checkpoints and environment metadata | auditable tables with per-row status |
| Validate remaining model builders | each declared backend/version and checkpoint | finite energy/force logs and documented limitations |
| Verify release and archive identity | PyPI artifacts, Git tag and Zenodo version DOI | metadata and archived code correspond |

Review requests must be made to real collaborators. Automated checks are not
peer review. Do not merge or release until required reviews and validations pass.

## Legacy to-do migration

- Checkpoint root configuration is implemented via models_path; review individual
  builders for exceptions using the compatibility matrix.
- Package publication automation remains a focused release task: configure
  trusted publishing only after artifact/DOI validation and maintainer review.
- Uncertainty quantification through shallow ensembles is a future scientific
  feature; define ensemble calibration and validation before implementation.
