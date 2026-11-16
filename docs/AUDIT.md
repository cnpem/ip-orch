# Audit evidence and outstanding validation

This file tracks evidence gaps, not completion claims.

- LICENSE now attributes the project collectively to IP-Orch contributors,
  consistent with project authorship metadata. THIRD_PARTY_NOTICES.md preserves
  the previous Bootstrap attribution and records the scope of the source audit.
  Institutional ownership and unmarked copied code still require author review.
- Local Git history begins on 2026-05-16. Commit dates do not establish when the
  repository became public. Obtain repository visibility history and public
  issues/PRs; do not manufacture historical maintenance activity.
- Saved surface notebook outputs provide developer-use evidence (see research-evidence.md). Full reproduction still requires experiment identifiers, structures,
  checkpoint versions, measured outputs and publication links. A newly created
  benchmark cannot establish prior research impact.
- Real MACE and ORB CPU execution and isolated virtualenv installation are recorded in examples/benchmark/results. GPU concurrency and Conda installation are now validated below.
  Legacy reference-table provenance remains unresolved. Remote CI passed:
  https://github.com/cnpem/ip-orch/actions/runs/43957054600.
- Manuscript AI disclosure, funding, ownership require
  factual author confirmation. This audit used OpenAI Codex for code/document
  changes; exact model/build information and human review must be recorded by
  the submitting authors. No human validation is implied here.

Do not submit to JOSS until these evidence gaps and the remaining audit items
are resolved. Open focused issues and review actual fixes through pull requests.

GitHub API metadata reports repository creation on 2026-04-10 and current public
visibility; neither establishes continuous public visibility since creation.
Issue creation through the connector failed with HTTP 403. See backlog.md.

## Status against the requested audit

| Item | Completed work | Remaining acceptance evidence |
| --- | --- | --- |
| 1 | Project MIT attribution; historical notice preserved and packaged | Author confirmation of institutional ownership and unmarked source provenance |
| 2 | Ruff, 82 tests on Python 3.9–3.13; remote CI and package job green | Reviewed merge and green main-branch run |
| 3 | Actual saved surface workflow and measured RMSE documented | Original inputs, full provenance and any associated publication |
| 4 | Public audit branch and real validation-driven fixes | Visibility history, public issues/PR and collaborator review; connector writes returned 403 |
| 5 | Contribution and support guides | Maintainer review of published expectations |
| 6 | Renamed manuscript, required sections, official compilation and diagram repair | Funding, exact AI versions/prior use and human validation |
| 7 | Real MACE/ORB CPU calculators, forces/energy and isolated environment logs | Failure cases and GPU evidence now recorded; other models remain unvalidated |
| 8 | Built wheel/sdist, fresh virtualenv install, external CLI and CSV checks; pinned MACE | Conda execution validated; Docker execution unavailable locally |
| 9 | Portable silicon input, common script, two model results, versions and checkpoint hashes | Independent reproduction |
| 10 | Potential semantics: energy/free-energy/force/stress scaling and derivative tests | No known implementation gap from this audit |
| 11 | Units, regeneration procedure and legacy provenance gaps documented | H/C/Si regenerated for MACE/ORB; legacy values for other models still lack provenance |
| 12 | API/options, extension guide, model status matrix and troubleshooting | Real compatibility evidence for remaining models |
| 13 | Mixed failure/success regression, CSV locking test and real parallel CPU run | Parallel GPU run validated on RTX 3050 6 GB; larger jobs remain resource-dependent |
| 14 | Local checkpoint path repair, stale code cleanup and explicit status matrix | Real validation of every remaining advertised backend |
| 15 | Organization URLs and live CI badge fixed; Alpha retained conservatively | Authors/affiliations confirmed by submitting authors |
| 16 | Invisible characters removed, citation keys parsed, Rootstock and archive cited | Final author bibliography review |
| 17 | Unified Unreleased notes, release procedure, artifact checks and verified Zenodo metadata | Public issue migration, immutable archive-to-tag match and reviewed release |
| 18 | Scoped comparison, isolation trade-offs and script reuse demonstrated | Final scientific review of contribution claims |

A successful draft compilation does not make this a submission-ready manuscript.
No new release, reviewed PR, independent adoption or historical public activity
is claimed by this audit.


## Follow-up validation

- Conda installation used Micromamba with Python 3.11. External CLI, packaged
  CSVs and `pip check` passed with PYTHONNOUSERSITE=1. Records are in validation/.
- Real parallel CUDA execution passed for both models on RTX 3050 6 GB.
  GPU JSON records, driver information and memory samples are in
  examples/benchmark/results/gpu/.
- ORB float32-high exceeded the force consistency tolerance on this GPU;
  builders now request float32-highest. Both attempts are retained. The final
  ORB discrepancy is 0.002144 eV/Å, below the 0.05 criterion.
- Selected isolated references (H/C/Si) were regenerated with full conditions and
  backend versions. Legacy tables are retained without invented provenance.
- Missing backend, missing checkpoint, incompatible constructor and continued
  sequential execution after a failed preflight have regression tests.
- Docker/Podman are absent. No container-build success is claimed.
- GitHub PR creation was retried and again returned HTTP 403. No review or
  merge is claimed. Author-specific evidence requested earlier remains absent.
