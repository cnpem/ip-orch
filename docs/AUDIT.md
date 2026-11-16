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
- Real MACE and ORB CPU execution and isolated virtualenv installation are recorded in examples/benchmark/results. GPU concurrency, Conda installation,
  reference-table provenance remain to be validated. Remote CI passed:
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
