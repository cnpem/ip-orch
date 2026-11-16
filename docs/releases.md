# Release procedure

The source version is declared in pyproject.toml. A version declaration does not
prove a PyPI release or an archived DOI. Before publishing:

1. Run the complete CI matrix and opt-in real backend suites; archive logs.
2. Build with `python -m build`; inspect wheel and sdist including reference CSVs.
3. Install the wheel in a fresh environment outside the checkout and run
   `ip-orch --help` plus the silicon benchmark in configured worker environments.
4. Record tested Python/backend versions, hardware and checkpoint hashes.
5. Resolve Unreleased notes into a dated version entry, review the PR, then tag
   the reviewed commit. Do not rewrite historical tags or fabricate old releases.
6. Compare installed `importlib.metadata.version('ip-orch')` with the tag and
   distribution metadata. Publish only validated artifacts.
7. Verify the Zenodo record resolves to that exact tag, distinguish version DOI
   from concept DOI, and add the verified archive citation to the manuscript.

Zenodo API record 21357945 was verified: version 1.0.0, publication date
2026-07-14, version DOI 10.5281/zenodo.21357945, parent record 21357944.
Its repository relation points to the moving main branch, so an exact Git commit
match remains unverified. The source still declares 1.0.0; audit changes are
unreleased and must not silently replace the archived distribution.
