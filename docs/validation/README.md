# Local validation records

Tests were run in separate virtual environments for Python 3.9–3.13.
The integration test skips unless IPORCH_REAL_MODEL is set; a green unit suite
is not evidence of real backend compatibility. Logs retain warnings.

A built wheel was installed in a fresh Python 3.12 virtual environment.
From /tmp (outside the checkout), ip-orch --help succeeded, metadata version
was 1.0.0 and both reference CSV resources were present. Wheel/sdist build
succeeded. Conda and Docker are unavailable locally; their execution is pending.
The official JOSS compiler is configured in .github/workflows/paper.yml;
official compilation passed remotely: https://github.com/cnpem/ip-orch/actions/runs/43956849850.

Zenodo metadata was fetched from https://zenodo.org/api/records/21357945.
The record confirms version and DOI, but its main-branch relation does not
identify an immutable Git commit.

Real MACE and ORB CPU runs passed in isolated Python 3.11 environments, and
IPOrch.run with parallel=2 completed both successfully. Logs and independent
JSON records are under examples/benchmark/results. GPU behavior remains untested.

The first remote CI run exposed unavailable Python 3.9 binaries on Ubuntu 26.04.
That job now uses Ubuntu 22.04; other versions retain ubuntu-latest.
