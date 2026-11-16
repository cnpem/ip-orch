# Local validation records

Tests were run in separate virtual environments for Python 3.9–3.13.
The integration test skips unless IPORCH_REAL_MODEL is set; a green unit suite
is not evidence of real backend compatibility. Logs retain warnings.

A built wheel was installed in a fresh Python 3.12 virtual environment.
From /tmp (outside the checkout), ip-orch --help succeeded, metadata version
was 1.0.0 and both reference CSV resources were present. Wheel/sdist build
succeeded. Docker is unavailable locally. Conda was subsequently installed and validated as recorded below.
The official JOSS compiler is configured in .github/workflows/paper.yml;
official compilation passed remotely: https://github.com/cnpem/ip-orch/actions/runs/43956849850.

Zenodo metadata was fetched from https://zenodo.org/api/records/21357945.
The record confirms version and DOI, but its main-branch relation does not
identify an immutable Git commit.

Real MACE and ORB CPU runs passed in isolated Python 3.11 environments, and
IPOrch.run with parallel=2 completed both successfully. Logs and independent
JSON records are under examples/benchmark/results. GPU validation was subsequently completed as recorded below.

The first remote CI run exposed unavailable Python 3.9 binaries on Ubuntu 26.04.
That job now uses Ubuntu 22.04; other versions retain ubuntu-latest.

Remote CI passed all five Python jobs and wheel/sdist installation checks:
https://github.com/cnpem/ip-orch/actions/runs/43957054600.
The initial official paper build exposed clipped labels in the legacy SVG;
the replacement figure uses explicit bounds and larger labels.

Remaining validation: Docker and backends other than the two recorded integrations. The main branch has not been merged; public issues
and PR creation returned HTTP 403 through the connector. Review and merge must
be completed by collaborators with an authorized publishing channel.

Follow-up: the suite now has 82 passing unit tests. Conda installation, external
CLI and dependency consistency passed with PYTHONNOUSERSITE=1; see conda-* files.
GPU execution and derivative checks passed after selecting ORB float32-highest;
records of both the failed reduced-precision attempt and corrected run remain
in examples/benchmark/results/gpu. Docker is still unavailable locally.
