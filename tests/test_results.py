import csv
from concurrent.futures import ProcessPoolExecutor

import pytest

from ip_orch.core.results import append_csv


def _append(args):
    path, index = args
    append_csv(path, {"id": index, "value": f"row,{index}"})


def test_concurrent_csv_writers_preserve_all_rows(tmp_path):
    path = tmp_path / "results.csv"
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_append, [(path, i) for i in range(40)]))
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert {int(row["id"]) for row in rows} == set(range(40))
    with pytest.raises(ValueError, match="schema"):
        append_csv(path, {"different": 1})
