from __future__ import annotations

import csv
import warnings
from importlib import resources

from .model_factory import ModelFactory

REFERENCE_ENERGIES_RESOURCE = "reference_energies.csv"
LEGACY_REFERENCE_WARNING = (
    "Bundled precomputed reference energies are legacy values with incomplete provenance. "
    "Regenerate references with the actual backend before scientific use."
)
NON_COMPUTABLE_REFERENCE_FAMILIES = ("grace", "matris", "m3gnet")


def canonical_model_name(model_name: str) -> str:
    aliases = getattr(ModelFactory, "_ALIASES", {})
    norm = _alias_match_token(model_name)
    for alias in aliases:
        if _alias_match_token(alias) == norm:
            return alias
    return _norm_token(model_name)


def is_reference_energy_computable(model_name: str) -> bool:
    canonical = canonical_model_name(model_name)
    return not any(
        canonical == family or canonical.startswith(f"{family}-") for family in NON_COMPUTABLE_REFERENCE_FAMILIES
    )


def load_precomputed_reference_energies(model_name: str, elements: list[str]) -> dict[str, float]:
    canonical = canonical_model_name(model_name)
    rows = _reference_energy_rows()
    if not rows:
        raise ValueError(f"No precomputed reference energy table found for {model_name!r}.")

    columns = set(rows[0].keys())
    if canonical not in columns:
        raise ValueError(f"No precomputed reference energies are available for model {model_name!r}.")

    values: dict[str, float] = {}
    missing = []
    for element in elements:
        row = _row_for_element(rows, element)
        raw = (row.get(canonical, "") if row else "").strip()
        if not raw:
            missing.append(element)
            continue
        values[element] = float(raw)

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing precomputed reference energies for {model_name!r}: {joined}.")
    warnings.warn(LEGACY_REFERENCE_WARNING, UserWarning, stacklevel=2)
    return values


def supported_reference_elements(model_name: str) -> list[str]:
    canonical = canonical_model_name(model_name)
    rows = _reference_energy_rows()
    if not rows or canonical not in rows[0]:
        return []
    return [row["element"] for row in rows if (row.get(canonical, "") or "").strip()]


def check_reference_elements(model_name: str, elements: list[str]) -> tuple[list[str], list[str]]:
    supported = set(supported_reference_elements(model_name))
    ok = [element for element in elements if element in supported]
    missing = [element for element in elements if element not in supported]
    return ok, missing


def _reference_energy_rows() -> list[dict[str, str]]:
    path = resources.files("ip_orch.scripts").joinpath(REFERENCE_ENERGIES_RESOURCE)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_for_element(rows: list[dict[str, str]], element: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("element") == element:
            return row
    return None


def _norm_token(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum() or ch in "-_")


def _alias_match_token(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())
