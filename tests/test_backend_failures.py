"""Failures must remain attributable to the selected optional backend."""

import sys
from types import SimpleNamespace

import pytest

from ip_orch.core.model_factory import ModelFactory


def test_missing_optional_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "mace.calculators", None)
    with pytest.raises(ModuleNotFoundError, match="mace.calculators"):
        ModelFactory.create("mace-mp-0", device="cpu")


def test_incompatible_constructor_preserves_error(monkeypatch):
    def incompatible(**kwargs):
        raise TypeError("backend API incompatible with model keyword")

    monkeypatch.setitem(sys.modules, "mace.calculators", SimpleNamespace(mace_mp=incompatible))
    with pytest.raises(TypeError, match="backend API incompatible"):
        ModelFactory.create("mace-mp-0", device="cpu")


def test_missing_checkpoint_is_reported(monkeypatch, tmp_path):
    def load(*, model):
        with open(model, "rb") as handle:
            return handle.read()

    monkeypatch.setitem(sys.modules, "deepmd.calculator", SimpleNamespace(DP=load))
    with pytest.raises(FileNotFoundError, match="dpa3-openlam.pth"):
        ModelFactory.create("dpa-3.1-3m-ft", device="cpu", models_path=str(tmp_path))
