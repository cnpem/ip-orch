import os
from types import SimpleNamespace
from unittest.mock import patch

from ip_orch.core.model_factory import ModelFactory, _norm


def test_norm_function():
    """Normalize model names into registry keys used by ModelFactory."""

    assert _norm("MACE-MP-0") == "mace_mp_0"
    assert _norm("  ORB-V3  ") == "orb_v3"
    assert _norm("") == ""


def test_register_model_builder():
    """Allow external callers to register a builder without editing factory code."""

    def builder(ctx):
        return {"device": ctx.device, "base_path": ctx.base_path}

    aliases = dict(ModelFactory._ALIASES)
    registry = dict(ModelFactory._REGISTRY)
    try:
        ModelFactory.register("Example Model", builder, "example-model")

        calc = ModelFactory.create("example-model", device="cpu", models_path="/tmp/models")

        assert calc == {"device": "cpu", "base_path": "/tmp/models"}
    finally:
        ModelFactory._ALIASES = aliases
        ModelFactory._REGISTRY = registry


def test_resolve_device_honors_explicit_device():
    """Expose the device selection used when creating calculators."""

    assert ModelFactory.resolve_device("cpu") == "cpu"


def test_model_factory_loads_entry_point_builders_once():
    """Load plugin builders from the ip_orch.model_builders entry point group."""

    def plugin_builder(ctx):
        return f"plugin:{ctx.device}"

    entry_point = SimpleNamespace(name="plugin-model", load=lambda: plugin_builder)
    aliases = dict(ModelFactory._ALIASES)
    registry = dict(ModelFactory._REGISTRY)
    entry_points_loaded = ModelFactory._ENTRY_POINTS_LOADED
    try:
        ModelFactory._ENTRY_POINTS_LOADED = False
        with patch("ip_orch.core.model_factory.metadata.entry_points") as mock_entry_points:
            mock_entry_points.return_value.select.return_value = [entry_point]

            assert ModelFactory.create("plugin-model", device="cpu") == "plugin:cpu"

        mock_entry_points.assert_called_once()
    finally:
        ModelFactory._ALIASES = aliases
        ModelFactory._REGISTRY = registry
        ModelFactory._ENTRY_POINTS_LOADED = entry_points_loaded


@patch("ip_orch.core.model_factory.os.path.expanduser")
@patch.dict("sys.modules", {"deepmd.calculator": __import__("unittest.mock").mock.MagicMock()})
def test_create_dpa_models(mock_expanduser):
    """Build DeepMD calculators from files at the configured models path root."""

    mock_expanduser.side_effect = lambda x: x  # identity
    from deepmd.calculator import DP

    ModelFactory.create("dpa-3.1-3m-ft", models_path="/my/models/path")
    DP.assert_called_with(model=os.path.join("/my/models/path", "dpa3-openlam.pth"))

    ModelFactory.create("dpa-3.1-mptrj", models_path="/my/models/path")
    DP.assert_called_with(model=os.path.join("/my/models/path", "dpa3-mptrj.pth"))


@patch("ip_orch.core.model_factory.os.path.expanduser")
@patch.dict("sys.modules", {"nequip.ase": __import__("unittest.mock").mock.MagicMock()})
def test_create_nequip_models(mock_expanduser):
    """Build NequIP calculators from compiled files at the models path root."""

    mock_expanduser.side_effect = lambda x: x
    from nequip.ase import NequIPCalculator

    ModelFactory.create("nequip-oam-xl", models_path="/temp")
    NequIPCalculator.from_compiled_model.assert_called_with(
        compile_path=os.path.join("/temp", "nequip-OAM-XL.nequip.pt2"),
        device="cpu",
    )


def test_device_environment_overrides_auto_detection(monkeypatch):
    monkeypatch.setenv("MLIP_DEVICE", "cpu")
    assert ModelFactory.resolve_device() == "cpu"
    assert ModelFactory.resolve_device("cuda:1") == "cuda:1"
