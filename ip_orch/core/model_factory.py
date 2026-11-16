"""
Factory to instantiate ASE calculators for common MLIP models.

Usage:
    from ip_orch.core.model_factory import ModelFactory
    calc = ModelFactory.create("mace-mp-0", device="cuda", models_path="~/pretrained")

Notes:
    - This runs inside the target Conda env where the model is installed.
    - Some models expect local weight files; pass models_path accordingly.
"""

import os
import re
from importlib import metadata
from typing import Any, Optional

from .model_builders import MODEL_ALIASES, MODEL_BUILDERS, ModelBuildContext, ModelBuilder

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

ENTRY_POINT_GROUP = "ip_orch.model_builders"


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


class ModelFactory:
    """Instantiate calculators by model alias."""

    _ALIASES = dict(MODEL_ALIASES)
    _REGISTRY = dict(MODEL_BUILDERS)
    _ENTRY_POINTS_LOADED = False

    @staticmethod
    def _device_to_str(device: Optional[str]) -> str:
        if device:
            return device
        if os.environ.get("MLIP_DEVICE"):
            return os.environ["MLIP_DEVICE"]
        if _HAS_TORCH:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return os.environ.get("MLIP_DEVICE", "cpu")

    @classmethod
    def resolve_device(cls, device: Optional[str] = None) -> str:
        """Return the device string that will be passed to model builders."""

        return cls._device_to_str(device)

    @classmethod
    def register(cls, key: str, builder: ModelBuilder, *aliases: str) -> None:
        """Register a calculator builder.

        ``key`` is normalized with the same rules used by :meth:`create`.
        Optional aliases are public user-facing names such as ``"mace-mp"``.
        """

        normalized_key = _norm(key)
        cls._REGISTRY[normalized_key] = builder
        cls._ALIASES.setdefault(key, normalized_key)
        for alias in aliases:
            cls._ALIASES[alias] = normalized_key

    @classmethod
    def _load_entry_points(cls) -> None:
        if cls._ENTRY_POINTS_LOADED:
            return
        cls._ENTRY_POINTS_LOADED = True
        try:
            eps = metadata.entry_points()
            if hasattr(eps, "select"):
                selected = eps.select(group=ENTRY_POINT_GROUP)
            else:  # pragma: no cover - compatibility for older importlib_metadata
                selected = eps.get(ENTRY_POINT_GROUP, [])
        except Exception:
            return

        for entry_point in selected:
            builder = entry_point.load()
            cls.register(entry_point.name, builder)

    @classmethod
    def create(cls, model_name: str, device: Optional[str] = None, models_path: Optional[str] = None) -> Any:
        """Instantiate an ASE calculator for the requested model alias."""

        cls._load_entry_points()
        key = cls._ALIASES.get(model_name, _norm(model_name))
        builder = cls._REGISTRY.get(key)
        if builder is None:
            return None

        resolved_device = cls.resolve_device(device)
        base_path = os.path.expanduser(models_path) if models_path else os.path.expanduser("~/.ip-orch/models")
        context = ModelBuildContext(device=resolved_device, base_path=base_path, models_path=models_path)
        return builder(context)
