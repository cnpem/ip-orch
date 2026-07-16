"""Built-in calculator builders for :mod:`ip_orch.core.model_factory`."""

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ModelBuildContext:
    device: str
    base_path: str
    models_path: Optional[str] = None


ModelBuilder = Callable[[ModelBuildContext], Any]

MODEL_ALIASES = {
    "pet-oam-xl": "pet_oam_xl",
    "nequip-oam-xl": "nequip_oam_xl",
    "matris-10m-oam": "matris_10m_oam",
    "sevennet-omni": "sevennet_omni",
    "nequip-oam-l": "nequip_oam_l",
    "tace-v1-oam-m": "tace_v1_oam_m",
    "grace-2l-oam": "grace_2l_oam",
    "grace-1l-oam": "grace_1l_oam",
    "grace-2l-mp-r6": "grace_2l_mp",
    "orb-v3": "orb_v3",
    "orb-v2": "orb_v2",
    "orb-v2-mptrj": "orb_v2_mptrj",
    "dpa-3.1-3m-ft": "dpa_3_1_3m_ft",
    "dpa-3.1-mptrj": "dpa_3_1_mptrj",
    "mace-mpa-0": "mace_mpa_0",
    "mace-mp-0": "mace_mp_0",
    "mace-mp": "mace_mp",
    "matris-10m-mp": "matris_10m_mp",
    "mattersim-v1": "mattersim_v1",
    "eqnorm-mptrj": "eqnorm_mptrj",
    "nequix-mp-1-pft": "nequix_mp_pft",
    "nequix-mp-pft": "nequix_mp_pft",
    "nequix-mp": "nequix_mp",
    "nequip-mp-l": "nequip_mp_l",
    "allegro-mp-l": "allegro_mp_l",
    "sevennet-l3i5": "sevennet_l3i5",
    "hienet": "hienet",
    "chgnet": "chgnet",
    "m3gnet": "m3gnet",
}

MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def model_builder(key: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        MODEL_BUILDERS[key] = builder
        return builder

    return decorator


def _model_file(ctx: ModelBuildContext, filename: str) -> str:
    return os.path.join(ctx.base_path, filename)


@model_builder("pet_oam_xl")
def build_pet_oam_xl(ctx: ModelBuildContext) -> Any:
    from upet.calculator import UPETCalculator

    return UPETCalculator(model="pet-oam-xl", version="1.0.0", device=ctx.device)


@model_builder("nequip_oam_xl")
def build_nequip_oam_xl(ctx: ModelBuildContext) -> Any:
    from nequip.ase import NequIPCalculator

    path = _model_file(ctx, "nequip-OAM-XL.nequip.pt2")
    return NequIPCalculator.from_compiled_model(compile_path=path, device=ctx.device)


@model_builder("matris_10m_oam")
def build_matris_10m_oam(ctx: ModelBuildContext) -> Any:
    from matris.applications.base import MatRISCalculator

    return MatRISCalculator(model="matris_10m_oam", device=ctx.device)


@model_builder("sevennet_omni")
def build_sevennet_omni(ctx: ModelBuildContext) -> Any:
    from sevenn.calculator import SevenNetCalculator
    from sevenn.calculator import SevenNetD3Calculator

    # return SevenNetCalculator(model="7net-omni", device=ctx.device, modal="mpa")
    return SevenNetD3Calculator(model='7net-omni', modal="mpa", device=ctx.device,functional_name='pbe',damping_type='damp_bj')

@model_builder("nequip_oam_l")
def build_nequip_oam_l(ctx: ModelBuildContext) -> Any:
    from nequip.ase import NequIPCalculator

    path = _model_file(ctx, "nequip-OAM-L.nequip.pt2")
    return NequIPCalculator.from_compiled_model(compile_path=path, device=ctx.device)


@model_builder("tace_v1_oam_m")
def build_tace_v1_oam_m(ctx: ModelBuildContext) -> Any:
    from tace.foundations import tace_foundations
    from tace.interface.ase import TACEAseCalc, add_dispersion  # noqa: F401

    model = tace_foundations["TACE-v1-OAM-M"]
    return TACEAseCalc(model=model, dtype="float32", device=ctx.device, level=0)


@model_builder("grace_2l_oam")
def build_grace_2l_oam(ctx: ModelBuildContext) -> Any:
    from tensorpotential.calculator import grace_fm

    return grace_fm("GRACE-2L-OAM", device=ctx.device)


@model_builder("orb_v3")
def build_orb_v3(ctx: ModelBuildContext) -> Any:
    from orb_models.forcefield import pretrained

    try:
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError:
        from orb_models.forcefield.calculator import ORBCalculator

    result = pretrained.orb_v3_conservative_inf_omat(
        device=ctx.device,
        precision="float32-high",
    )
    if isinstance(result, tuple):
        orbff, atoms_adapter = result
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=ctx.device)
    return ORBCalculator(result, device=ctx.device)


@model_builder("dpa_3_1_3m_ft")
def build_dpa_3_1_3m_ft(ctx: ModelBuildContext) -> Any:
    from deepmd.calculator import DP

    path = _model_file(ctx, "dpa3-openlam.pth")
    return DP(model=path)


@model_builder("mace_mpa_0")
def build_mace_mpa_0(ctx: ModelBuildContext) -> Any:
    from mace.calculators import mace_mp

    return mace_mp(model="medium-mpa-0", device=ctx.device)


@model_builder("matris_10m_mp")
def build_matris_10m_mp(ctx: ModelBuildContext) -> Any:
    from matris.applications.base import MatRISCalculator

    return MatRISCalculator(model="matris_10m_mp", device=ctx.device)


@model_builder("mattersim_v1")
def build_mattersim_v1(ctx: ModelBuildContext) -> Any:
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=ctx.device)


@model_builder("grace_1l_oam")
def build_grace_1l_oam(ctx: ModelBuildContext) -> Any:
    from tensorpotential.calculator import grace_fm

    return grace_fm("GRACE-1L-OAM", device=ctx.device)


@model_builder("eqnorm_mptrj")
def build_eqnorm_mptrj(ctx: ModelBuildContext) -> Any:
    from eqnorm.calculator import EqnormCalculator

    return EqnormCalculator(model_name="eqnorm", model_variant="eqnorm-mptrj", device=ctx.device)


@model_builder("nequix_mp_pft")
def build_nequix_mp_pft(ctx: ModelBuildContext) -> Any:
    from nequix.calculator import NequixCalculator

    return NequixCalculator(model_name="nequix-mp-1-pft", backend="jax")


@model_builder("nequip_mp_l")
def build_nequip_mp_l(ctx: ModelBuildContext) -> Any:
    from nequip.ase import NequIPCalculator

    path = _model_file(ctx, "nequip-MP-L.nequip.pt2")
    return NequIPCalculator.from_compiled_model(compile_path=path, device=ctx.device)


@model_builder("nequix_mp")
def build_nequix_mp(ctx: ModelBuildContext) -> Any:
    from nequix.calculator import NequixCalculator

    return NequixCalculator("nequix-mp-1", backend="torch")


@model_builder("allegro_mp_l")
def build_allegro_mp_l(ctx: ModelBuildContext) -> Any:
    from nequip.ase import NequIPCalculator

    path = _model_file(ctx, "allegro-MP-L.nequip.pt2")
    return NequIPCalculator.from_compiled_model(compile_path=path, device=ctx.device)


@model_builder("dpa_3_1_mptrj")
def build_dpa_3_1_mptrj(ctx: ModelBuildContext) -> Any:
    from deepmd.calculator import DP

    path = _model_file(ctx, "dpa3-mptrj.pth")
    return DP(model=path)


@model_builder("sevennet_l3i5")
def build_sevennet_l3i5(ctx: ModelBuildContext) -> Any:
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model="SevenNet-l3i5", device=ctx.device, modal="mpa")


@model_builder("hienet")
def build_hienet(ctx: ModelBuildContext) -> Any:
    from hienet.hienet_calculator import HIENetCalculator

    candidates = [
        _model_file(ctx, "HIENet-V3.pth"),
    ]
    checkpoint = next((path for path in candidates if os.path.exists(path)), None)

    if checkpoint is None:
        if ctx.models_path:
            checked = "\n".join(f"  - {path}" for path in candidates)
            raise FileNotFoundError(f"HIENet checkpoint not found. Checked:\n{checked}")
        return HIENetCalculator()

    signature = inspect.signature(HIENetCalculator)
    for parameter in ("checkpoint_path", "ckpt_path", "model_path", "weight_path", "weights_path"):
        if parameter in signature.parameters:
            return HIENetCalculator(**{parameter: checkpoint})

    return HIENetCalculator(checkpoint)


@model_builder("grace_2l_mp")
def build_grace_2l_mp(ctx: ModelBuildContext) -> Any:
    from tensorpotential.calculator import grace_fm

    return grace_fm("GRACE-2L-MP-r6", device=ctx.device)


@model_builder("mace_mp_0")
def build_mace_mp_0(ctx: ModelBuildContext) -> Any:
    from mace.calculators import mace_mp

    return mace_mp(model="large", device=ctx.device)


@model_builder("mace_mp")
def build_mace_mp(ctx: ModelBuildContext) -> Any:
    from mace.calculators import mace_mp

    return mace_mp(model="medium", device=ctx.device)


@model_builder("orb_v2")
def build_orb_v2(ctx: ModelBuildContext) -> Any:
    from orb_models.forcefield import pretrained

    try:
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError:
        from orb_models.forcefield.calculator import ORBCalculator

    result = pretrained.orb_v2(device=ctx.device, precision="float32-high")
    if isinstance(result, tuple):
        orbff, atoms_adapter = result
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=ctx.device)
    return ORBCalculator(result, device=ctx.device)


@model_builder("orb_v2_mptrj")
def build_orb_v2_mptrj(ctx: ModelBuildContext) -> Any:
    from orb_models.forcefield import pretrained

    try:
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError:
        from orb_models.forcefield.calculator import ORBCalculator

    result = pretrained.orb_mptraj_only_v2(device=ctx.device, precision="float32-high")
    if isinstance(result, tuple):
        orbff, atoms_adapter = result
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=ctx.device)
    return ORBCalculator(result, device=ctx.device)


@model_builder("chgnet")
def build_chgnet(ctx: ModelBuildContext) -> Any:
    from chgnet.model import CHGNetCalculator

    return CHGNetCalculator()


@model_builder("m3gnet")
def build_m3gnet(ctx: ModelBuildContext) -> Any:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    from m3gnet.models import M3GNetCalculator

    signature = inspect.signature(M3GNetCalculator)
    potential_parameter = signature.parameters.get("potential")
    if potential_parameter is None or potential_parameter.default is not inspect.Parameter.empty:
        return M3GNetCalculator()

    from m3gnet.models import M3GNet, Potential

    model = M3GNet.load()
    try:
        potential = Potential(model=model)
    except TypeError:
        potential = Potential(model)

    return M3GNetCalculator(potential=potential)
