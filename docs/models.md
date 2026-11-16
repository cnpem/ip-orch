# Model compatibility matrix

Implemented means a builder exists; it does not imply real validation. Versions
not backed by archived execution logs are deliberately marked unvalidated.

| Registry key | Import dependencies | Local checkpoint | Validation |
| --- | --- | --- | --- |
| pet_oam_xl | upet.calculator | upstream loader | unvalidated; see benchmark logs |
| nequip_oam_xl | nequip.ase | nequip-OAM-XL.nequip.pt2 | unvalidated; see benchmark logs |
| matris_10m_oam | matris.applications.base | upstream loader | unvalidated; see benchmark logs |
| sevennet_omni | sevenn.calculator | upstream loader | unvalidated; see benchmark logs |
| nequip_oam_l | nequip.ase | nequip-OAM-L.nequip.pt2 | unvalidated; see benchmark logs |
| tace_v1_oam_m | tace.foundations, tace.interface.ase | upstream loader | unvalidated; see benchmark logs |
| grace_2l_oam | tensorpotential.calculator | upstream loader | unvalidated; see benchmark logs |
| orb_v3 | orb_models.forcefield, orb_models.forcefield.inference.calculator, orb_models.forcefield.calculator | upstream loader | CPU validated: orb-models 0.5.5; results/orb-run.log |
| dpa_3_1_3m_ft | deepmd.calculator | dpa3-openlam.pth | unvalidated; see benchmark logs |
| mace_mpa_0 | mace.calculators | upstream loader | unvalidated; see benchmark logs |
| matris_10m_mp | matris.applications.base | upstream loader | unvalidated; see benchmark logs |
| mattersim_v1 | mattersim.forcefield | MatterSim-v1.0.0-5M.pth | unvalidated; see benchmark logs |
| grace_1l_oam | tensorpotential.calculator | upstream loader | unvalidated; see benchmark logs |
| eqnorm_mptrj | eqnorm.calculator | upstream loader | unvalidated; see benchmark logs |
| nequix_mp_pft | nequix.calculator | upstream loader | unvalidated; see benchmark logs |
| nequip_mp_l | nequip.ase | nequip-MP-L.nequip.pt2 | unvalidated; see benchmark logs |
| nequix_mp | nequix.calculator | upstream loader | unvalidated; see benchmark logs |
| allegro_mp_l | nequip.ase | allegro-MP-L.nequip.pt2 | unvalidated; see benchmark logs |
| dpa_3_1_mptrj | deepmd.calculator | dpa3-mptrj.pth | unvalidated; see benchmark logs |
| sevennet_l3i5 | sevenn.calculator | upstream loader | unvalidated; see benchmark logs |
| hienet | hienet.hienet_calculator | HIENet-V3.pth | unvalidated; see benchmark logs |
| grace_2l_mp | tensorpotential.calculator | upstream loader | unvalidated; see benchmark logs |
| mace_mp_0 | mace.calculators | upstream loader | CPU validated: mace-torch 0.3.14; results/mace-run.log |
| mace_mp | mace.calculators | upstream loader | unvalidated; see benchmark logs |
| orb_v2 | orb_models.forcefield, orb_models.forcefield.inference.calculator, orb_models.forcefield.calculator | upstream loader | unvalidated; see benchmark logs |
| orb_v2_mptrj | orb_models.forcefield, orb_models.forcefield.inference.calculator, orb_models.forcefield.calculator | upstream loader | unvalidated; see benchmark logs |
| chgnet | chgnet.model | upstream loader | unvalidated; see benchmark logs |
| m3gnet | m3gnet.models | upstream loader | unvalidated; see benchmark logs |

Checkpoint filenames are resolved under `models_path` (default
`~/.ip-orch/models`). Upstream loaders may download to their own caches.
Archive checkpoints and hashes: aliases alone do not identify immutable weights.

## Extending the registry

Implement `builder(context: ModelBuildContext)` returning an ASE calculator.
Context contains device, base_path and optional models_path. Register using
`ModelFactory.register("my_model", builder, "my-model")`. For a package plugin,
declare `[project.entry-points."ip_orch.model_builders"]` mapping a model key to
`package.module:builder`. Import backend dependencies inside the builder.
Test aliases, missing weights, missing dependencies and finite energy/forces.

The integration suite is opt-in:
`IPORCH_REAL_MODEL=mace-mp-0 pytest -m integration tests/integration`.
Run separately in each backend environment. No GPU validation is implied.
