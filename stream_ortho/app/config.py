"""YAML configuration -> dataclasses, with strict validation.

Unknown keys raise (a silent typo in an ablation config would otherwise run the baseline and
look like a null result); every field has a default so a minimal YAML is valid; the resolved
configuration is written next to the run for reproducibility.
"""
from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class DataCfg:
    images: str = ""
    seq: str = "ieu"
    image_size: int = 512
    stride: int = 1
    intrinsics: str = "auto"            # auto | exif | provided | none
    intrinsics_value: Optional[List[float]] = None
    max_frames: int = 0                 # 0 = all
    plane_mode: str = "auto"            # live: auto | gt | plane | identity
    #: optional plane file (the legacy ``config.cfg``); when present the NPU dataset is used
    config_cfg: Optional[str] = None


@dataclass
class FrontendCfg:
    backend: str = "mast3r_slam"
    use_calib: str = "auto"             # auto | true | false
    keyframe_policy: str = "default"
    single_thread: bool = False


@dataclass
class FilterCfg:
    enabled: bool = True
    gate: bool = True
    nfp: bool = True
    prune: bool = True
    min_conf: float = 0.5
    depth_min_m: float = 5.0
    depth_max_m: float = 1500.0
    height_min_m: float = -3.0
    height_max_m: float = 120.0
    iso_radius_m: float = 1.5
    iso_conf_max: float = 1.2
    curtain_ratio: float = 4.0
    max_prune_frac: float = 0.35
    nfp_tau_m: float = 4.0          # how far above the column top counts as free space
    nfp_min_obs: int = 2            # how many observations mark a column as resolved
    nfp_mode: str = "above"         # above | below | both
    nfp_top_pct: float = 90.0
    nfp_below_min_gap_m: float = 1.5
    protect_unique: bool = True
    protect_low_obs: bool = True


@dataclass
class GaussianCfg:
    enabled: bool = True
    head_ckpt: str = ""
    keyframes_only: bool = True
    anchor: str = "head"                # head | pointmap
    apply_filter: bool = True
    # P5.1c: offline 3DGS final render (after streaming completes)
    final_enabled: bool = False         # render mosaic_final_3dgs after the run
    gsd_3dgs: float = 0.0               # 0 = auto (native pixel footprint)
    conf_threshold: float = 0.5         # min head confidence for a gaussian
    blend_levels: int = 4               # Laplacian pyramid depth (offline blend mode)
    feather_radius: int = 24            # blend feather width (canvas px)
    max_kf: int = 0                     # 0 = all keyframes; else first N (debug)
    # Streaming 3DGS (true per-keyframe ortho rendering & nadir consensus)
    streaming_3dgs: bool = False        # true = render 3DGS ortho on every keyframe
    adaptive_stride: bool = True        # stride=1 on edges/textures, flat_stride on smooth areas
    flat_stride: int = 2                # stride on homogeneous regions
    edge_grad_pct: float = 70.0         # image gradient percentile promoted to stride=1
    depth_upsample: str = "edge"        # edge | nearest | linear
    edge_rel_tau: float = 0.03          # relative depth-jump threshold (scale-invariant)
    edge_reject: int = 1                # depth-discontinuity curtain dilation radius (512px grid)
    edge_pct: float = 95.0              # top inverse-depth gradient percentile culled as curtains
    voxel_gsd_mul: float = 2.0          # voxel size = voxel_gsd_mul * gsd for nadir dedup
    nadir_replace_margin: float = 0.08  # replace voxel winner when nadir_score improves by margin
    gsd_mul: float = 1.0                # 1.0 = native image pixel GSD
    max_canvas_px: int = 6144           # safety cap on mosaic longest side in pixels
    tile_halo: int = 16                 # guard band (px) for seamless gsplat tile borders


@dataclass
class MapCfg:
    cell: float = 0.45
    z_gap: float = 2.0
    max_layers: int = 3
    freeze_owner: bool = True
    online_calib: bool = True
    window: int = 0
    window_every: int = 5
    window_rounds: int = 2
    #: live path: preallocated square product grid.  Without this the e2e build falls back to
    #: "first-frame bbox + 8 cells" and 84-93% of the later keyframes land outside it.
    grid_cells: int = 2048
    bootstrap_kfs: int = 2              # initial frames before plane alignment is locked
    progressive_align: bool = True      # per-keyframe local ground-normal & height rectification


@dataclass
class OrthoCfg:
    gsd: float = 0.30
    tile: int = 512
    render: str = "rgb"                 # rgb | texture | cell | hybrid | dsm
    fov_only: bool = True
    dirty_tiles: bool = True
    budget_ms: float = 500.0
    max_tiles: int = 6
    dump_every: int = 10
    edge_pct: float = 85.0
    edge_dilate: int = 2
    update_on: str = "frame"        # frame | keyframe
    async_worker: bool = True       # decouple SLAM frontend from 3DGS mapping/tile flush
    async_dump: bool = True         # write mosaics on a background thread
    dump_scale: float = 0.5         # preview scale (final always 1.0)
    dump_format: str = "png"        # png | jpg
    dump_queue: int = 2             # bounded queue length
    show: bool = False              # live OpenCV preview window (--show)


@dataclass
class OfflineCfg:
    enabled: bool = True
    iters: int = 3
    model: str = "hybrid"
    lam_p: float = 0.1
    render_texture: bool = True
    filter: bool = True          # run Module A on the table before the offline calibration
    carve: bool = True           # multi-view support carving on top of the filter


@dataclass
class OutputCfg:
    save_tiles: bool = True
    save_product: bool = True
    save_mosaic: bool = True
    save_json: bool = True


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    frontend: FrontendCfg = field(default_factory=FrontendCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    gaussian: GaussianCfg = field(default_factory=GaussianCfg)
    map: MapCfg = field(default_factory=MapCfg)
    ortho: OrthoCfg = field(default_factory=OrthoCfg)
    offline: OfflineCfg = field(default_factory=OfflineCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save(self, path) -> pathlib.Path:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml())
        return p


_SECTIONS = {"data": DataCfg, "frontend": FrontendCfg, "filter": FilterCfg,
             "gaussian": GaussianCfg, "map": MapCfg, "ortho": OrthoCfg,
             "offline": OfflineCfg, "output": OutputCfg}


def _fill(cls, values: Dict[str, Any], where: str):
    if not isinstance(values, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(values).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; known: {sorted(known)}")
    return cls(**values)


def load_config(path: str | pathlib.Path) -> Config:
    p = pathlib.Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError("top level of the config must be a mapping")
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"unknown section(s) {sorted(unknown)}; known: {sorted(_SECTIONS)}")
    cfg = Config()
    for name, cls in _SECTIONS.items():
        if name in raw:
            setattr(cfg, name, _fill(cls, raw[name], name))
    _validate(cfg)
    return cfg


def _validate(cfg: Config):
    if cfg.data.intrinsics not in ("auto", "exif", "provided", "none"):
        raise ConfigError(f"data.intrinsics must be auto|exif|provided|none, "
                          f"got {cfg.data.intrinsics!r}")
    if cfg.data.intrinsics == "provided" and not cfg.data.intrinsics_value:
        raise ConfigError("data.intrinsics=provided requires data.intrinsics_value")
    if cfg.frontend.use_calib not in ("auto", "true", "false"):
        raise ConfigError("frontend.use_calib must be auto|true|false")
    if cfg.filter.nfp_mode not in ("above", "below", "both"):
        raise ConfigError("filter.nfp_mode must be above|below|both")
    if cfg.ortho.dump_format not in ("png", "jpg", "jpeg"):
        raise ConfigError("ortho.dump_format must be png|jpg")
    if not (0.0 < cfg.ortho.dump_scale <= 1.0):
        raise ConfigError("ortho.dump_scale must be in (0, 1]")
    if cfg.ortho.update_on not in ("frame", "keyframe"):
        raise ConfigError("ortho.update_on must be frame|keyframe")
    if cfg.ortho.render not in ("texture", "cell", "hybrid", "rgb", "dsm"):
        raise ConfigError("ortho.render must be texture|cell|hybrid|rgb|dsm")
    if cfg.gaussian.anchor not in ("head", "pointmap"):
        raise ConfigError("gaussian.anchor must be head|pointmap")
    if cfg.gaussian.gsd_3dgs < 0:
        raise ConfigError("gaussian.gsd_3dgs must be >= 0 (0 = auto)")
    if not (0 <= cfg.gaussian.conf_threshold <= 10):
        raise ConfigError("gaussian.conf_threshold must be in [0, 10] (exp-activated)")
    if not (1 <= cfg.gaussian.blend_levels <= 8):
        raise ConfigError("gaussian.blend_levels must be in [1, 8]")
    if cfg.gaussian.feather_radius < 0:
        raise ConfigError("gaussian.feather_radius must be >= 0")
    if cfg.map.max_layers < 1:
        raise ConfigError("map.max_layers must be >= 1")
    if cfg.ortho.gsd <= 0 or cfg.map.cell <= 0:
        raise ConfigError("ortho.gsd and map.cell must be positive")
    if cfg.gaussian.depth_upsample not in ("edge", "nearest", "linear"):
        raise ConfigError("gaussian.depth_upsample must be edge|nearest|linear")
    if cfg.gaussian.flat_stride < 1:
        raise ConfigError("gaussian.flat_stride must be >= 1")


def override(cfg: Config, **kwargs) -> Config:
    """CLI overrides (only the documented ones)."""
    for key, value in kwargs.items():
        if value is None:
            continue
        section, _, name = key.partition("__")
        if not hasattr(cfg, section):
            raise ConfigError(f"cannot override {key}")
        setattr(getattr(cfg, section), name, value)
    _validate(cfg)
    return cfg
