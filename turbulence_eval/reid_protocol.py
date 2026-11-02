"""Shared Figure 6 condition names and retrieval conventions."""

PROTOCOL = "caliper-reid-v1"
SCOPES = {"combined": None, "brc1_only": "brc1", "brc2_only": "brc2"}
SEVERITIES = {"weak": 1e-13, "medium": 4e-13, "strong": 1e-12}
STEMS = {
    "real-500m": "test_ground-range_realq500-chipset",
    "real-1000m": "test_ground-range_realq1000-chipset",
    **{key: f"test_ground-range_simq1000-cn2-{value:.0e}" for key, value in SEVERITIES.items()},
}
DEFAULT_MODELS = {
    "bot-market1501": ("BoT", "Market1501", "#0072B2"),
    "sinet-lsvid": ("SINet", "LS-VID", "#009E73"),
    "clip-reid-msmt17": ("CLIP-ReID", "MSMT17", "#E69F00"),
    "personvit-msmt17": ("PersonViT", "MSMT17", "#CC79A7"),
    "sapiensid-wb12m": ("SapiensID", "WebBody-12M", "#D55E00"),
}


def folder_name(value):
    from pathlib import Path
    if not value or value in (".", "..") or Path(value).name != value or "\\" in value:
        raise ValueError(f"Expected a folder name, got {value!r}")
    return value


def in_scope(tracklet, scope):
    return SCOPES[scope] is None or tracklet["collection"] == SCOPES[scope]
