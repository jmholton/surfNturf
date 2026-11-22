"""Bundled pretrained hydration PMF+LP+TetOZ weights and physical mobility presets for surfNturf."""
import json
import os

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

PRETRAINED_FILES = {
    "l2": "pmf_full_l2.npz",
    "default": "pmf_full_l2.npz",
    "kl": "pmf_full_kl.npz",
    "v1_l2": "pmf_v1_l2.npz",
    "v1_kl": "pmf_v1_kl.npz",
}


def get_pretrained_weights_path(variant: str = "l2") -> str:
    """Return absolute path to bundled pretrained surfNturf weights ('l2', 'kl', 'v1_l2', 'v1_kl'), or an existing path."""
    if os.path.isfile(variant):
        return os.path.abspath(variant)
    key = str(variant).strip().lower()
    if key not in PRETRAINED_FILES:
        raise ValueError(
            f"Unknown pretrained weights variant {variant!r}; choose 'l2', 'kl', 'v1_l2', 'v1_kl', or a .npz path."
        )
    p = os.path.join(_DATA_DIR, PRETRAINED_FILES[key])
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Bundled weights file missing: {p}")
    return p


def load_bmob_phys_preset(name: str = "dry_a4") -> dict:
    """Load a transferable 6-parameter physical water mobility preset ('dry_a4' or 'wet_b5b')."""
    p = os.path.join(_DATA_DIR, "bmob_phys_presets.json")
    with open(p, "r", encoding="utf-8") as f:
        presets = json.load(f)
    if name not in presets:
        raise KeyError(f"Unknown bmob preset {name!r}; available: {sorted(presets.keys())}")
    return dict(presets[name])
