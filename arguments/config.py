"""
YAML-based configuration for NURBS optimization.

Loads from a default YAML, merges with an optional scene-specific YAML,
and allows CLI overrides.  Exposes a flat attribute interface for backward
compatibility (opt.nurbs_weight_lr still works).
"""
from __future__ import annotations

import copy
import os
from math import inf
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# ─── Paths ───────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_YAML = _THIS_DIR.parent / "configs" / "nurbs_default.yaml"


# ─── YAML helpers ────────────────────────────────────────────────────────────

def _yaml_inf_constructor(loader, node):
    """Handle .inf in YAML → Python float('inf')."""
    return inf


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:float", _yaml_inf_constructor)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict: {'a': {'b': 1}} → {'a.b': 1}."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ─── Core config object ─────────────────────────────────────────────────────

class NurbsConfig:
    """
    Hierarchical, YAML-backed configuration for NURBS optimization.

    Access patterns (all equivalent):
        cfg.nurbs_weight_lr          # flat attribute (backward compat)
        cfg["learning_rates.nurbs_weight_lr"]  # dotted key
        cfg.raw["learning_rates"]["nurbs_weight_lr"]  # raw nested dict

    Construction precedence (last wins):
        1. configs/nurbs_default.yaml
        2. scene-specific YAML (--nurbs_config path)
        3. CLI overrides (--nurbs.learning_rates.nurbs_weight_lr 0.01)
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ):
        # 1. Load defaults
        with open(_DEFAULT_YAML) as f:
            self.raw: dict = yaml.safe_load(f)

        # 2. Merge scene-specific YAML
        if config_path is not None:
            with open(config_path) as f:
                scene_cfg = yaml.safe_load(f) or {}
            _deep_merge(self.raw, scene_cfg)

        # 3. Apply CLI overrides (dotted keys → nested dict)
        if cli_overrides:
            for dotted_key, value in cli_overrides.items():
                self._set_nested(dotted_key, value)

        # 4. Post-process computed fields
        self._post_process()

        # 5. Build flat lookup for backward-compatible attribute access
        self._flat: Dict[str, Any] = _flatten(self.raw)

    # ── attribute access (backward compat) ───────────────────────────────
    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion during init
        if name.startswith("_") or name == "raw":
            raise AttributeError(name)
        flat = self.__dict__.get("_flat", {})
        # Direct leaf-name lookup (e.g. opt.nurbs_weight_lr)
        candidates = [v for k, v in flat.items() if k.rsplit(".", 1)[-1] == name]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise AttributeError(
                f"Ambiguous config key '{name}' — matches: "
                f"{[k for k in flat if k.rsplit('.', 1)[-1] == name]}.  "
                f"Use the full dotted path."
            )
        raise AttributeError(f"NurbsConfig has no key '{name}'")

    def __getitem__(self, dotted_key: str) -> Any:
        return self._flat[dotted_key]

    def __contains__(self, dotted_key: str) -> bool:
        return dotted_key in self._flat

    # ── serialization ────────────────────────────────────────────────────
    def save(self, path: Union[str, Path]) -> None:
        """Dump full resolved config to YAML (for reproducibility)."""
        with open(path, "w") as f:
            yaml.dump(self.raw, f, default_flow_style=False, sort_keys=False)

    # ── internal helpers ─────────────────────────────────────────────────
    def _set_nested(self, dotted_key: str, value: Any) -> None:
        keys = dotted_key.split(".")
        d = self.raw
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def _post_process(self) -> None:
        """Apply derived / computed defaults after merge."""
        ref = self.raw.get("refinement", {})
        ref["residual_scaling"] = ref.get("residual_scaling", False) and ref.get("refine_scales", True)
        ref["residual_rots"] = ref.get("residual_rots", False) and ref.get("refine_rotations", True)
        ref["residual_opacity"] = ref.get("residual_opacity", False) and ref.get("refine_opacities", True)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the flat param dict."""
        return dict(self._flat)