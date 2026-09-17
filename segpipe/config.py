"""YAML config loading with `base:` inheritance and `--set key=value` overrides."""

from pathlib import Path

import yaml


class Config(dict):
    """A dict that also allows attribute access (cfg.train.seed)."""

    def __init__(self, data=None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = _wrap(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(value):
    if isinstance(value, dict) and not isinstance(value, Config):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value):
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_file(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("base", None)
    if base is None:
        return data
    return deep_merge(_load_file(path.parent / base), data)


def apply_override(cfg: dict, item: str) -> None:
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"override '{item}' must look like key=value")
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        if not isinstance(node.get(part), dict):
            raise KeyError(f"unknown config section '{part}' in override '{item}'")
        node = node[part]
    if leaf not in node:
        print(f"> Warning: override adds a new key '{key}' (typo?)")
    node[leaf] = yaml.safe_load(raw)


def load_config(path, overrides=()) -> Config:
    path = Path(path)
    cfg = _load_file(path)
    for item in overrides or ():
        apply_override(cfg, item)
    cfg.setdefault("experiment", path.stem)
    return Config(cfg)


def save_config(cfg, path) -> None:
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
