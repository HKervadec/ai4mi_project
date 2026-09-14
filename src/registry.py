"""Name -> factory lookup so configs can select components by string.

Usage:
    @register("model", "enet")
    def build_enet(in_channels, num_classes, kernels=8): ...

    net = build("model", "enet", in_channels=1, num_classes=5, kernels=8)
"""
from typing import Any, Callable

_REGISTRIES: dict[str, dict[str, Callable]] = {}


def register(kind: str, name: str) -> Callable:
    def decorator(factory: Callable) -> Callable:
        entries = _REGISTRIES.setdefault(kind, {})
        if name in entries:
            raise KeyError(f"{kind} '{name}' registered twice")
        entries[name] = factory
        return factory
    return decorator


def available(kind: str) -> list[str]:
    return sorted(_REGISTRIES.get(kind, {}))


def build(kind: str, name: str, **kwargs: Any) -> Any:
    try:
        factory = _REGISTRIES[kind][name]
    except KeyError:
        raise KeyError(f"unknown {kind} '{name}', available: {available(kind)}") from None
    return factory(**kwargs)
