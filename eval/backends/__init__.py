from eval.backends.base import BaseBackend, GenerationConfig, PromptRecord


_BACKEND_REGISTRY: dict[str, type[BaseBackend]] = {}


def register_backend(backend_cls: type[BaseBackend]) -> None:
    if not backend_cls.name:
        raise ValueError("Backend class must define a non-empty name")
    _BACKEND_REGISTRY[backend_cls.name] = backend_cls


def get_backend_class(name: str) -> type[BaseBackend]:
    try:
        return _BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backend: {name}. Supported backends: {', '.join(list_backends())}"
        ) from exc


def list_backends() -> list[str]:
    return sorted(_BACKEND_REGISTRY)


__all__ = [
    "BaseBackend",
    "GenerationConfig",
    "PromptRecord",
    "get_backend_class",
    "list_backends",
    "register_backend",
]


def _register_builtin_backends():
    try:
        from eval.backends.vllm_backend import VllmBackend

        register_backend(VllmBackend)
    except ImportError:
        pass

    try:
        from eval.backends.semantiq_backend import SemantiqBackend

        register_backend(SemantiqBackend)
    except ImportError:
        pass


_register_builtin_backends()
