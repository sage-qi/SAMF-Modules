"""Small, local registries without dependencies on BatteryML's training code."""

from typing import Any, Dict, Mapping, Tuple, Type


class Registry:
    """Register module classes and construct them from explicit parameters."""

    def __init__(self, name: str):
        self.name = name
        self._modules: Dict[str, Type] = {}

    def register(self, name=None):
        def decorator(module_class):
            module_name = name or module_class.__name__
            if module_name in self._modules:
                raise ValueError(f"{module_name!r} is already registered in {self.name}.")
            self._modules[module_name] = module_class
            return module_class

        return decorator

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._modules))

    def build(self, name: str, parameters: Mapping[str, Any]):
        if name not in self._modules:
            available = ", ".join(self.names)
            raise ValueError(f"Unknown {self.name} module {name!r}. Available: {available}")
        return self._modules[name](**dict(parameters))


ENCODERS = Registry("encoder")
PREDICTORS = Registry("predictor")
