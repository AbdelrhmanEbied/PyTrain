from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

Task = Literal["classification", "regression"]
ParamType = Literal["int", "float", "str", "bool"]


@dataclass
class ParameterSchema:
    name: str
    type: ParamType
    default: Any = None
    choices: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    nullable: bool = False
    description: str = ""
    group: str = "general"
    advanced: bool = False


class ModelError(Exception):
    pass


class UnsupportedModelError(ModelError):
    pass


class InvalidTaskError(ModelError):
    pass


class InvalidParameterError(ModelError):
    pass


class DependencyError(ModelError):
    pass


class BaseModel(ABC):
    model_name: str
    task: Task
    framework: str

    @abstractmethod
    def get_params(self) -> dict[str, Any]: ...

    @abstractmethod
    def set_params(self, **params: Any) -> None: ...

    @abstractmethod
    def build(self) -> Any: ...

    @abstractmethod
    def get_parameter_schema(self) -> list[ParameterSchema]: ...

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def list_available_models(cls) -> list[str]: ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"task={self.task!r}, "
            f"framework={self.framework!r})"
        )
