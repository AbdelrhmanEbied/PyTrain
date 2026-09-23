from __future__ import annotations

from abc import abstractmethod
from typing import Any

from models.base import (
    BaseModel,
    InvalidParameterError,
    InvalidTaskError,
    ParameterSchema,
    Task,
    UnsupportedModelError,
)


class EstimatorModel(BaseModel):
    framework: str

    def __init__(self, model_name: str, task: Task, params: dict[str, Any] | None = None) -> None:
        self.model_name = model_name
        self.task = task
        self._params: dict[str, Any] = params.copy() if params else {}
        self._validate_task()
        self._validate_model_name()

    @abstractmethod
    def _model_registry(self) -> dict[str, Any]: ...

    @abstractmethod
    def _schema(self) -> dict[str, ParameterSchema]: ...

    def _valid_tasks(self) -> tuple[str, ...]:
        return ("classification", "regression")

    def _validate_task(self) -> None:
        if self.task not in self._valid_tasks():
            raise InvalidTaskError(
                f"Task {self.task!r} not supported by {self.model_name!r}. "
                f"Valid tasks: {self._valid_tasks()}"
            )

    def _validate_model_name(self) -> None:
        available = sorted(self._model_registry().keys())
        if self.model_name not in available:
            raise UnsupportedModelError(
                f"Unknown model {self.model_name!r}. Available: {available}"
            )

    def get_params(self) -> dict[str, Any]:
        return self._params.copy()

    def set_params(self, **params: Any) -> None:
        invalid = set(params.keys()) - set(self._schema().keys())
        if invalid:
            raise InvalidParameterError(f"Unknown parameters for {self.model_name!r}: {invalid}")
        self._validate_param_values(params)
        self._params.update(params)

    def _validate_param_values(self, params: dict[str, Any]) -> None:
        model_schema = self._schema()
        for name, value in params.items():
            schema = model_schema[name]
            if value is None and not schema.nullable:
                raise InvalidParameterError(f"Parameter {name!r} does not accept None")
            if value is None:
                continue
            if schema.choices is not None and value not in schema.choices:
                raise InvalidParameterError(
                    f"Parameter {name!r} must be one of {schema.choices}, got {value!r}"
                )
            if schema.min_value is not None and value < schema.min_value:
                raise InvalidParameterError(
                    f"Parameter {name!r} must be >= {schema.min_value}, got {value!r}"
                )
            if schema.max_value is not None and value > schema.max_value:
                raise InvalidParameterError(
                    f"Parameter {name!r} must be <= {schema.max_value}, got {value!r}"
                )

    def build(self) -> Any:
        return self._model_registry()[self.model_name](**self._params)

    def get_parameter_schema(self) -> list[ParameterSchema]:
        return list(self._schema().values())

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "framework": self.framework,
            "parameters": self.get_params(),
        }
