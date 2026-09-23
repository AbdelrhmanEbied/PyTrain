from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamConfig:
    grid: dict[str, list[Any]] = field(default_factory=dict)
    cv: int = 5
    scoring: str | None = None
    refit: bool = True
    n_jobs: int | None = None
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.cv < 2:
            raise ValueError("cv must be >= 2.")
        if not self.grid:
            raise ValueError("grid must not be empty.")
        for name, values in self.grid.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"grid[{name!r}] must be a non-empty list.")
