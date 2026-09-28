"""Dynamic task component loading.

Every task package exposes the same component modules.  This loader contains
no task-name registry: adding ``tasks/<name>`` is sufficient for ``--task
<name>`` to resolve it.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
import re


_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_COMPONENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def load_component(task_name: str, component: str) -> ModuleType:
    if _TASK_NAME.fullmatch(task_name) is None:
        raise ValueError(f"invalid task name: {task_name!r}")
    if _COMPONENT_NAME.fullmatch(component) is None:
        raise ValueError(f"invalid task component: {component!r}")
    try:
        return import_module(f"tasks.{task_name}.{component}")
    except ModuleNotFoundError as exc:
        expected = f"tasks.{task_name}"
        if exc.name == expected or exc.name == f"{expected}.{component}":
            raise ValueError(
                f"task {task_name!r} does not provide component {component!r}"
            ) from exc
        raise


__all__ = ["load_component"]
