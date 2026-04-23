"""
Lightweight standalone utilities shared across the framework.

Provides a minimal RunBatchProgressManager (rich-based progress display) and
save_traj (JSON trajectory serializer) so that framework code does not depend on
private submodules of mini-swe-agent that are not part of its public release.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from rich.console import Group
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


class RunBatchProgressManager:
    """Thread-safe per-instance status display for parallel batch runs."""

    def __init__(self, total: int, status_file: Optional[Path] = None):
        self._total = total
        self._done = 0
        self._statuses: dict[str, str] = {}
        self._lock = threading.Lock()
        self._status_file = status_file
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        self._task_id = self._progress.add_task("Instances", total=total)

    def on_instance_start(self, instance_id: str) -> None:
        with self._lock:
            self._statuses[instance_id] = "running"

    def update_instance_status(self, instance_id: str, status: str) -> None:
        with self._lock:
            self._statuses[instance_id] = status

    def on_instance_end(self, instance_id: str, message: str = "") -> None:
        with self._lock:
            self._statuses[instance_id] = f"done: {message}"
            self._done += 1
        self._progress.advance(self._task_id)
        if self._status_file:
            self._flush()

    def on_uncaught_exception(self, instance_id: str, exc: Exception) -> None:
        with self._lock:
            self._statuses[instance_id] = f"error: {exc}"
            self._done += 1
        self._progress.advance(self._task_id)

    def _flush(self) -> None:
        if not self._status_file:
            return
        try:
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = dict(self._statuses)
            self._status_file.write_text(
                json.dumps({"done": self._done, "total": self._total, "statuses": data}, indent=2)
            )
        except Exception:
            pass

    @property
    def render_group(self) -> Group:
        with self._lock:
            running = {k: v for k, v in self._statuses.items() if not v.startswith("done")}
        table = Table.grid(padding=(0, 1))
        for iid, status in list(running.items())[:10]:
            table.add_row(f"[cyan]{iid}[/cyan]", status)
        return Group(self._progress, table)


def save_traj(
    agent,
    path: Path,
    *,
    exit_status: str,
    result: str,
    extra_info: Optional[dict] = None,
    instance_id: str = "",
) -> None:
    """Serialize an agent trajectory (or a summary dict) to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "instance_id": instance_id,
        "exit_status": exit_status,
        "result": result,
        "timestamp": time.time(),
    }
    if extra_info:
        payload.update(extra_info)
    if agent is not None and hasattr(agent, "messages"):
        payload["messages"] = agent.messages
    if agent is not None and hasattr(agent, "model") and hasattr(agent.model, "cost"):
        payload["cost"] = agent.model.cost
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
