"""Lifecycle of the extraction pool workers in `src.features.event_loader`.

A worker whose parent dies without running `atexit` must end on its own instead of blocking on
its task queue: on 2026-11-19 eleven orphaned workers held 7 GB until the machine ran out of
memory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from src.features import event_loader

ROOT = Path(__file__).resolve().parents[1]


class _Exited(Exception):
    pass


def test_watchdog_exits_once_the_parent_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    parents = iter([100, 100, 1])
    monkeypatch.setattr(event_loader.os, "getppid", lambda: next(parents))
    monkeypatch.setattr(event_loader.time, "sleep", lambda _s: None)
    codes = []

    def fake_exit(code: int) -> None:
        codes.append(code)
        raise _Exited

    monkeypatch.setattr(event_loader.os, "_exit", fake_exit)
    with pytest.raises(_Exited):
        event_loader._exit_when_orphaned(100, interval=0.0)
    assert codes == [0]


def test_watchdog_keeps_polling_while_the_parent_lives(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"sleep": 0}

    def fake_sleep(_s: float) -> None:
        calls["sleep"] += 1
        if calls["sleep"] == 5:
            raise _Exited

    monkeypatch.setattr(event_loader.os, "getppid", lambda: 100)
    monkeypatch.setattr(event_loader.time, "sleep", fake_sleep)
    monkeypatch.setattr(event_loader.os, "_exit", lambda code: pytest.fail("exited with a live parent"))
    with pytest.raises(_Exited):
        event_loader._exit_when_orphaned(100, interval=0.0)
    assert calls["sleep"] == 5


def test_executor_passes_the_parent_pid_to_each_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakePool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(event_loader, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(event_loader, "_EXECUTORS", {})
    event_loader._executor(3)
    assert captured["initializer"] is event_loader._init_worker
    assert captured["initargs"][1] == os.getpid()


def test_init_worker_without_parent_starts_no_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    started = []
    monkeypatch.setattr(event_loader.threading.Thread, "start", lambda self: started.append(self.name))
    monkeypatch.setattr(event_loader.os, "environ", dict(os.environ))
    event_loader._init_worker(1)
    assert started == []
    event_loader._init_worker(1, parent_pid=os.getpid())
    assert started == ["parent-watchdog"]


def test_spawned_worker_dies_after_its_parent_is_killed(tmp_path: Path) -> None:
    """A real spawn worker started by a stand-in parent; the parent is SIGKILLed, the worker ends.

    Linux only (reads /proc), like the rest of the project.
    """
    pid_file = tmp_path / "worker.pid"
    parent = textwrap.dedent(f"""
        import multiprocessing, os, sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from src.features.event_loader import _exit_when_orphaned

        def child(parent_pid, path):
            import threading
            threading.Thread(target=_exit_when_orphaned, args=(parent_pid, 0.2), daemon=True).start()
            open(path, "w").write(str(os.getpid()))
            time.sleep(120)

        if __name__ == "__main__":
            ctx = multiprocessing.get_context("spawn")
            ctx.Process(target=child, args=(os.getpid(), {str(pid_file)!r})).start()
            time.sleep(120)
    """)
    script = tmp_path / "parent.py"
    script.write_text(parent)
    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        deadline = time.time() + 60
        while not pid_file.exists() or not pid_file.read_text():
            assert time.time() < deadline, "worker never started"
            time.sleep(0.1)
        worker = int(pid_file.read_text())
        proc.kill()
        proc.wait()
        deadline = time.time() + 10
        while Path(f"/proc/{worker}").exists():
            stat = Path(f"/proc/{worker}/stat").read_text() if Path(f"/proc/{worker}/stat").exists() else ""
            if stat.split(") ")[-1].startswith("Z"):
                break
            assert time.time() < deadline, "worker outlived its parent"
            time.sleep(0.1)
    finally:
        proc.kill()
