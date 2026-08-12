"""One bounded scheduler for UI, API, and MCP generation requests."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator, TypeVar

from .runtime_config import CONFIG

T = TypeVar("T")


class QueueFullError(RuntimeError):
    pass


class TaskCancelledError(RuntimeError):
    pass


_pending_slots = threading.BoundedSemaphore(CONFIG.mcp_max_pending)
_executor = ThreadPoolExecutor(
    max_workers=CONFIG.gpu_concurrency,
    thread_name_prefix="imagegen-worker",
)


class _FairGenerationGate:
    """FIFO gate for the single process-global ComfyUI runtime."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._active = False
        self._cancelled_tickets: set[int] = set()

    def _skip_cancelled_locked(self) -> None:
        while not self._active and self._serving_ticket in self._cancelled_tickets:
            self._cancelled_tickets.remove(self._serving_ticket)
            self._serving_ticket += 1

    def acquire(self, cancel_event: threading.Event | None = None) -> None:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._cancelled_tickets.add(ticket)
                    self._skip_cancelled_locked()
                    self._condition.notify_all()
                    raise TaskCancelledError("任务已在等待 GPU 时取消。")

                self._skip_cancelled_locked()
                if ticket == self._serving_ticket and not self._active:
                    self._active = True
                    return

                # A cancellation event does not notify the gate, so poll it at
                # a low frequency. Jobs without cancellation remain fully
                # condition-driven.
                self._condition.wait(timeout=0.2 if cancel_event is not None else None)

    def release(self) -> None:
        with self._condition:
            self._active = False
            self._serving_ticket += 1
            self._skip_cancelled_locked()
            self._condition.notify_all()


_generation_gate = _FairGenerationGate()


@contextmanager
def generation_slot(cancel_event: threading.Event | None = None) -> Iterator[None]:
    """Serialize access to the shared in-process ComfyUI runtime by default."""

    _generation_gate.acquire(cancel_event)
    try:
        yield
    finally:
        _generation_gate.release()


def generation_guard(function: Callable[..., T]) -> Callable[..., T]:
    @wraps(function)
    def wrapped(*args, **kwargs):
        ui_inputs = kwargs.get("ui_inputs")
        if ui_inputs is None:
            ui_inputs = next(
                (value for value in args if isinstance(value, dict)), None
            )
        cancel_event = (
            ui_inputs.get("_cancel_event") if isinstance(ui_inputs, dict) else None
        )
        with generation_slot(cancel_event):
            if cancel_event is not None and cancel_event.is_set():
                raise TaskCancelledError("任务已在进入 GPU 前取消。")
            return function(*args, **kwargs)

    return wrapped


def submit_background(function: Callable[..., T], *args, **kwargs) -> Future[T]:
    """Submit an MCP job without creating an unbounded daemon thread."""

    if not _pending_slots.acquire(blocking=False):
        raise QueueFullError(
            f"任务队列已满（最多 {CONFIG.mcp_max_pending} 个待处理任务），请稍后再试。"
        )

    def run_and_release() -> T:
        try:
            return function(*args, **kwargs)
        finally:
            _pending_slots.release()

    return _executor.submit(run_and_release)
