"""Slot guard: a failure inside a slot is reported where the user can see it.

PySide6 prints the traceback of an exception raised in a slot and keeps running, so a
mistake in one handler no longer takes the process down - but the error lands on a
console that a windowed or packaged run does not have, and a handler that silently does
nothing is a defect the user cannot report.  Wrapping the user-interaction slots keeps
the traceback *and* routes the message to the log tab and the status bar.
"""
from __future__ import annotations

import functools
import inspect
import traceback


def safe_slot(func):
    """Decorator for Qt slots; reports failures instead of letting them vanish.

    Qt may pass signal arguments the slot does not declare, so the wrapper
    forwards only as many positional arguments as the method accepts.
    """
    arity = max(0, len([
        p for p in inspect.signature(func).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]) - 1)   # exclude ``self``: Qt often passes a signal argument the slot ignores

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args[:arity], **kwargs)
        except Exception as exc:  # noqa: BLE001 - last line of defence
            traceback.print_exc()
            logger = getattr(self, "log", None)
            if callable(logger):
                logger(f"[error] {func.__qualname__}: {type(exc).__name__}: {exc}")
            status = getattr(self, "statusBar", None)
            if callable(status):
                status().showMessage(f"{type(exc).__name__}: {exc}")
            return None

    return wrapper
