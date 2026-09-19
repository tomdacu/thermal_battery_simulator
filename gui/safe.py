"""Slot guard: an exception inside a PyQt6 slot aborts the process by default.

Wrapping user-interaction slots keeps a mistake in one handler from killing the
application: the traceback is printed and, when the window is available, shown
in the log tab and the status bar.
"""
from __future__ import annotations

import functools
import inspect
import traceback


def safe_slot(func):
    """Decorator for Qt slots; reports failures instead of aborting.

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
