"""
DEPRECATED — three-process IPC runtime.

This module previously hosted the active ``IPCProtocol`` class. After the
heterogeneous refactor it has been replaced by:

  * :mod:`heterogeneous_protocol` — the active runtime
  * :mod:`legacy_ipc_stub` — preserved interface for audit / multi-host
    preview

For backwards compatibility with external scripts that import
``IPCProtocol`` from this module, we re-export the ``LegacyIPCStub`` class
under the historical name. A ``DeprecationWarning`` is emitted at import
time.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "Importing from src.parties.ipc_protocol is deprecated. "
    "Use src.parties.heterogeneous_protocol (active runtime) or "
    "src.parties.legacy_ipc_stub (audit/multi-host preview) instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the legacy stub under the historical name so external scripts
# that still ``import IPCProtocol`` keep working — but go through the
# stub's own deprecation warning as well.
from .legacy_ipc_stub import LegacyIPCStub as IPCProtocol  # noqa: E402,F401

# The worker entry points have always been public; re-export them so
# ``from src.parties.ipc_protocol import _worker_U_entry`` still works.
from .legacy_ipc_stub import (  # noqa: E402,F401
    _worker_U_entry,
    _worker_M_entry,
    _worker_S_entry,
)

__all__ = ["IPCProtocol", "_worker_U_entry", "_worker_M_entry", "_worker_S_entry"]
