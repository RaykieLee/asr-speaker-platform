"""SQLite and local-file storage for the ASR application.

``SQLiteStore`` owns schema initialization, migrations and repository operations.
Create it once per process, call :meth:`initialize`, then use its explicit
methods (``create_speaker``, ``create_job``, ``append_hotword_version``, etc.).
Each operation is protected by a re-entrant lock and committed as a transaction;
the shared SQLite connection is safe for threaded API handlers. Rows are
returned as dictionaries, JSON columns are decoded, and embedding BLOBs are
returned as lists of float32 values. The API intentionally leaves ASR-specific
validation and profile re-aggregation to higher layers.

``FileStore`` accepts only relative paths under a configured root. Writes use a
temporary sibling and atomic rename. Database cascades remove dependent rows:
deleting a speaker also deletes samples and its profile, while task segments
retain their history with registered speaker IDs set to NULL. File deletion is
available explicitly through ``FileStore.delete``; call it with paths returned
by the relevant query before/after deleting their owning rows according to the
application's retention policy.
"""

from .database import SQLiteStore
from .files import FileStore, UnsafeStoragePath

__all__ = ["SQLiteStore", "FileStore", "UnsafeStoragePath"]
