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
temporary sibling and atomic rename. ``StorageService`` applies the application
deletion policy: deleting samples or speakers cascades database relations and
removes unreferenced sample audio paths. Task segments retain their history with
registered speaker IDs set to NULL. Deleting a hotword set deactivates it and
retains immutable versions; ``purge_hotword_set`` is reserved for explicit
retention cleanup.
"""

from .database import SQLiteStore
from .files import FileStore, UnsafeStoragePath
from .service import FileCleanupError, StorageService

__all__ = ["SQLiteStore", "FileStore", "UnsafeStoragePath", "StorageService", "FileCleanupError"]
