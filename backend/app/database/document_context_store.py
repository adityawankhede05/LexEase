import uuid
from app.schemas.document import ClauseSegment


class DocumentContextStore:
    """
    Temporary server-side storage for masked ClauseSegment lists.

    Designed with a modular public interface (store, get, delete) so that
    Sprint 9 can replace this in-memory implementation with PostgreSQL
    persistence without modifying callers or public APIs.
    """

    def __init__(self):
        self._store: dict[str, list[ClauseSegment]] = {}

    def store(self, clauses: list[ClauseSegment]) -> str:
        """
        Stores a list of ClauseSegment objects and returns a generated document_id.
        Only already-cleaned and PII-masked ClauseSegment data should be stored.
        """
        document_id = str(uuid.uuid4())
        self._store[document_id] = clauses
        return document_id

    def get(self, document_id: str) -> list[ClauseSegment] | None:
        """
        Retrieves the list of ClauseSegment objects for a given document_id,
        or None if not found.
        """
        return self._store.get(document_id)

    def delete(self, document_id: str) -> None:
        """
        Removes stored clause context for a given document_id.
        """
        self._store.pop(document_id, None)


# Module-level singleton instance
document_context_store = DocumentContextStore()
