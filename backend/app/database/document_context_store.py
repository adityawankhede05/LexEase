import logging
import uuid
from typing import Callable
from sqlalchemy.orm import Session

from app.database.models import Clause, Document
from app.database.session import SessionLocal
from app.schemas.document import ClauseSegment

logger = logging.getLogger(__name__)


class DocumentContextStore:
    """
    Database-backed server-side storage for Document and ClauseSegment records.

    Preserves the modular public interface (store, get, delete) while persisting
    records via SQLAlchemy ORM.
    """

    def __init__(self, session_factory: Callable[[], Session] | None = None):
        self._session_factory = session_factory or SessionLocal
        # In-memory fallback in case of transient DB write failures
        self._fallback_store: dict[str, list[ClauseSegment]] = {}

    def _get_session(self) -> Session:
        return self._session_factory()

    def store(
        self,
        clauses: list[ClauseSegment],
        filename: str | None = None,
        page_count: int | None = None,
        character_count: int | None = None,
        document_id: str | None = None,
    ) -> str:
        """
        Stores a list of ClauseSegment objects and document metadata in the database,
        returning the generated or provided document_id.
        """
        doc_id = document_id or str(uuid.uuid4())
        fname = filename or "uploaded_document.pdf"
        p_count = page_count if page_count is not None else 1
        c_count = (
            character_count
            if character_count is not None
            else sum(len(c.text) for c in clauses)
        )

        try:
            with self._get_session() as session:
                # Upsert/Replace document if exists
                existing_doc = session.query(Document).filter(Document.id == doc_id).first()
                if existing_doc:
                    existing_doc.filename = fname
                    existing_doc.page_count = p_count
                    existing_doc.character_count = c_count
                    session.query(Clause).filter(Clause.document_id == doc_id).delete()
                else:
                    doc = Document(
                        id=doc_id,
                        filename=fname,
                        page_count=p_count,
                        character_count=c_count,
                    )
                    session.add(doc)

                for pos, clause in enumerate(clauses):
                    clause_entity = Clause(
                        document_id=doc_id,
                        clause_id=clause.clause_id,
                        clause_number=clause.clause_number,
                        text=clause.text,
                        position=pos,
                    )
                    session.add(clause_entity)

                session.commit()
                return doc_id
        except Exception as e:
            logger.error(f"Failed to persist document {doc_id} to database: {e}", exc_info=True)
            # Retain in fallback store so operations can proceed without losing context
            self._fallback_store[doc_id] = clauses
            return doc_id

    def get(self, document_id: str) -> list[ClauseSegment] | None:
        """
        Retrieves the list of ClauseSegment objects for a given document_id,
        or None if not found.
        """
        try:
            with self._get_session() as session:
                clauses = (
                    session.query(Clause)
                    .filter(Clause.document_id == document_id)
                    .order_by(Clause.position.asc())
                    .all()
                )
                if clauses:
                    return [
                        ClauseSegment(
                            clause_id=c.clause_id,
                            clause_number=c.clause_number,
                            text=c.text,
                        )
                        for c in clauses
                    ]
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id} from database: {e}", exc_info=True)

        return self._fallback_store.get(document_id)

    def delete(self, document_id: str) -> None:
        """
        Removes stored document and clause context for a given document_id.
        """
        try:
            with self._get_session() as session:
                session.query(Clause).filter(Clause.document_id == document_id).delete()
                session.query(Document).filter(Document.id == document_id).delete()
                session.commit()
        except Exception as e:
            logger.error(f"Failed to delete document {document_id} from database: {e}", exc_info=True)

        self._fallback_store.pop(document_id, None)


# Module-level singleton instance
document_context_store = DocumentContextStore()
