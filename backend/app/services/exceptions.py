class DocumentError(Exception):
    """Base exception for document processing errors."""
    pass

class EmptyFileError(DocumentError):
    """Raised when the uploaded file is empty."""
    pass

class InvalidTypeError(DocumentError):
    """Raised when the file format is not a valid PDF."""
    pass

class CorruptPDFError(DocumentError):
    """Raised when the PDF file is corrupt or unreadable by PyMuPDF."""
    pass

class NoExtractableTextError(DocumentError):
    """Raised when the PDF does not contain any extractable text."""
    pass

class EmptyClauseListError(DocumentError):
    """Raised when an empty list of clauses is provided for summarization."""
    pass

class SummaryGenerationError(DocumentError):
    """Raised when document summarization fails in the AI service layer."""
    pass

class ClauseAnalysisError(DocumentError):
    """Raised when clause risk analysis fails in the AI service layer."""
    pass

class DocumentContextNotFoundError(DocumentError):
    """Raised when no stored document context is found for the given document_id."""
    pass

class QAGenerationError(DocumentError):
    """Raised when document Q&A generation fails in the AI service layer."""
    pass

