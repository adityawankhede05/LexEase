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
