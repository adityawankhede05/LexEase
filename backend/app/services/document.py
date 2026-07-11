import fitz
from app.schemas.document import DocumentUploadResponse
from app.services.exceptions import (
    EmptyFileError,
    CorruptPDFError,
    NoExtractableTextError,
)

class DocumentService:
    @staticmethod
    def extract_text_from_pdf(filename: str, pdf_bytes: bytes) -> DocumentUploadResponse:
        # 1. Validate empty file
        if not pdf_bytes or len(pdf_bytes) == 0:
            raise EmptyFileError("Uploaded file is empty.")
        
        # 2. Try parsing PDF with PyMuPDF
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise CorruptPDFError("Invalid or corrupted PDF file.") from e
        
        try:
            page_count = len(doc)
            if page_count == 0:
                raise CorruptPDFError("PDF file contains no pages.")
            
            extracted_text_list = []
            for page in doc:
                text = page.get_text()
                extracted_text_list.append(text)
            
            extracted_text = "".join(extracted_text_list)
            
            # 3. Check for empty extracted text
            if not extracted_text.strip():
                raise NoExtractableTextError("PDF contains no extractable text.")
            
            return DocumentUploadResponse(
                filename=filename,
                page_count=page_count,
                character_count=len(extracted_text),
                extracted_text=extracted_text
            )
        finally:
            doc.close()
