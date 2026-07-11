from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.schemas.document import DocumentUploadResponse
from app.services.document import DocumentService
from app.services.exceptions import (
    EmptyFileError,
    CorruptPDFError,
    NoExtractableTextError,
)

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Validate file extension
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )
    
    # 2. Validate MIME content_type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Only application/pdf is allowed."
        )
    
    try:
        # Read the file bytes
        pdf_bytes = await file.read()
        
        # Process the document
        return DocumentService.extract_text_from_pdf(filename, pdf_bytes)
        
    except EmptyFileError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except CorruptPDFError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except NoExtractableTextError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
