import io
import fitz
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def create_mock_pdf(text: str = "Hello, this is a test PDF content for LexEase.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def create_empty_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def test_health_endpoint():
    """Verify that the existing /health endpoint is preserved and works correctly."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "version": "0.1.0",
        "service": "LexEase Backend"
    }

def test_upload_valid_pdf():
    """Verify that a valid PDF file can be uploaded and text is extracted successfully."""
    pdf_text = "Standard legal agreement clause 1.1: The party of the first part..."
    pdf_bytes = create_mock_pdf(pdf_text)
    
    response = client.post(
        "/documents/upload",
        files={"file": ("agreement.pdf", pdf_bytes, "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "agreement.pdf"
    assert data["page_count"] == 1
    assert pdf_text in data["extracted_text"]
    assert data["character_count"] == len(data["extracted_text"])

def test_upload_invalid_extension():
    """Verify that uploading a file with an invalid extension returns 400 Bad Request."""
    response = client.post(
        "/documents/upload",
        files={"file": ("agreement.txt", b"some plain text", "application/pdf")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."

def test_upload_invalid_content_type():
    """Verify that uploading a PDF with an incorrect MIME content type returns 400."""
    pdf_bytes = create_mock_pdf()
    response = client.post(
        "/documents/upload",
        files={"file": ("agreement.pdf", pdf_bytes, "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid content type. Only application/pdf is allowed."

def test_upload_empty_file():
    """Verify that uploading an empty PDF file returns 400 Bad Request."""
    response = client.post(
        "/documents/upload",
        files={"file": ("agreement.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."

def test_upload_corrupted_pdf():
    """Verify that uploading a corrupted or malformed PDF returns 400 Bad Request."""
    corrupted_bytes = b"PDF-1.4 %not-a-real-pdf-header-but-garbage-bytes"
    response = client.post(
        "/documents/upload",
        files={"file": ("agreement.pdf", corrupted_bytes, "application/pdf")}
    )
    assert response.status_code == 400
    assert "Invalid or corrupted PDF file." in response.json()["detail"]

def test_upload_no_extractable_text_pdf():
    """Verify that uploading a PDF with no extractable text returns 400 Bad Request."""
    empty_pdf_bytes = create_empty_pdf()
    response = client.post(
        "/documents/upload",
        files={"file": ("agreement.pdf", empty_pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "PDF contains no extractable text."
