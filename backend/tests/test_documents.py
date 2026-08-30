import io
import fitz
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.preprocessing import PreprocessingService

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
    """Verify that a valid PDF file can be uploaded and text is extracted and preprocessed successfully."""
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
    assert "clauses" in data
    assert len(data["clauses"]) > 0
    assert "Standard legal agreement clause 1.1:" in data["clauses"][0]["text"]
    assert "extracted_text" not in data

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


def test_clean_text():
    """Verify safe normalization cleans spacing, newlines, and hyphens without stripping clause separators."""
    text = "  Hello   World!\r\nThis is a test- \n  word.   \n\n\n\nPreserved paragraph."
    cleaned = PreprocessingService.clean_text(text)
    assert cleaned == "Hello World!\nThis is a testword.\n\nPreserved paragraph."


def test_mask_pii():
    """Verify standard Indian PII (Email, Phone, Aadhaar, PAN) are masked properly."""
    text = "Email me at user@example.com or call 9876543210. Aadhaar: 1234-5678-9012. PAN: ABCDE1234F."
    masked = PreprocessingService.mask_pii(text)
    assert "user@example.com" not in masked
    assert "9876543210" not in masked
    assert "1234-5678-9012" not in masked
    assert "ABCDE1234F" not in masked
    assert "[EMAIL]" in masked
    assert "[PHONE]" in masked
    assert "[AADHAAR]" in masked
    assert "[PAN]" in masked


def test_segment_clauses():
    """Verify clause segmentation detects numbers and masks PII prior to ClauseSegment instantiation."""
    text = "Preamble paragraph text.\n\n1. First main clause.\nSome details. Email: test@test.com\n\n1.1 Sub-clause title\nMore text."
    clauses = PreprocessingService.segment_clauses(text)
    assert len(clauses) == 3
    
    assert clauses[0].clause_id == "clause_1"
    assert clauses[0].clause_number is None
    assert clauses[0].text == "Preamble paragraph text."
    
    assert clauses[1].clause_id == "clause_2"
    assert clauses[1].clause_number == "1."
    assert "First main clause." in clauses[1].text
    assert "[EMAIL]" in clauses[1].text  # Verifies PII masking is applied before ClauseSegment construction
    
    assert clauses[2].clause_id == "clause_3"
    assert clauses[2].clause_number == "1.1"
    assert "Sub-clause title" in clauses[2].text


def test_upload_and_preprocess_integration():
    """Verify that uploading a document preprocesses and masks it in the upload response."""
    legal_doc = (
        "This is a legal agreement.\n\n"
        "1. Parties involved.\n"
        "The first party is contact@firstparty.com with PAN ABCDE1234F.\n\n"
        "2. Payment Terms.\n"
        "Payment of Rs 10,000 shall be made to Aadhaar number 9876-5432-1098."
    )
    pdf_bytes = create_mock_pdf(legal_doc)
    response = client.post(
        "/documents/upload",
        files={"file": ("legal_doc.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["filename"] == "legal_doc.pdf"
    assert data["page_count"] == 1
    assert "clauses" in data
    assert "extracted_text" not in data
    clauses = data["clauses"]
    assert len(clauses) == 3
    
    # Preamble
    assert clauses[0]["clause_number"] is None
    assert "This is a legal agreement." in clauses[0]["text"]
    
    # Clause 1
    assert clauses[1]["clause_number"] == "1."
    assert "[EMAIL]" in clauses[1]["text"]
    assert "[PAN]" in clauses[1]["text"]
    assert "contact@firstparty.com" not in clauses[1]["text"]
    
    # Clause 2
    assert clauses[2]["clause_number"] == "2."
    assert "[AADHAAR]" in clauses[2]["text"]
    assert "9876-5432-1098" not in clauses[2]["text"]


def test_document_title_exclusion():
    """Verify that a non-substantive document title like 'Residential Rental Agreement' is excluded from segments."""
    text = (
        "Residential Rental Agreement\n\n"
        "This is a legal preamble.\n\n"
        "1. Monthly Rent.\n"
        "Tenant shall pay monthly rent of INR 25,000.\n\n"
        "2. Repairs.\n"
        "Tenant shall maintain ordinary repairs."
    )
    clauses = PreprocessingService.segment_clauses(text)
    
    # "Residential Rental Agreement" should be filtered out
    # "This is a legal preamble." has trailing period, so it is kept as a preamble.
    # The two numbered clauses are kept.
    # Total clauses should be 3 (Preamble + Rent + Repairs).
    assert len(clauses) == 3
    assert clauses[0].clause_id == "clause_1"
    assert clauses[0].text == "This is a legal preamble."
    assert clauses[1].clause_id == "clause_2"
    assert "Monthly Rent" in clauses[1].text
    assert clauses[2].clause_id == "clause_3"
    assert "Repairs" in clauses[2].text


