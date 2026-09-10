import os
import sys
import asyncio
from dotenv import load_dotenv
from unittest.mock import patch

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load env variables
load_dotenv(os.path.join(project_root, ".env"))

# Set encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database.document_context_store import document_context_store
from app.database.migrations.init_db import init_db
from app.schemas.document import ClauseSegment
from app.services.qa import DocumentQAService

async def run_qa_regression_test():
    try:
        init_db()
    except Exception:
        pass
    print("================================================================================")
    print("RUNNING Q&A REGRESSION AND PARAPHRASE TESTS")
    print("================================================================================\n")
    
    # 1. Define the 5 clauses from "testing new model.pdf"
    clauses = [
        ClauseSegment(
            clause_id="clause_1",
            clause_number="1",
            text="Monthly rent: Tenant shall pay INR 25,000 per month on or before the 5th day of each month."
        ),
        ClauseSegment(
            clause_id="clause_2",
            clause_number="2",
            text="Repairs: Tenant shall maintain the premises in good condition and be responsible for reasonable repairs arising from ordinary use."
        ),
        ClauseSegment(
            clause_id="clause_3",
            clause_number="3",
            text="Termination: Company may terminate the Employee's employment immediately without notice and without payment of outstanding dues."
        ),
        ClauseSegment(
            clause_id="clause_4",
            clause_number="4",
            text="Indemnity: Vendor shall indemnify and hold harmless the Company against all losses, claims, damages, liabilities and expenses, without any limitation or cap."
        ),
        ClauseSegment(
            clause_id="clause_5",
            clause_number="5",
            text="Non-compete: Employee shall not engage in any competing business anywhere in India for three years after termination."
        )
    ]
    
    # Store clauses in the memory database and get a document_id
    doc_id = document_context_store.store(clauses)
    
    # Initialize QA service
    qa_service = DocumentQAService()
    
    # Define questions to test
    questions = [
        # Original 5 Questions
        {
            "q": "What is the monthly rent mentioned in the agreement?",
            "expected_source": "clause_1",
            "tag": "Monthly Rent (Original)"
        },
        {
            "q": "Who is responsible for ordinary repairs?",
            "expected_source": "clause_2",
            "tag": "Repairs (Original)"
        },
        {
            "q": "Can the company terminate the employee immediately?",
            "expected_source": "clause_3",
            "tag": "Termination (Original)"
        },
        {
            "q": "Is there any limit on the vendor's indemnity liability?",
            "expected_source": "clause_4",
            "tag": "Vendor Indemnity (Original)"
        },
        {
            "q": "What is the duration of the non-compete restriction?",
            "expected_source": "clause_5",
            "tag": "Non-compete Duration (Original)"
        },
        
        # Paraphrased Questions
        {
            "q": "How much does the tenant have to pay every month?",
            "expected_source": "clause_1",
            "tag": "Rent Payment (Paraphrase)"
        },
        {
            "q": "What is the deadline for rent?",
            "expected_source": "clause_1",
            "tag": "Rent Deadline (Paraphrase)"
        },
        {
            "q": "Who pays for normal repairs?",
            "expected_source": "clause_2",
            "tag": "Repairs Responsible (Paraphrase)"
        },
        {
            "q": "Does the company have to give notice before firing the employee?",
            "expected_source": "clause_3",
            "tag": "Termination Notice (Paraphrase)"
        },
        {
            "q": "Is the vendor's liability capped?",
            "expected_source": "clause_4",
            "tag": "Indemnity Cap (Paraphrase)"
        },
        {
            "q": "How long does the non-compete last?",
            "expected_source": "clause_5",
            "tag": "Non-compete Last (Paraphrase)"
        },
        {
            "q": "How broad is the non-compete restriction?",
            "expected_source": "clause_5",
            "tag": "Non-compete Breadth (Paraphrase)"
        }
    ]
    
    print(f"{'Question Tag':<35} | {'Source':<10} | {'Status':<6} | Answer Extract")
    print("-" * 110)
    
    all_pass = True
    
    for item in questions:
        q = item["q"]
        expected = item["expected_source"]
        
        # Run Q&A Service E2E (queries Groq for grounded answer!)
        response = await qa_service.answer_question(doc_id, q)
        
        # Check if the expected clause is in the source list
        is_pass = expected in response.source_clauses and not response.cannot_answer
        if not is_pass:
            all_pass = False
            
        status = "PASS" if is_pass else "FAIL"
        ans_preview = response.answer[:50].replace("\n", " ").strip() + "..."
        
        print(f"{item['tag']:<35} | {str(response.source_clauses):<10} | {status:<6} | {ans_preview}")
        
    print("\n================================================================================")
    if all_pass:
        print("VERIFICATION RESULT: ALL QA REGRESSION TESTS PASSED SUCCESSFULLY!")
    else:
        print("VERIFICATION RESULT: FAILURE (Some test cases failed).")
    print("================================================================================\n")
    
    assert all_pass

if __name__ == "__main__":
    asyncio.run(run_qa_regression_test())
