from pathlib import Path
from app.schemas.ai import AITask

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

class PromptBuilder:
    @classmethod
    def build_prompt(cls, task_type: AITask, payload: dict) -> str:
        """
        Builds and formats a prompt based on the AITask type and payload data.
        
        Args:
            task_type: The enum representing the task type.
            payload: A dictionary containing key-value inputs for the task.
            
        Returns:
            The formatted prompt string.
            
        Raises:
            ValueError: If a required payload parameter is missing or task_type is invalid.
            FileNotFoundError: If the prompt template file is missing.
        """
        if not isinstance(task_type, AITask):
            raise ValueError(f"task_type must be an instance of AITask. Got: {task_type}")

        if task_type == AITask.SIMPLIFICATION:
            clause_text = payload.get("clause_text") or payload.get("clause")
            if not clause_text:
                raise ValueError("Missing required key 'clause_text' in payload for simplification task.")
            
            template_path = PROMPTS_DIR / "simplification.txt"
            if not template_path.is_file():
                raise FileNotFoundError(f"Prompt template file not found at: {template_path}")
                
            template = template_path.read_text(encoding="utf-8")
            return template.replace("{{clause}}", str(clause_text))

        if task_type == AITask.DOCUMENT_SUMMARY:
            document_text = payload.get("document_text") or payload.get("document")
            if not document_text:
                raise ValueError("Missing required key 'document_text' in payload for document summary task.")

            template_path = PROMPTS_DIR / "document_summary.txt"
            if not template_path.is_file():
                raise FileNotFoundError(f"Prompt template file not found at: {template_path}")

            template = template_path.read_text(encoding="utf-8")
            return template.replace("{{document}}", str(document_text))

        if task_type == AITask.CLAUSE_ANALYSIS:
            clauses_json = payload.get("clauses_json")
            if not clauses_json:
                raise ValueError(
                    "Missing required key 'clauses_json' in payload for clause analysis task."
                )

            template_path = PROMPTS_DIR / "clause_risk.txt"
            if not template_path.is_file():
                raise FileNotFoundError(f"Prompt template file not found at: {template_path}")

            template = template_path.read_text(encoding="utf-8")
            return template.replace("{{clauses_json}}", str(clauses_json))

        raise ValueError(f"Unsupported AI task: {task_type}")
