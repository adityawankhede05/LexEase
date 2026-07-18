from app.schemas.ai import AITask

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
        """
        if not isinstance(task_type, AITask):
            raise ValueError(f"task_type must be an instance of AITask. Got: {task_type}")

        if task_type == AITask.SIMPLIFICATION:
            clause_text = payload.get("clause_text")
            if not clause_text:
                raise ValueError("Missing required key 'clause_text' in payload for simplification task.")
            return f"Simplify this legal clause:\n\n{clause_text}"
            
        raise ValueError(f"Unsupported AI task: {task_type}")
