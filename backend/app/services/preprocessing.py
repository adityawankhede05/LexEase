import re
from app.utils.regex import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    AADHAAR_PATTERN,
    PAN_PATTERN,
    CLAUSE_NUM_PATTERN,
)
from app.schemas.document import ClauseSegment

class PreprocessingService:
    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Safely normalizes whitespace, line endings, and line-end hyphenations.
        """
        if not text:
            return ""

        # 1. Normalize line endings to standard LF (\n)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Remove line-wrap hyphenation (e.g. agree- \n ment -> agreement)
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        # 3. Collapse 3 or more consecutive newlines to exactly 2 newlines (preserve paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4. Collapse spaces and tabs (non-newline whitespaces) to a single space
        text = re.sub(r"[ \t]+", " ", text)

        # 5. Strip spaces around newlines to clean up indentation artifacts
        text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)

        # 6. Final strip of overall text
        return text.strip()

    @classmethod
    def mask_pii(cls, text: str) -> str:
        """
        Replaces sensitive Indian PII (Email, Phone, Aadhaar, PAN) with placeholders.
        """
        if not text:
            return ""

        text = EMAIL_PATTERN.sub("[EMAIL]", text)
        text = PHONE_PATTERN.sub("[PHONE]", text)
        text = AADHAAR_PATTERN.sub("[AADHAAR]", text)
        text = PAN_PATTERN.sub("[PAN]", text)
        return text

    @classmethod
    def segment_clauses(cls, text: str) -> list[ClauseSegment]:
        """
        Splits text on paragraph/clause boundaries, applies PII masking, 
        and returns a list of ClauseSegment Pydantic models.
        """
        if not text:
            return []

        lines = text.split("\n")
        segments: list[tuple[str | None, str]] = []
        current_clause_num: str | None = None
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue

            # Check if line begins with a clause numbering pattern
            match = CLAUSE_NUM_PATTERN.match(stripped)
            if match:
                # Save previous segment
                if current_lines:
                    raw_segment = "\n".join(current_lines).strip()
                    if raw_segment:
                        segments.append((current_clause_num, raw_segment))
                # Start new segment
                current_clause_num = match.group(1)
                current_lines = [line]
            else:
                current_lines.append(line)

        # Save the final segment
        if current_lines:
            raw_segment = "\n".join(current_lines).strip()
            if raw_segment:
                segments.append((current_clause_num, raw_segment))

        # Instantiate ClauseSegment objects after masking
        clause_segments: list[ClauseSegment] = []
        for index, (clause_num, raw_text) in enumerate(segments, start=1):
            # Ensure mask_pii is applied before constructing ClauseSegment
            masked_text = cls.mask_pii(raw_text)
            clause_segments.append(
                ClauseSegment(
                    clause_id=f"clause_{index}",
                    clause_number=clause_num,
                    text=masked_text
                )
            )

        return clause_segments
