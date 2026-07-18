import re

# Centralized regular expressions for PII detection
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+91|91|0)?[ -]?[6-9]\d{9}\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Regex to detect clause numbering at the beginning of a text segment
CLAUSE_NUM_PATTERN = re.compile(
    r"^\s*("
    r"\d+(?:\.\d+)*\.?"
    r"|Clause\s+\d+"
    r"|Section\s+\d+(?:\.\d+)*"
    r"|Article\s+[IVXLCDM\d]+"
    r")(?=\s|$)",
    re.IGNORECASE
)
