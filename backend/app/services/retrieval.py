import re
from app.schemas.document import ClauseSegment


class ClauseRetrievalService:
    """
    Lightweight lexical relevance retrieval service for legal document clauses.

    Completely independent of Gemini, vector databases, or embedding models.
    Normalizes text, removes English stop words, scores clauses by term overlap,
    ranks results, and returns up to top_n relevant clauses.
    """

    STOP_WORDS: frozenset[str] = frozenset(
        {
            "a", "about", "above", "after", "again", "against", "all", "am", "an",
            "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
            "before", "being", "below", "between", "both", "but", "by", "can",
            "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
            "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
            "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
            "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
            "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
            "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
            "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
            "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
            "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
            "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
            "so", "some", "such", "than", "that", "that's", "the", "their",
            "theirs", "them", "themselves", "then", "there", "there's", "these",
            "they", "they'd", "they'll", "they're", "they've", "this", "those",
            "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
            "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
            "what's", "when", "when's", "where", "where's", "which", "while",
            "who", "who's", "whom", "why", "why's", "with", "won't", "would",
            "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
            "yours", "yourself", "yourselves", "shall", "will",
            # Common query template and framing terms
            "due", "amount", "much", "many", "tell", "explain", "state", "mention", "mentioned", "give", "find",
            "agreement", "agreements", "contract", "contracts", "document", "documents", "clause", "clauses"
        }
    )

    LEGAL_SYNONYMS: dict[str, str] = {
        "normal": "ordinary",
        "routine": "ordinary",
        "fired": "terminate",
        "firing": "terminate",
        "limit": "limitation",
        "cap": "limitation",
        "capped": "limitation",
        "deadline": "day",
        "duration": "year",
        "last": "year",
        "restriction": "compete",
        "broad": "anywhere",
    }

    def __init__(self, top_n: int = 5, min_score: float = 0.55):
        self.top_n = top_n
        self.min_score = min_score

    def retrieve(
        self, question: str, clauses: list[ClauseSegment]
    ) -> list[ClauseSegment]:
        """
        Retrieves top_n clauses ranked by lexical relevance to the question.

        Args:
            question: The user's question string.
            clauses: List of ClauseSegment objects to rank.

        Returns:
            Ranked list of up to top_n ClauseSegment objects matching with a score
            at or above min_score. Returns empty list if no clauses meet threshold.
        """
        if not question or not clauses:
            return []

        question_terms = self._tokenize_and_filter(question)
        if not question_terms:
            return []

        scored_clauses: list[tuple[float, int, ClauseSegment]] = []

        for index, clause in enumerate(clauses):
            clause_tokens = self._tokenize(clause.text)
            if not clause_tokens:
                continue

            # Count how many unique question terms appear in clause text
            matching_terms = sum(1 for term in question_terms if term in clause_tokens)
            score = matching_terms / len(question_terms)

            # Require relevance score at or above min_score threshold
            if score >= self.min_score:
                # Store negative index as tie-breaker to preserve document order
                scored_clauses.append((score, -index, clause))

            if not scored_clauses:
                continue

        if not scored_clauses:
            return []

        # Sort descending by score, then by document order
        scored_clauses.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return [clause for _, _, clause in scored_clauses[: self.top_n]]

    @classmethod
    def _normalize_legal_term(cls, term: str) -> str:
        """Maps query terms to standard contract vocabulary."""
        stemmed = cls._stem(term)
        return cls.LEGAL_SYNONYMS.get(term, cls.LEGAL_SYNONYMS.get(stemmed, stemmed))

    @classmethod
    def _stem(cls, word: str) -> str:
        """Lightweight suffix stripping for common English inflections."""
        word = word.lower()
        if len(word) <= 3:
            return word
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        if word.endswith("ing") and len(word) > 5:
            return word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            return word[:-2]
        elif word.endswith("ly") and len(word) > 4:
            return word[:-2]
        elif word.endswith("ment") and len(word) > 6:
            return word[:-4]
        elif word.endswith("able") and len(word) > 6:
            return word[:-4]
        elif word.endswith("tion") and len(word) > 6:
            return word[:-4]
        elif word.endswith("al") and len(word) > 4:
            return word[:-2]
        return word

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        """Normalizes text to lowercase and returns raw, stemmed, and legal-canonical tokens."""
        if not text:
            return set()
        raw_tokens = set(re.findall(r"\b[a-z0-9]+\b", text.lower()))
        tokens = set(raw_tokens)
        for t in raw_tokens:
            stemmed = cls._stem(t)
            tokens.add(stemmed)
            tokens.add(cls._normalize_legal_term(t))
            tokens.add(cls._normalize_legal_term(stemmed))
        return tokens

    @classmethod
    def _tokenize_and_filter(cls, text: str) -> set[str]:
        """Tokenizes text, filters out stop words and single-letter possessives, and normalizes legal terms."""
        if not text:
            return set()
        raw_tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        filtered = {
            term
            for term in raw_tokens
            if term not in cls.STOP_WORDS
            and cls._stem(term) not in cls.STOP_WORDS
            and len(term) > 1
        }
        return {cls._normalize_legal_term(term) for term in filtered}
