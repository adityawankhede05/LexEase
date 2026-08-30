import os
import re

class LegalDomainDetector:
    def __init__(self):
        # Keyword sets for rule-based domain detection
        self.rules = {
            "Rental/Lease": [
                r"\brent\b", r"\btenant\b", r"\blandlord\b", r"\blease\b", r"\bsublet\b",
                r"\bsecurity deposit\b", r"\bpremises\b", r"\blessor\b", r"\blessee\b",
                r"\btenancy\b", r"\brental\b"
            ],
            "Employment": [
                r"\bemployee\b", r"\bemployer\b", r"\bsalary\b", r"\bprobation\b",
                r"\bnon-compete\b", r"\bcompensation\b", r"\bworking hours\b",
                r"\bprovident fund\b", r"\bgratuity\b", r"\bemployment\b"
            ],
            "Property": [
                r"\bstamp duty\b", r"\bsale consideration\b", r"\bpossession\b",
                r"\bregistration\b", r"\bsub-registrar\b", r"\bencumbrance\b",
                r"\bpurchaser\b", r"\bvendor\b", r"\bproperty\b", r"\bsale deed\b"
            ],
            "NDA/Confidentiality": [
                r"\bnon-disclosure\b", r"\bnda\b", r"\bconfidential information\b",
                r"\bdisclosing party\b", r"\breceiving party\b", r"\bconfidentiality\b"
            ],
            "Finance/Loan": [
                r"\blender\b", r"\bborrower\b", r"\bloan\b", r"\binterest rate\b",
                r"\bcollateral\b", r"\brepayment\b", r"\bguarantor\b"
            ],
            "Partnership/Shareholder": [
                r"\bshareholder\b", r"\bpartnership\b", r"\bshares\b", r"\bdividend\b",
                r"\bequity\b", r"\bpartner\b"
            ],
            "Service": [
                r"\bservice provider\b", r"\bclient\b", r"\bservice level\b", r"\bsla\b",
                r"\bdeliverables\b", r"\bconsulting\b"
            ]
        }

    def detect_domain(self, text: str) -> str:
        text_lower = text.lower()
        domain_counts = {}
        
        for domain, patterns in self.rules.items():
            count = 0
            for pattern in patterns:
                count += len(re.findall(pattern, text_lower))
            if count > 0:
                domain_counts[domain] = count
                
        if not domain_counts:
            return "Corporate/Commercial"
            
        # Return the domain with the highest keyword frequency
        best_domain = max(domain_counts, key=domain_counts.get)
        return best_domain
