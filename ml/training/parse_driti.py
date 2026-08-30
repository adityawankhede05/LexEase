import os
import re
import pandas as pd

def parse_risky_patterns(filepath):
    """
    Parses risky_patterns.txt and extracts sections, patterns, examples, risks, recommendations.
    Maps them to: clause_text, clause_type, domain, risk_level, source, is_synthetic.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into sections by SECTION X:
    sections = re.split(r'={5,}\nSECTION \d+:\s*(.*?)\n={5,}', text)
    
    records = []
    
    header = sections[0]
    for idx in range(1, len(sections), 2):
        sec_name = sections[idx].strip()
        sec_content = sections[idx+1]
        
        # Determine domain and clause_type based on section name
        domain = "General Indian"
        clause_type = sec_name
        
        if "NON-COMPETE" in sec_name.upper():
            domain = "Employment"
            clause_type = "Non-Compete"
        elif "TERMINATION" in sec_name.upper():
            domain = "General Indian"
            clause_type = "Termination"
        elif "INDEMNITY" in sec_name.upper():
            domain = "General Indian"
            clause_type = "Indemnity"
        elif "LIABILITY" in sec_name.upper():
            domain = "General Indian"
            clause_type = "Limitation of Liability"
        elif "CONFIDENTIALITY" in sec_name.upper() or "NDA" in sec_name.upper():
            domain = "NDA"
            clause_type = "Confidentiality"
        elif "ARBITRATION" in sec_name.upper():
            domain = "General Indian"
            clause_type = "Arbitration"
        elif "INTELLECTUAL PROPERTY" in sec_name.upper():
            domain = "Licensing"
            clause_type = "Intellectual Property"
        elif "GOVERNING LAW" in sec_name.upper():
            domain = "General Indian"
            clause_type = "Governing Law"
        elif "PAYMENT" in sec_name.upper():
            domain = "Vendor/Supply"
            clause_type = "Rent" if "RENT" in sec_name.upper() else "Payment"
        elif "FORCE MAJEURE" in sec_name.upper():
            domain = "General Indian"
            clause_type = "Force Majeure"
        elif "AUTO-RENEWAL" in sec_name.upper():
            domain = "General Indian"
            clause_type = "Renewal Term"
        elif "DATA PROTECTION" in sec_name.upper():
            domain = "General Indian"
            clause_type = "Data Protection"

        # Split into HIGH RISK and MEDIUM RISK patterns
        high_part = ""
        med_part = ""
        
        parts = re.split(r'--- (HIGH RISK PATTERNS|MEDIUM RISK PATTERNS) ---', sec_content)
        current_risk = "HIGH"
        
        for p_idx in range(1, len(parts), 2):
            risk_header = parts[p_idx].strip()
            risk_content = parts[p_idx+1]
            risk_val = "HIGH" if "HIGH" in risk_header else "MEDIUM"
            
            # Find all patterns in this block
            # PATTERN: ... \nEXAMPLE: ... \nRISK: ... \nRECOMMENDED: ...
            pattern_blocks = re.split(r'PATTERN:\s*', risk_content)
            for pb in pattern_blocks[1:]:
                lines = pb.strip().split('\n')
                pattern_desc = lines[0].strip()
                pb_text = '\n'.join(lines[1:])
                
                # Extract Example
                ex_match = re.search(r'EXAMPLE:\s*(.*?)(?=\nRISK:|\nRECOMMENDED:|$)', pb_text, re.DOTALL)
                risk_match = re.search(r'RISK:\s*(.*?)(?=\nRECOMMENDED:|$)', pb_text, re.DOTALL)
                rec_match = re.search(r'RECOMMENDED:\s*(.*?)(?=$)', pb_text, re.DOTALL)
                
                example_text = ex_match.group(1).strip() if ex_match else ""
                risk_desc = risk_match.group(1).strip() if risk_match else ""
                rec_desc = rec_match.group(1).strip() if rec_match else ""
                
                # Remove quotes from example
                example_text = example_text.strip('"').strip("'")
                
                if example_text:
                    records.append({
                        "clause_text": example_text,
                        "clause_type": clause_type,
                        "domain": domain,
                        "risk_level": risk_val,
                        "source": "d-riti/Risky Patterns clause.pdf",
                        "is_synthetic": False
                    })
                    
                    # Generate a matched LOW risk clause based on the recommendation if possible
                    low_text = generate_low_risk_clause(clause_type, pattern_desc, rec_desc)
                    if low_text:
                        records.append({
                            "clause_text": low_text,
                            "clause_type": clause_type,
                            "domain": domain,
                            "risk_level": "LOW",
                            "source": "d-riti/Risky Patterns clause.pdf (derived recommended)",
                            "is_synthetic": True # derived/synthetic but contextually accurate
                        })

    return pd.DataFrame(records)

def generate_low_risk_clause(clause_type, pattern_desc, rec_desc):
    """
    Infers a realistic LOW-risk clause based on recommendations.
    """
    rec_lower = rec_desc.lower()
    pat_lower = pattern_desc.lower()
    
    if clause_type == "Non-Compete":
        if "employment only" in rec_lower or "during employment" in rec_lower:
            return "During the term of employment, the Employee shall not directly engage in any business competitive with the Employer."
        if "limit to city" in rec_lower or "worked" in rec_lower:
            return "For a period of 6 months following termination, Employee shall not compete with the Company within the city of Mumbai."
        if "6 months maximum" in rec_lower:
            return "For a period of 6 months post-termination, the Employee shall not solicit any client with whom they had direct business contact."
        return "During the term of employment, the Employee shall not work for any direct competitor of the Employer."
        
    elif clause_type == "Termination":
        if "30 days" in rec_lower or "notice" in rec_lower:
            return "Either party may terminate this agreement at any time by providing at least 30 days prior written notice to the other party."
        if "mutual" in rec_lower:
            return "Either party may terminate this agreement for convenience upon 30 days written notice. Both parties retain mutual rights of termination."
        if "dues earned" in rec_lower or "dues" in rec_lower:
            return "Upon termination of this agreement, the Company shall pay all legally earned and outstanding dues to the Vendor within 7 days."
        return "This Agreement may be terminated by either party upon 30 days written notice to the other party."
        
    elif clause_type == "Indemnity":
        if "cap" in rec_lower or "contract value" in rec_lower:
            return "Vendor shall indemnify the Company for direct losses arising from breach of contract, capped at the total contract value."
        if "mutual" in rec_lower:
            return "Each party shall indemnify, defend, and hold harmless the other party from and against any direct third-party claims arising out of its negligence."
        if "exclude indirect" in rec_lower or "consequential" in rec_lower:
            return "The indemnity obligation shall exclude any indirect, consequential, special, punitive, or incidental damages."
        return "Vendor shall indemnify Company against direct third-party claims, subject to a cap equal to the fees paid in the preceding 12 months."
        
    elif clause_type == "Limitation of Liability":
        if "cap total" in rec_lower or "contract value" in rec_lower:
            return "Except for fraud or willful misconduct, either party's maximum aggregate liability shall be limited to the total fees paid under this contract."
        if "minimum liability" in rec_lower or "maintain" in rec_lower:
            return "The total liability of either party for all claims arising under this Agreement shall be limited to the total contract value."
        if "carve out" in rec_lower or "gross negligence" in rec_lower:
            return "Neither party's liability is limited in cases of gross negligence, willful misconduct, or fraud, but is otherwise capped at the contract value."
        return "Each party's liability under this Agreement shall be limited to the total contract price."
        
    elif clause_type == "Confidentiality":
        if "3-5 years" in rec_lower or "time limit" in rec_lower:
            return "The receiving party shall keep all confidential information secret for a period of 3 years following the termination of this Agreement."
        if "marked as confidential" in rec_lower or "specifically" in rec_lower:
            return "Confidential Information refers to proprietary information disclosed in writing and clearly marked as confidential."
        if "legally required" in rec_lower or "carve-out" in rec_lower:
            return "Confidentiality obligations shall not apply to information required to be disclosed by law, provided prior notice is given to the disclosing party."
        if "return" in rec_lower:
            return "Upon termination, the receiving party shall return or certifiably destroy all confidential documents of the disclosing party within 30 days."
        return "The Receiving Party agrees to maintain the confidentiality of the Disclosing Party's proprietary information for 3 years post-termination."
        
    elif clause_type == "Arbitration":
        if "seat" in rec_lower or "india" in rec_lower:
            return "Any dispute arising out of this contract shall be settled by arbitration in Mumbai in accordance with the Arbitration Act, 1996."
        if "mutually appointed" in rec_lower or "sole arbitrator" in rec_lower:
            return "The parties shall mutually appoint a sole arbitrator to resolve all disputes arising under this agreement."
        if "12 month" in rec_lower or "time limit" in rec_lower:
            return "The arbitral tribunal shall endeavor to complete proceedings and publish the award within 12 months of constitution."
        return "Any dispute shall be referred to arbitration in New Delhi under the rules of the Indian Arbitration and Conciliation Act."
        
    elif clause_type == "Intellectual Property":
        if "specifically created" in rec_lower or "limit assignment" in rec_lower:
            return "All intellectual property created specifically for this project shall assign to the Company upon receipt of full payment."
        if "pre-existing" in rec_lower or "carve out" in rec_lower:
            return "Each party retains ownership of its pre-existing intellectual property. Vendor grants Company a license to use pre-existing tools solely for the project."
        if "moral rights" in rec_lower:
            return "The Vendor assigns the copyright of the deliverables to the Company, excluding moral rights which remain with the author."
        return "All intellectual property rights developed solely under this project shall be assigned to the Client upon final payment."
        
    elif clause_type == "Governing Law":
        if "courts at" in rec_lower or "jurisdiction" in rec_lower:
            return "This Agreement shall be governed by Indian law, and the courts at Mumbai shall have exclusive jurisdiction."
        return "This Agreement shall be governed by and construed in accordance with the laws of India."
        
    elif clause_type == "Payment":
        if "45 days" in rec_lower or "msme" in rec_lower:
            return "Payments shall be made within 45 days of receipt of a valid invoice, in compliance with the MSMED Act, 2006."
        if "interest" in rec_lower or " आरबीआई" in rec_lower or "bank rate" in rec_lower:
            return "Delayed payments shall attract interest at the RBI bank rate plus 2% per annum."
        if "disputed amounts" in rec_lower:
            return "In case of an invoice dispute, only the specific disputed amount may be withheld. All undisputed amounts must be paid when due."
        return "Payment shall be made within 30 days of receipt of invoice."
        
    elif clause_type == "Force Majeure":
        if "natural disasters" in rec_lower or "unforeseeable" in rec_lower:
            return "Force Majeure includes natural disasters, acts of God, war, riot, and government actions beyond the control of either party."
        if "notice within" in rec_lower:
            return "A party claiming force majeure must notify the other party in writing within 7 days of the occurrence of the event."
        if "beyond 30-60 days" in rec_lower or "termination" in rec_lower:
            return "If a Force Majeure event continues to prevent performance for more than 45 days, either party may terminate the contract."
        return "Force Majeure includes events beyond the reasonable control of the parties, including natural disasters and government orders."
        
    elif clause_type == "Renewal Term":
        if "30 days" in rec_lower or "notice" in rec_lower:
            return "This Agreement shall automatically renew for successive 1-year terms unless either party gives 30 days prior written notice of non-renewal."
        if "agreed in writing" in rec_lower:
            return "Upon renewal, any price escalation shall be negotiated and agreed in writing by both parties."
        return "The term of this Agreement may be extended by mutual written consent of both parties."
        
    elif clause_type == "Data Protection":
        if "dpdp act" in rec_lower or "security" in rec_lower:
            return "The parties shall comply with all applicable data protection laws, including the Digital Personal Data Protection Act, 2023."
        if "notification within 72" in rec_lower:
            return "In the event of a personal data breach, the processor shall notify the data fiduciary within 72 hours of discovery."
        return "The Processor shall maintain reasonable security practices to protect personal data from unauthorized access."

    return None

if __name__ == "__main__":
    df = parse_risky_patterns("risky_patterns.txt")
    print("Parsed records count:", len(df))
    print(df.groupby(["clause_type", "risk_level"]).size())
    df.to_csv("driti_parsed_risky_patterns.csv", index=False)
