export const mockDocument = {
  name: "Service_Agreement_NDA_Redacted.pdf",
  uploadedAt: "August 2, 2026, 6:35 PM",
  status: "Analyzed",
  fileSize: "1.2 MB",
  pageCount: 5,
};

export const mockClauses = [
  {
    id: 1,
    number: "Section 4.2",
    title: "Survival of Obligations",
    original: "The obligations of confidentiality, non-disclosure, and non-use contained within this Agreement shall survive the expiration or termination of this Agreement for a period of ten (10) years from the date of such termination or expiration, or in perpetuity with respect to any Trade Secrets of the Disclosing Party.",
    simplified: "The duty to keep secrets lasts for 10 years after the contract ends. However, if the information is classified as a 'trade secret', you must keep it secret forever.",
  },
  {
    id: 2,
    number: "Section 8.1",
    title: "Unilateral Indemnification",
    original: "The Receiving Party agrees to indemnify, defend, and hold harmless the Disclosing Party and its affiliates, officers, directors, and employees, from and against any and all claims, liabilities, lawsuits, damages, costs, or expenses (including reasonable attorneys' fees) arising out of or resulting from any breach of this agreement.",
    simplified: "If you break this agreement, you have to cover all legal costs, damages, and lawsuits for the other company and their staff. This obligation is one-sided; they don't have to cover your costs if they breach it.",
  },
  {
    id: 3,
    number: "Section 11.4",
    title: "Governing Law & Dispute Resolution",
    original: "This Agreement shall be governed by, construed and enforced in accordance with the laws of the State of Delaware. Any dispute, controversy or claim arising out of this Agreement shall be settled exclusively by final and binding arbitration in Dover, Delaware, under the AAA Commercial Rules, with no right of appeal.",
    simplified: "Any legal fights will be decided under Delaware law, not your home state's laws. You must resolve arguments through private arbitration in Delaware, and you cannot appeal the arbitrator's decision.",
  }
];

export const mockRisks = {
  score: 75,
  rating: "High",
  summary: "This agreement presents elevated risk levels due to unilateral indemnity obligations and a perpetual confidentiality hold on trade secrets. Immediate legal review is advised.",
  items: [
    {
      id: "risk-1",
      severity: "High",
      category: "Indemnification",
      title: "One-Sided Indemnity Obligation",
      description: "You are solely responsible for covering all expenses/damages for any breaches, without reciprocal protection from the contractor.",
      remedy: "Request a mutual indemnification clause where both parties protect each other for breaches.",
    },
    {
      id: "risk-2",
      severity: "Medium",
      category: "Confidentiality",
      title: "Perpetual Trade Secret Protection",
      description: "An indefinite obligation for trade secrets can expose you to liability long after the business relationship has concluded.",
      remedy: "Seek a maximum duration of 5-7 years for all confidential items, or define trade secrets restrictively.",
    },
    {
      id: "risk-3",
      severity: "Low",
      category: "Jurisdiction",
      title: "Exclusive Out-of-State Jurisdiction",
      description: "Disputes must be settled through binding arbitration in Wilmington/Dover, Delaware, which can significantly raise legal costs.",
      remedy: "Request a local jurisdiction or mutual agreement venue for arbitration.",
    }
  ]
};
