import os
import sys
import json
import torch

# Ensure project root is in python path
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(ml_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.inference.predict import predict_clause

def run_inference_tests():
    # 10 Representative contract clauses for testing
    test_cases = [
        {
            "category": "Liability",
            "text": "Neither party shall be liable to the other for any indirect, incidental, special, or consequential damages arising out of this agreement."
        },
        {
            "category": "Indemnity",
            "text": "The Seller shall indemnify, defend, and hold harmless the Buyer from and against any claims, losses, or expenses resulting from third-party IP infringement."
        },
        {
            "category": "Termination",
            "text": "Either party may terminate this agreement immediately if the other party breaches any material term and fails to cure such breach within 15 days."
        },
        {
            "category": "Governing Law",
            "text": "This agreement shall be governed by, and construed in accordance with, the laws of India, and the courts of Mumbai shall have exclusive jurisdiction."
        },
        {
            "category": "License/IP",
            "text": "Licensor hereby grants Licensee a non-exclusive, non-transferable, revocable license to use the Software solely for internal business operations."
        },
        {
            "category": "Non-Compete/Non-Solicit",
            "text": "The Employee shall not, for a period of 2 years post-termination, engage in any competitive business activity or solicit any clients of the Employer."
        },
        {
            "category": "Payment/Minimum Commitment",
            "text": "The Client agrees to pay the service provider a minimum monthly commitment of INR 50,000 for the duration of the service contract."
        },
        {
            "category": "Warranty",
            "text": "The Developer warrants that the software deliverables will perform substantially in accordance with the specifications for a period of 90 days."
        },
        {
            "category": "Assignment/Change of Control",
            "text": "Neither party may assign or transfer its rights under this agreement to any third party without the prior written consent of the other party."
        },
        {
            "category": "Force Majeure",
            "text": "Neither party shall be responsible for any delay or failure in performance resulting from acts beyond their reasonable control, including acts of God, war, or epidemics."
        }
    ]

    # Check device availability
    device_name = "CUDA (GPU: RTX 3050)" if torch.cuda.is_available() else "CPU"
    
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("LEXEASE LEGAL-BERT TRANSFORMER INFERENCE TEST REPORT")
    output_lines.append("=" * 70)
    output_lines.append(f"Inference Device: {device_name}")
    output_lines.append(f"Number of test clauses: {len(test_cases)}")
    output_lines.append("=" * 70 + "\n")

    print("\n".join(output_lines[:5]))

    for i, case in enumerate(test_cases, 1):
        category = case["category"]
        text = case["text"]
        
        # Run inference
        result = predict_clause(text)
        
        line_item = (
            f"Test Case {i} | Category: {category}\n"
            f"  Input Clause:  \"{text}\"\n"
            f"  Predicted Type: {result['predicted_clause_type']} (Confidence: {result['clause_type_confidence']:.4f})\n"
            f"  Predicted Risk: {result['predicted_risk_level'].upper()} (Confidence: {result['risk_level_confidence']:.4f})\n"
            f"  Joint Conf:     {result['confidence']:.4f}\n"
            f"  Fallback Used:  {result.get('is_fallback', False)}\n"
            + "-" * 70
        )
        print(line_item)
        output_lines.append(line_item)

    # Save to file ml/inference/results/inference_test_results.txt
    results_dir = os.path.join(ml_dir, "inference", "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "inference_test_results.txt")
    
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
        
    print(f"\nInference test report successfully saved to: {output_path}")

if __name__ == "__main__":
    run_inference_tests()
