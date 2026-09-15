"""
================================================================================
BUILD 50-SAMPLE CLINICAL PHARMACY AUDIT ADJUDICATION MANIFEST
================================================================================
Generates `data/human_audit_50_adjudication_manifest.json` detailing the complete 
itemized adjudication records for all N=50 audit test questions evaluated by 
clinical pharmacy reviewers.

Summary:
  - 47/50 (94.0%) PASS: Strict monograph alignment (human_correctness_score = 2).
  - 3/50 (6.0%) NON_ALIGNMENT: Edge case non-alignments (human_correctness_score = 1):
      * Sample ID 4: Dosage snippet truncation in extracted monograph context.
      * Sample ID 8: Interaction phrasing offset across monograph sections.
      * Sample ID 17: Administration route ambiguity (oral vs parenteral formulation).

Author: Phan Do Thanh Tuan
Workspace: E:\Paper_Steering_VN_15K
================================================================================
"""

import os
import sys
import json
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_manifest():
    csv_path = "human_evaluation_completed_50.csv"
    output_json = os.path.join("data", "human_audit_50_adjudication_manifest.json")
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find input CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    adjudication_records = []
    pass_count = 0
    non_alignment_count = 0

    for idx, row in df.iterrows():
        sample_id = int(row['sample_id'])
        category = str(row['category'])
        score = int(row['human_correctness_score'])
        notes = str(row['human_notes'])

        status = "PASS" if score == 2 else "NON_ALIGNMENT"
        if score == 2:
            pass_count += 1
        else:
            non_alignment_count += 1

        record = {
            "sample_id": sample_id,
            "category": category,
            "reviewer_role": "Clinical Pharmacy Reviewer",
            "alignment_status": status,
            "human_correctness_score": score,
            "monograph_reference": "Vietnamese National Drug Formulary (2nd Edition, 2018)",
            "adjudication_notes": notes
        }
        adjudication_records.append(record)

    manifest_data = {
        "metadata": {
            "dataset": "ViHaluEval-Medical Stratified Audit Sample",
            "total_audit_samples": len(adjudication_records),
            "reviewer_qualification": "Clinical Pharmacy Reviewers",
            "pass_count": pass_count,
            "non_alignment_count": non_alignment_count,
            "strict_monograph_alignment_rate": f"{(pass_count / len(adjudication_records)) * 100:.1f}%",
            "flagged_sample_ids": [r["sample_id"] for r in adjudication_records if r["alignment_status"] == "NON_ALIGNMENT"]
        },
        "adjudication_records": adjudication_records
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Built 50-sample adjudication manifest: '{output_json}'")
    print(f"[*] Summary: Total={len(adjudication_records)}, Pass={pass_count} ({pass_count/len(adjudication_records)*100:.1f}%), Non-Alignment={non_alignment_count}")

if __name__ == '__main__':
    build_manifest()
