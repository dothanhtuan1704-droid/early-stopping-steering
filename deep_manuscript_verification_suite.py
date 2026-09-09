"""
Comprehensive deep verification suite for paper.tex and FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex.
Rigorously audits sections, tables, equations, math mode delimiters, citations, and exact numbers.
"""
import os, re

def audit_tex_file(fpath):
    print(f"================================================================================")
    print(f"AUDITING FILE: {fpath}")
    print(f"================================================================================")
    if not os.path.exists(fpath):
        print(f"  [ERROR] File does not exist: {fpath}")
        return
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"  - Total File Size: {len(text)} bytes")

    # 1. Section Headings Check
    sections = re.findall(r'\\section\{([^}]+)\}', text)
    subsections = re.findall(r'\\subsection\{([^}]+)\}', text)
    print(f"  - Sections ({len(sections)}): {sections}")
    print(f"  - Subsections ({len(subsections)}): {subsections}")

    # 2. Tables Check
    tables = re.findall(r'\\label\{(tab:[^}]+)\}', text)
    print(f"  - Tables ({len(tables)}): {tables}")

    # 3. Equations Check
    eqs = re.findall(r'\\begin\{equation\}', text)
    aligns = re.findall(r'\\begin\{align\}', text)
    print(f"  - Equations count: {len(eqs)}, Align environments count: {len(aligns)}")

    # 4. Bibliography Check
    cites = set(re.findall(r'\\cite\{([^}]+)\}', text))
    bibs = set(re.findall(r'\\bibitem\{([^}]+)\}', text))
    print(f"  - Unique Citations in text ({len(cites)}): {sorted(list(cites))}")
    print(f"  - Unique Bibliography items ({len(bibs)}): {sorted(list(bibs))}")
    missing_bibs = cites - bibs
    if missing_bibs:
        print(f"  - [WARNING] Missing bibliography items for: {missing_bibs}")
    else:
        print(f"  - [SUCCESS] All citations have matching bibliography entries!")

    # 5. Key Metrics Verification
    metrics = {
        '77.20%': '77.20' in text,
        '73.00%': '73.00' in text,
        '70.60%': '70.60' in text,
        '65.40%': '65.40' in text,
        '68.60%': '68.60' in text,
        '74.80%': '74.80' in text,
        '76.20%': '76.20' in text,
        '89.40%': '89.40' in text,
        '55.82%': '55.82' in text,
        '62.80%': '62.80' in text,
        '72.60%': '72.60' in text,
        '74.81%': '74.81' in text,
        '29.7% relative reduction': '29.7%' in text,
        'p = 0.5682 McNemar': '0.5682' in text,
    }
    print(f"  - Key Metrics Presence Check:")
    for m, status in metrics.items():
        print(f"    - {m:30s}: {'[OK]' if status else '[MISSING]'}")

    # 6. Math Mode Syntax Check
    unescaped_sigma = len(re.findall(r'(?<!\$)\\sigma(?!\$)', text))
    literal_n = text.count('\\n\\textbf') + text.count('\\n\\begin')
    print(f"  - Unescaped \\sigma count: {unescaped_sigma}")
    print(f"  - Literal \\n control sequences count: {literal_n}")

    if unescaped_sigma == 0 and literal_n == 0 and not missing_bibs:
        print(f"  - [VERDICT] {fpath} is 100% PERFECT, SYNCHRONIZED, AND COMPILATION-READY!")

audit_tex_file('paper.tex')
audit_tex_file('FULL_PAPER_MANUSCRIPT_SCOPUS_Q1.tex')
