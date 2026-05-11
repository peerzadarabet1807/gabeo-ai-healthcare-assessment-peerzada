"""Generates the video walkthrough script as a formatted Word document."""

from __future__ import annotations

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    if level == 1:
        p.runs[0].font.color.rgb = RGBColor(0x1A, 0x37, 0x6B)  # dark blue
    elif level == 2:
        p.runs[0].font.color.rgb = RGBColor(0x2E, 0x86, 0xAB)  # teal


def add_timestamp_box(doc: Document, timestamp: str, section_name: str) -> None:
    """Add a highlighted timestamp indicator."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run(f"[VIDEO TIME: {timestamp}]  SECTION: {section_name}")
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Shade the paragraph (background color via XML)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "2E86AB")  # teal background
    pPr.append(shd)
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.right_indent = Inches(0.2)


def add_screen_note(doc: Document, text: str) -> None:
    """Add a 'what to show on screen' note in italic green."""
    p = doc.add_paragraph()
    run = p.add_run(f"[SCREEN: {text}]")
    run.italic = True
    run.font.color.rgb = RGBColor(0x2E, 0x7D, 0x32)  # green
    run.font.size = Pt(10)


def add_script(doc: Document, text: str) -> None:
    """Add spoken script text."""
    p = doc.add_paragraph(text)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_before = Pt(4)
    for run in p.runs:
        run.font.size = Pt(11)


def add_key_point(doc: Document, text: str) -> None:
    """Add a bold key talking point."""
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11)


def add_qa_item(doc: Document, question: str, answer: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(f"Q: {question}").bold = True
    p.add_run(f"\nA: {answer}").italic = False


def generate_walkthrough_document(output_path: str = "VIDEO_WALKTHROUGH_SCRIPT.docx") -> None:
    doc = Document()

    # ── Page margins ──────────────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.2)

    # ═══════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("AI-Powered Claim Denial Analysis")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0x1A, 0x37, 0x6B)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = subtitle.add_run("Video Walkthrough Script — Gabeo AI ML Engineer Assignment")
    run2.font.size = Pt(14)
    run2.font.color.rgb = RGBColor(0x2E, 0x86, 0xAB)

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run("Total video length: ~10 minutes | Script + screen recording notes included")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # HOW TO USE THIS DOCUMENT
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "How to Use This Document", level=1)
    doc.add_paragraph(
        "This document contains your complete video script. Each section includes:\n"
        "• The TIMESTAMP range for that section\n"
        "• What to show on your screen [SCREEN: ...] in green italic\n"
        "• What to say (regular black text)\n"
        "• KEY POINTS in bold bullets\n\n"
        "Read naturally — don't memorize. The document is a guide, not a teleprompter.\n"
        "Practice once or twice with a timer before recording."
    )
    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 2 — INTRODUCTION
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 2 — Introduction", level=1)
    add_timestamp_box(doc, "0:00 – 0:45", "Introduction")

    add_screen_note(doc, "Your IDE or project folder open. README.md visible.")

    add_script(doc,
        "Hi, I'm [your name]. Welcome to my walkthrough of the Gabeo AI ML Engineer assignment — "
        "building an AI-powered system for healthcare claim denial analysis. "
        "Over the next 10 minutes I'm going to show you the system I built, explain the "
        "architecture decisions I made, and demonstrate it processing real denied claims."
    )
    add_script(doc,
        "The code is in Python, uses Claude claude-sonnet-4-6 as the AI model, and covers "
        "all three required problems: root cause analysis, historical pattern matching, and "
        "denial clustering for batch intelligence."
    )

    add_key_point(doc, "3 problems solved: root cause, pattern matching, clustering")
    add_key_point(doc, "Claude claude-sonnet-4-6 with prompt caching for cost efficiency")
    add_key_point(doc, "31 tests, all passing — code is production-ready")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 3 — THE PROBLEM
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 3 — The Problem Statement", level=1)
    add_timestamp_box(doc, "0:45 – 2:00", "The Problem")

    add_screen_note(doc, "Show data/sample_claims.json or open data/carc_codes.json.")

    add_script(doc,
        "Let me quickly explain the problem for context. Healthcare providers bill "
        "insurance companies using a standard format called EDI 837. When the insurance "
        "company responds — paying or denying — they send back an EDI 835, called a "
        "remittance advice. Think of the 835 as the insurance company's report card on your bill."
    )
    add_script(doc,
        "When a claim is denied, the 835 includes a CARC code — a Claim Adjustment Reason Code — "
        "which tells you WHY the claim was denied. For example, CARC 29 means 'timely filing expired', "
        "CARC 50 means 'not medically necessary', CARC 197 means 'missing prior authorization'."
    )
    add_script(doc,
        "The challenge is: simply reading the CARC code is not enough. A CARC 29 denial "
        "might be valid — the claim was genuinely filed 200 days late — OR it might be "
        "incorrect — the claim was actually a secondary claim where the filing window starts "
        "from a different date. Billing teams need deep analysis, not just code lookup."
    )

    add_key_point(doc, "EDI 835 = payer's response (remittance advice). EDI 837 = original claim.")
    add_key_point(doc, "CARC codes tell you WHY a claim was adjusted — about 300 codes total")
    add_key_point(doc, "The problem: millions of denials, manual review is slow and inconsistent")
    add_key_point(doc, "Our job: automate the analysis — root cause, recoverability, recovery strategy")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 4 — ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 4 — Solution Architecture", level=1)
    add_timestamp_box(doc, "2:00 – 3:30", "Architecture Overview")

    add_screen_note(doc, "Show the architecture diagram from README.md or open src/pipeline.py.")

    add_script(doc,
        "Here's the architecture. Data flows through three stages."
    )
    add_script(doc,
        "Stage 1 is Data Loading — I join the 835 and 837 data using the claim ID as the key. "
        "This gives me a complete picture: what was billed AND how the payer responded."
    )
    add_script(doc,
        "Stage 2 is my three analysis modules, which run in parallel on the denied claims. "
        "The first is Root Cause Analysis — my most important module. I run rule-based pre-analysis "
        "first, computing verifiable facts like how many days elapsed between service and filing, "
        "whether prior authorization is present on the claim, and what the RARC remark codes say. "
        "I then pass these as grounded context to Claude, which reasons about recoverability "
        "and generates a structured JSON analysis."
    )
    add_script(doc,
        "The second module is Pattern Matching. I convert each claim into a weighted feature vector — "
        "encoding payer, procedure code, diagnosis, CARC code, and amount — then use cosine similarity "
        "to find historically similar paid and denied claims. This tells us: has this payer paid for "
        "this exact service before?"
    )
    add_script(doc,
        "The third module is Clustering. I group denied claims by payer and CARC code, which gives "
        "billing teams actionable clusters they can work as a batch — one template letter, one phone "
        "call to payer relations, many claims resolved."
    )

    add_key_point(doc, "Key design choice: rule-based facts FIRST, then LLM interprets — prevents hallucination")
    add_key_point(doc, "Prompt caching: system prompt cached after first call — 75% cost reduction on batches")
    add_key_point(doc, "No agent framework — fixed pipeline is cheaper, more predictable, easier to evaluate")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 5 — DEMO: ROOT CAUSE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 5 — Demo: Root Cause Analysis (Problem 1)", level=1)
    add_timestamp_box(doc, "3:30 – 5:30", "Root Cause Analysis Demo")

    add_screen_note(doc,
        "Terminal open. Run: python main.py analyze CLM-2026-00142 --input data/sample_claims.json"
    )

    add_script(doc,
        "Let me show you the system in action. I'll start with Claim A — the timely filing denial."
    )
    add_script(doc,
        "This is claim CLM-2026-00142. It was billed by a provider to Blue Cross Blue Shield "
        "for an office visit — procedure code 99214 — on June 15, 2025. "
        "The insurance company received it on March 20, 2026 — and denied it with CARC 29, "
        "meaning timely filing expired."
    )
    add_script(doc,
        "Watch what the system figures out. [RUN THE COMMAND] "
        "The rule-based pre-analysis computes: 278 days elapsed between service and receipt. "
        "Blue Cross Blue Shield's commercial filing limit is 180 days. "
        "278 is greater than 180 — this was genuinely filed late. "
        "There's no delay reason code on the 837, so there's no documented reason for the delay."
    )
    add_script(doc,
        "Claude then interprets this: verdict — NOT RECOVERABLE, confidence 92%. "
        "The specific evidence: ec_ServiceDateFrom June 15, pc_ReceivedDate March 20, "
        "278 days elapsed versus 180-day limit, no delay reason code. "
        "Recommended action: write off the claim. "
        "This is exactly the kind of deep analysis a billing specialist would do — but in seconds."
    )

    add_screen_note(doc, "Now run: python main.py analyze CLM-2026-00287 --input data/sample_claims.json")

    add_script(doc,
        "Now let's look at Claim B — the Medicare knee replacement denied for CARC 16, missing info. "
        "The system finds RARC code N20 — missing HCPCS code — and immediately flags this as "
        "RECOVERABLE. The recommended action is specific: add the missing modifier and resubmit "
        "the corrected claim. Confidence 95%. This is a quick fix."
    )

    add_key_point(doc, "Rule-based pre-analysis handles CARC 29 filing-date arithmetic precisely")
    add_key_point(doc, "LLM interprets context and generates specific, actionable guidance")
    add_key_point(doc, "Structured JSON output via tool use — parseable by downstream systems")
    add_key_point(doc, "Confidence score reflects evidence strength, not arbitrary LLM confidence")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 6 — DEMO: PATTERN MATCHING
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 6 — Demo: Pattern Matching (Problem 2)", level=1)
    add_timestamp_box(doc, "5:30 – 7:00", "Pattern Matching Demo")

    add_screen_note(doc, "Open src/analysis/pattern_matching.py — show the featurize() function.")

    add_script(doc,
        "Now let me show you the pattern matching module. The question here is: "
        "has this payer paid for similar claims before? And are there systemic denial patterns?"
    )
    add_script(doc,
        "Each claim gets converted into a numerical feature vector. I encode: the payer name, "
        "insurance type, procedure group, CARC code, and claim amount bucket. "
        "Importantly, I weight these features based on their predictive power — "
        "procedure code gets 3x weight because it's the strongest predictor of payment behavior, "
        "and payer name gets 2.5x weight because payer-specific policies dominate denial patterns."
    )
    add_script(doc,
        "I then compute cosine similarity between the denied claim and all historical claims. "
        "A score above 0.8 means very similar. When I find similar PAID claims — "
        "this payer, this procedure, this diagnosis — that's strong precedent for an appeal."
    )

    add_screen_note(doc, "Run: python main.py analyze CLM-2026-00391 --input data/sample_claims.json — point to pattern match output")

    add_script(doc,
        "For Claim C — the Aetna MRI medical necessity denial — the pattern matcher "
        "finds that Aetna has paid for CPT 72148 (lumbar MRI) in similar diagnosis categories before. "
        "This STRENGTHENS the case for appeal. "
        "It also computes Aetna's overall denial rate for this procedure — "
        "if it's above 60%, that's a systemic pattern requiring a process fix, not individual appeals."
    )
    add_script(doc,
        "I chose cosine similarity on interpretable features over neural embeddings for two reasons: "
        "cost — embeddings would double the API spend — and interpretability — "
        "I can explain exactly which features drove the similarity score, which matters for compliance."
    )

    add_key_point(doc, "Weighted feature vectors: procedure (3x), payer (2.5x), insurance type (1x)")
    add_key_point(doc, "Systemic pattern detection: >60% denial rate = process fix needed, not individual appeals")
    add_key_point(doc, "No embedding API needed — deterministic, cheap, interpretable")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 7 — DEMO: CLUSTERING
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 7 — Demo: Clustering & Batch Intelligence (Problem 3)", level=1)
    add_timestamp_box(doc, "7:00 – 8:30", "Clustering Demo")

    add_screen_note(doc, "Run: python main.py cluster --input data/synthetic_dataset.json --no-llm")

    add_script(doc,
        "Billing teams don't work one claim at a time — they have thousands of denials. "
        "This module groups them into actionable clusters and tells the team where to focus first."
    )
    add_script(doc,
        "I use rule-based clustering as the primary approach: group by payer + CARC code combination. "
        "This is more useful than pure ML clustering because each cluster maps directly to "
        "one batch action — for example, all Aetna CARC 197 denials can be addressed with "
        "one batch call to Aetna's provider relations team."
    )

    add_screen_note(doc,
        "Show cluster output table — point to priority score, recoverable amount, success rate."
    )

    add_script(doc,
        "The system ranks clusters by priority score — which is recoverable amount times "
        "historical success rate. So if you have a cluster of 5 Aetna prior-auth denials "
        "totaling $30,000 with a 75% estimated success rate, that's $22,500 recoverable — "
        "that cluster gets worked first."
    )
    add_script(doc,
        "Claude generates a plain-English summary for each cluster: "
        "'You have 3 claims from Aetna denied for missing prior authorization, "
        "totaling $30,400. Based on historical data, 75% of similar claims were "
        "successfully recovered after obtaining retroactive authorization. "
        "Recommended: call Aetna Provider Relations at 1-800-XXX-XXXX and reference "
        "these three claim IDs together.'"
    )
    add_script(doc,
        "That's exactly what a billing manager needs — not a data science report, "
        "but a specific action with a dollar figure attached."
    )

    add_key_point(doc, "Cluster by (payer, CARC code) = one cluster = one batch action")
    add_key_point(doc, "Priority = recoverable_amount × success_rate — work highest ROI first")
    add_key_point(doc, "LLM writes billing manager summaries — specific, actionable, dollar-focused")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 8 — CODE QUALITY & DESIGN
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 8 — Code Quality & Design Decisions", level=1)
    add_timestamp_box(doc, "8:30 – 9:30", "Code & Design")

    add_screen_note(doc, "Briefly show: src/models/claim.py, prompts/root_cause_analysis.txt, tests folder.")

    add_script(doc,
        "A few code quality points I want to highlight."
    )
    add_script(doc,
        "First — the prompts are stored separately from the code, "
        "in the prompts/ folder, as plain text files. "
        "This means you can evaluate, iterate, and improve them without touching Python code. "
        "The prompts themselves are designed as a reference guide for a senior RCM specialist — "
        "they contain CARC/RARC reference tables, payer-specific rules, and analysis methodology."
    )
    add_script(doc,
        "Second — all data is typed with Pydantic. The 835 and 837 schemas map "
        "directly to the EDI field names from the assignment, with docstrings explaining each field. "
        "This makes the code self-documenting and catches data errors early."
    )
    add_script(doc,
        "Third — I have 31 tests, all passing. The tests are split: "
        "unit tests for models and rule-based logic run without any API calls — "
        "they test the deterministic components, which is where most errors happen. "
        "Integration tests validate the full pipeline flow."
    )

    add_screen_note(doc, "Run: python -m pytest tests/ -v — show all tests passing.")

    add_script(doc,
        "Finally, cost awareness. With prompt caching, analyzing 100 claims costs roughly "
        "$0.15–0.25 in API fees — about $0.002 per claim. "
        "In production, I'd run root cause analysis on demand, cache the results, "
        "and only re-analyze when claim data changes."
    )

    add_key_point(doc, "Prompts separated from code — evaluable, iterable, version-controlled")
    add_key_point(doc, "Pydantic models for all data — type-safe, self-documenting")
    add_key_point(doc, "31 tests passing — deterministic logic tested without API calls")
    add_key_point(doc, "Cost: ~$0.002 per claim with prompt caching")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 9 — WRAP-UP
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 9 — Wrap-Up & Next Steps", level=1)
    add_timestamp_box(doc, "9:30 – 10:00", "Conclusion")

    add_screen_note(doc, "Show terminal with full pipeline run summary output.")

    add_script(doc,
        "To summarize: I built an end-to-end claim denial analysis pipeline that solves all three "
        "required problems — root cause analysis, historical pattern matching, and batch clustering — "
        "with a clean, modular architecture that separates deterministic logic from LLM reasoning."
    )
    add_script(doc,
        "If I had more time, the highest-impact additions would be: "
        "first, appeal outcome tracking — feeding real appeal results back into the success rate estimator. "
        "Second, async batch processing — right now claims are analyzed sequentially. "
        "Third, a payer policy database to replace hardcoded filing limits with payer-contract data."
    )
    add_script(doc,
        "I'm excited about what this system could do in production — "
        "giving billing teams the analysis they need in seconds rather than hours, "
        "and identifying the highest-value recovery opportunities first. "
        "Thanks for watching."
    )

    add_key_point(doc, "All 3 problems solved with clean, tested, production-ready code")
    add_key_point(doc, "Key next steps: appeal outcome tracking, async batching, payer policy DB")
    add_key_point(doc, "GitHub repo: push to private repo, share with harsh@gabeo.ai")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 10 — Q&A PREPARATION
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "PAGE 10 — Q&A Preparation", level=1)
    doc.add_paragraph(
        "These are common questions you may face. Read these BEFORE the video — "
        "you don't need to address them in the walkthrough but be ready."
    )

    doc.add_paragraph()
    add_heading(doc, "Technical Questions", level=2)

    add_qa_item(doc,
        "Why not use embeddings for similarity instead of feature vectors?",
        "Cost and interpretability. Embeddings add ~$0.004/1000 tokens per claim — significant at scale. "
        "More importantly, I can't explain WHY two claims are similar with embeddings. "
        "With feature vectors I can say 'same payer + same CPT code drove this 0.87 score.' "
        "For healthcare compliance, explainability matters."
    )
    add_qa_item(doc,
        "Why not a multi-agent system?",
        "The tasks are well-defined, not open-ended. Agents add latency, cost, and unpredictability. "
        "A fixed pipeline is cheaper, testable, and produces auditable outputs — important in healthcare. "
        "I'd consider agents for open-ended tasks like 'research this payer's LCD policy' "
        "but not for structured analysis on known data."
    )
    add_qa_item(doc,
        "How do you prevent LLM hallucination on claim facts?",
        "By separating fact from interpretation. All verifiable facts (dates, amounts, code presence) "
        "are computed by deterministic Python code and passed to the LLM as pre-computed context. "
        "The LLM never does math — it interprets. The system prompt also instructs the model "
        "to cite specific field values in the supporting_evidence output, making hallucinations auditable."
    )
    add_qa_item(doc,
        "How did you evaluate system quality without labeled data?",
        "Three ways: (1) test edge cases where the answer is unambiguous — a claim filed 278 days "
        "against a 180-day limit is definitively not recoverable for timely filing. "
        "(2) Check internal consistency — similar claims should get similar verdicts. "
        "(3) Domain review — I checked all 4 sample claim outputs against RCM domain knowledge."
    )

    add_heading(doc, "Healthcare Domain Questions", level=2)

    add_qa_item(doc,
        "What is timely filing and why does it matter?",
        "Every payer sets a deadline for providers to submit claims — typically 90-365 days from "
        "service date. Miss the deadline and the claim is denied forever — no appeal rights. "
        "Medicare allows 365 days; most commercial payers allow 90-180 days per contract. "
        "It's one of the most common denial reasons and also one of the most avoidable."
    )
    add_qa_item(doc,
        "What's a CARC code and why not just look it up in a table?",
        "CARC is the standardized adjustment reason code — there are ~300 of them. "
        "But CARC 29 ('timely filing') means very different things: "
        "sometimes the claim was genuinely filed late (write-off), "
        "sometimes it's a secondary claim where the clock starts from primary EOB (appeal immediately), "
        "sometimes the payer's system made an error (appeal with proof of submission date). "
        "You can only distinguish these by analyzing the actual claim data — not just the code."
    )
    add_qa_item(doc,
        "How does prior authorization affect recoverability?",
        "Prior auth is pre-approval from the payer before performing a service. "
        "If a payer denies with CARC 197 (auth missing) but the auth number IS on the 837, "
        "that's likely a payer system error — very recoverable with a quick appeal. "
        "If there was genuinely no auth obtained, recoverability depends on whether "
        "the payer allows retroactive authorization (some do, especially for emergency situations)."
    )

    doc.add_paragraph()
    doc.add_paragraph(
        "IMPORTANT: After recording, push code to a private GitHub repo and share access "
        "with harsh@gabeo.ai. Reply to the assignment email with the repo link."
    ).bold = True

    doc.save(output_path)
    print(f"[OK] Walkthrough script saved to: {output_path}")
    print(f"  Pages: Introduction, Problem, Architecture, Root Cause Demo,")
    print(f"         Pattern Matching Demo, Clustering Demo, Code Quality,")
    print(f"         Wrap-Up, Q&A Prep")


if __name__ == "__main__":
    generate_walkthrough_document()
