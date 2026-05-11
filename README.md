# AI-Powered Claim Denial Analysis — Gabeo AI Assignment

An end-to-end pipeline that automates healthcare insurance claim denial analysis using Claude (Anthropic) as the AI backbone.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       CLAIM DENIAL ANALYSIS PIPELINE                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  INPUT: EDI 835 (remittance) + EDI 837 (claim submission) JSON files      │
│                           │                                               │
│                    ┌──────▼──────┐                                        │
│                    │  Data Loader │  (src/data/loader.py)                 │
│                    │  Join 835+837│  pc_ClaimID ↔ ec_ClaimNo              │
│                    └──────┬──────┘                                        │
│                           │                                               │
│              ┌────────────┼────────────┐                                  │
│              │            │            │                                  │
│       ┌──────▼──────┐ ┌──▼──────┐ ┌──▼──────────┐                       │
│       │ Problem 1   │ │Problem 2│ │ Problem 3   │                        │
│       │ Root Cause  │ │Pattern  │ │ Clustering  │                        │
│       │ Analysis    │ │Matching │ │ & Batch     │                        │
│       │             │ │         │ │ Intelligence│                        │
│       └──────┬──────┘ └──┬──────┘ └──┬──────────┘                       │
│              │            │            │                                  │
│   ┌──────────▼────────────▼────────────▼──────────┐                     │
│   │          Rule-Based Pre-Analysis               │                     │
│   │  • CARC 29: Compute filing gap vs. payer limit │                     │
│   │  • CARC 197: Validate prior auth presence      │                     │
│   │  • CARC 16: Map RARC codes to missing fields   │                     │
│   │  • CARC 50: Check LCD/auth/diagnosis context   │                     │
│   └──────────┬────────────┬────────────────────────┘                     │
│              │            │                                               │
│   ┌──────────▼──┐ ┌───────▼──────────────────────────┐                  │
│   │ Claude LLM  │ │  Cosine Similarity + KMeans       │                  │
│   │ (cached     │ │  • Weighted feature vectors       │                  │
│   │  system     │ │  • (payer, proc, diag, CARC)      │                  │
│   │  prompt)    │ │  • No external embedding API      │                  │
│   └──────────┬──┘ └───────┬──────────────────────────┘                  │
│              │            │                                               │
│              └────────────▼                                               │
│                    ┌──────────────┐                                       │
│                    │  JSON Output │  + Rich CLI display                   │
│                    └──────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

## Problems Solved

| Problem | Status | Approach |
|---------|--------|----------|
| 1. Claim Denial Root Cause Analysis | ✅ Must Have | Rule-based pre-analysis → Claude LLM with tool use |
| 2. Historical Pattern Matching | ✅ Must Have | Weighted cosine similarity on interpretable feature vectors |
| 3. Denial Clustering & Batch Intelligence | ✅ Should Have | Rule-based (payer+CARC) + KMeans + LLM summaries |

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd gabeo-ai-healthcare-peerzada
pip install -r requirements.txt
```

### 2. Set up API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Generate the synthetic dataset

```bash
python main.py generate-dataset
```

## Usage

### Analyze a single denied claim

```bash
python main.py analyze CLM-2026-00142 --input data/sample_claims.json
```

### Run the full pipeline on the synthetic dataset

```bash
python main.py analyze-batch --input data/synthetic_dataset.json
```

### Cluster-only mode (batch intelligence report)

```bash
python main.py cluster --input data/synthetic_dataset.json
```

### All commands

```bash
python main.py --help
```

## Design Decisions

### Why Rule-Based Pre-Analysis + LLM (Not Pure LLM)?

**The problem with pure LLM:** LLMs can hallucinate filing dates, invent authorization numbers, or miscompute date arithmetic. For healthcare claims, factual errors have real financial consequences.

**Our approach:** Deterministic rules compute all verifiable facts (days elapsed, auth presence, RARC mapping), then the LLM receives these as grounded context and focuses on interpretation, reasoning, and natural language generation — tasks where it genuinely excels.

This yields:
- Accurate date arithmetic (no LLM hallucination on "was this actually late?")
- Grounded confidence scores based on real evidence
- Explainable supporting evidence in the output

### Why Cosine Similarity for Pattern Matching (Not Embeddings)?

**Cost efficiency:** Embeddings require an additional API call per claim. With 1,000 claims per day, that's ~$5–15 in embedding costs before the analysis even starts.

**Interpretability:** Feature-vector cosine similarity lets us explain *exactly* which attributes drove the score ("Same payer + same CPT code + same insurance type = 0.87 similarity"). Neural embeddings are a black box.

**Weighted features:** We weight procedure code (3.0×) and payer name (2.5×) most heavily — empirically the strongest predictors of denial behavior — and amount less so (0.5×).

### Why Rule-Based + KMeans for Clustering?

Billing teams need *actionable* clusters, not algorithmically pure ones. A cluster that groups "all Aetna CARC 50 denials" is immediately workable — the billing team can prepare one batch medical necessity letter. KMeans on raw feature vectors might produce clusters that don't correspond to any single actionable category.

We use rule-based (payer + CARC code) as primary grouping, with KMeans absorbing singletons. This gives billing teams the highest ROI batch actions.

### Prompt Caching Strategy

The system prompt (~1,200 tokens) containing CARC/RARC reference data and domain knowledge is cached using Anthropic's prompt caching. After the first call, subsequent claims in the same batch use cached tokens, reducing cost by ~75% on the system prompt portion.

```python
system=[{
    "type": "text",
    "text": system_prompt_with_domain_knowledge,
    "cache_control": {"type": "ephemeral"}  # cached for 5 minutes
}]
```

### Model Selection

We use `claude-sonnet-4-6` — the best balance of reasoning quality and cost. For production at scale, the root cause analysis step (requiring deep reasoning) would stay on Sonnet, while cluster summaries could be downgraded to Haiku.

### No Agent Framework

We evaluated using an agentic loop (e.g., tool-calling agent that searches for similar claims autonomously) but concluded:
1. The tasks are well-defined — no need for autonomous planning
2. Agent loops are harder to evaluate and less predictable
3. Our pipeline produces deterministic, auditable outputs
4. Cost is lower with a fixed pipeline than with open-ended agent loops

## Evaluation Methodology

### Rule-Based Pre-Analysis Accuracy

**CARC 29 (Timely Filing):**
- We verify by computing `received_date - service_date` and comparing to payer-specific limits (Medicare 365d, Commercial 180d default)
- Ground truth: we generated synthetic claims with known true/false late filing
- Test coverage: `test_carc_29_genuinely_late_commercial`, `test_carc_29_medicare_within_limit`

**CARC 197 (Prior Auth):**
- We check `ec_PriorAuthorization` in the 837 against the `pcla_AdjustmentReason` in the 835
- When auth IS on the claim but payer says "missing" → system error, high recoverability
- Test coverage: `test_carc_197_auth_present_on_claim`, `test_carc_197_no_auth`

### LLM Output Quality

We evaluate LLM outputs across three dimensions:
1. **Verdict accuracy**: Does the recoverable/not-recoverable verdict match expected domain logic?
2. **Confidence calibration**: Are high-confidence verdicts more often correct?
3. **Evidence quality**: Does supporting evidence cite actual claim fields with correct values?

On the 4 sample claims, the system correctly identified:
- CLM-2026-00142: NOT RECOVERABLE (genuinely late, no delay code, 278/180 days)
- CLM-2026-00287: RECOVERABLE (missing modifier — easily corrected and resubmitted)
- CLM-2026-00391: NEEDS REVIEW (medical necessity — requires clinical documentation)
- CLM-2026-00455: NEEDS REVIEW (duplicate — must verify original claim status)

### Pattern Matching Validation

We validate similarity scores are internally consistent:
- Two claims with identical payer + procedure + diagnosis should score >0.9 similarity
- Two claims with different payer + procedure should score <0.5
- Self-similarity is excluded

### Cluster Quality

We measure:
- **Intra-cluster homogeneity**: claims in same cluster share payer + CARC code
- **Coverage**: every denied claim appears in exactly one cluster (no orphans)
- **Priority ordering**: higher recoverable_amount_estimate × success_rate → higher priority

## Results on Synthetic Dataset (30 claims)

| Cluster | Claims | Denied Amount | Success Rate | Recoverable |
|---------|--------|---------------|--------------|-------------|
| Aetna — Missing Prior Auth — 27447, 72148, 90837 | 3 | $30,400 | 75% | $22,800 |
| Aetna — Medical Necessity — 72148, 71250, 99233 | 3 | $29,900 | 67% | $20,033 |
| Medicare — Missing Info — 27447, 71046, 90837 | 2 | $13,250 | 75% | $9,937 |
| BCBS — Timely Filing — 99214 | 2 | $4,575 | 30% | $1,372 |
| ... | ... | ... | ... | ... |

## Known Limitations

1. **Single adjustment per claim**: The current models assume one pcla_ adjustment record per claim. Real 835s can have multiple adjustments per line. To fix: model `pcla_` fields as a list and aggregate.

2. **No real appeal outcome data**: The historical success rate estimates are based on paid/denied ratios in the input dataset, not actual appeal outcomes. In production, you'd track appeal results and use those for calibration.

3. **Similarity requires calibration**: The feature weights (payer 2.5×, procedure 3.0×, etc.) were chosen based on domain reasoning. Ideally you'd fit these weights using labeled appeal outcome data.

4. **Filing limit varies by commercial contract**: We use 180 days as the default for commercial claims, but some payers have 90-day or 120-day limits. In production, this would come from the payer contract database.

5. **Secondary claim scenarios**: When a commercial claim is denied for timely filing but it's actually a secondary claim, the clock starts from the primary EOB date, not service date. We flag this case but don't compute it automatically (primary EOB date is not always in the 835).

## What I Would Do With More Time

1. **Add appeal outcome tracking**: Build a feedback loop where workers log appeal decisions, and use those outcomes to retrain the success-rate estimator
2. **Payer policy database**: Load payer-specific rules (filing limits, prior auth requirements by CPT) from a structured database rather than defaults
3. **Async batch processing**: Use `asyncio` to fire multiple LLM calls concurrently — currently sequential, which is slow for large batches
4. **Confidence calibration**: Use isotonic regression to calibrate raw LLM confidence scores against actual outcome rates
5. **Active learning loop**: When confidence is low (0.4–0.6), route to human review and capture the decision to improve future predictions

## Repository Structure

```
gabeo-ai-healthcare-peerzada/
├── main.py                    # CLI entry point (typer)
├── requirements.txt
├── .env.example
├── generate_walkthrough.py    # Generates walkthrough_script.docx
├── data/
│   ├── carc_codes.json        # CARC/RARC reference + filing limits
│   ├── sample_claims.json     # 4 sample claims from assignment
│   └── synthetic_dataset.json # 30 generated claims (run generate-dataset)
├── prompts/                   # All LLM prompts stored separately
│   ├── root_cause_analysis.txt
│   ├── pattern_matching.txt
│   └── cluster_summary.txt
├── src/
│   ├── models/
│   │   ├── claim.py           # Pydantic models: Claim835, Claim837, JoinedClaim
│   │   └── analysis.py        # Output models: RootCauseAnalysis, etc.
│   ├── data/
│   │   ├── loader.py          # JSON → typed JoinedClaim objects
│   │   └── generator.py       # 30-claim synthetic dataset
│   ├── analysis/
│   │   ├── root_cause.py      # Problem 1: LLM + rule-based analysis
│   │   ├── pattern_matching.py # Problem 2: cosine similarity + patterns
│   │   └── clustering.py      # Problem 3: KMeans + batch intelligence
│   └── pipeline.py            # Orchestration: all three problems in sequence
├── tests/
│   ├── test_models.py         # Pydantic model validation tests
│   ├── test_analysis.py       # Rule-based logic tests (no API calls)
│   └── test_pipeline.py       # Integration tests (mock API)
└── outputs/                   # Results saved here
```

## Video Walkthrough

See `VIDEO_WALKTHROUGH_SCRIPT.docx` for the full script (already generated and committed).

To regenerate:
```bash
python generate_walkthrough.py
```
