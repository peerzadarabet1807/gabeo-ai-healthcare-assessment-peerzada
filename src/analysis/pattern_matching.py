"""Problem 2: Historical Pattern Matching using feature-based cosine similarity."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.models.claim import JoinedClaim
from src.models.analysis import PatternMatchResult, SimilarClaim

CARC_CODES_PATH = Path(__file__).parent.parent.parent / "data" / "carc_codes.json"
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "pattern_matching.txt"

# ─── Feature Engineering ──────────────────────────────────────────────────────

# Known categorical values for one-hot encoding
_PAYERS = [
    "Medicare Part B", "State Medicaid", "Blue Cross Blue Shield",
    "Aetna", "United Healthcare", "Cigna", "Humana", "Other"
]
_INSURANCE_TYPES = ["Medicare", "Medicaid", "Commercial"]
_CARC_CODES = ["4", "16", "18", "29", "45", "50", "96", "97", "197", "252", "other"]
_PROCEDURE_GROUPS = [
    "E&M Office", "E&M Inpatient", "Orthopedic Surgery", "Radiology MRI",
    "Radiology XRay", "Radiology CT", "GI Endoscopy", "Mental Health",
    "Cardiology", "Other"
]

_PROCEDURE_TO_GROUP = {
    "99201": "E&M Office", "99202": "E&M Office", "99203": "E&M Office",
    "99204": "E&M Office", "99205": "E&M Office", "99211": "E&M Office",
    "99212": "E&M Office", "99213": "E&M Office", "99214": "E&M Office",
    "99215": "E&M Office",
    "99231": "E&M Inpatient", "99232": "E&M Inpatient", "99233": "E&M Inpatient",
    "27447": "Orthopedic Surgery", "29827": "Orthopedic Surgery",
    "23472": "Orthopedic Surgery", "22612": "Orthopedic Surgery",
    "20610": "Orthopedic Surgery",
    "72148": "Radiology MRI", "72141": "Radiology MRI", "72156": "Radiology MRI",
    "71046": "Radiology XRay", "71010": "Radiology XRay",
    "71250": "Radiology CT", "74177": "Radiology CT",
    "43239": "GI Endoscopy", "43235": "GI Endoscopy",
    "90837": "Mental Health", "90832": "Mental Health", "90834": "Mental Health",
    "93000": "Cardiology", "93010": "Cardiology",
}


def _normalize_payer(name: str) -> str:
    name_lower = name.lower()
    if "medicare" in name_lower:
        return "Medicare Part B"
    if "medicaid" in name_lower or "state" in name_lower:
        return "State Medicaid"
    if "blue cross" in name_lower or "bcbs" in name_lower:
        return "Blue Cross Blue Shield"
    if "aetna" in name_lower:
        return "Aetna"
    if "united" in name_lower or "uhc" in name_lower:
        return "United Healthcare"
    if "cigna" in name_lower:
        return "Cigna"
    if "humana" in name_lower:
        return "Humana"
    return "Other"


def _one_hot(value: str, categories: list[str]) -> list[float]:
    return [1.0 if value == cat else 0.0 for cat in categories]


def _amount_bucket(amount: float) -> list[float]:
    """Encode claim amount into 5 buckets: <500, 500-2k, 2k-10k, 10k-25k, >25k."""
    buckets = [0.0, 0.0, 0.0, 0.0, 0.0]
    if amount < 500:
        buckets[0] = 1.0
    elif amount < 2000:
        buckets[1] = 1.0
    elif amount < 10000:
        buckets[2] = 1.0
    elif amount < 25000:
        buckets[3] = 1.0
    else:
        buckets[4] = 1.0
    return buckets


def _diag_prefix(code: str) -> str:
    """Return first 3 characters of ICD-10 code (category level)."""
    return code[:3] if code and len(code) >= 3 else ""


def featurize(claim: JoinedClaim) -> np.ndarray:
    """Convert a claim into a numerical feature vector for similarity computation.

    Feature dimensions (weighted):
    - Payer name (8 dims) × weight 2.5 — most predictive for denial behavior
    - Insurance type (3 dims) × weight 1.0
    - Procedure group (10 dims) × weight 3.0 — strongest clinical signal
    - CARC code (11 dims) × weight 1.5 — for denied claims only
    - Amount bucket (5 dims) × weight 0.5
    - Procedure code exact match seed (1 dim) × weight 2.0
    - Diagnosis prefix match seed (1 dim) × weight 1.5
    """
    payer = _normalize_payer(claim.payer_name)
    proc = claim.procedure_code
    proc_group = _PROCEDURE_TO_GROUP.get(proc, "Other")
    carc = claim.carc_code if claim.carc_code in _CARC_CODES else "other"
    amount = claim.claimed_amount
    diag = _diag_prefix(claim.principal_diagnosis)

    weights = []
    features = []

    # Payer (weight 2.5)
    features.extend(_one_hot(payer, _PAYERS))
    weights.extend([2.5] * len(_PAYERS))

    # Insurance type (weight 1.0)
    features.extend(_one_hot(claim.insurance_type, _INSURANCE_TYPES))
    weights.extend([1.0] * len(_INSURANCE_TYPES))

    # Procedure group (weight 3.0)
    features.extend(_one_hot(proc_group, _PROCEDURE_GROUPS))
    weights.extend([3.0] * len(_PROCEDURE_GROUPS))

    # CARC code (weight 1.5)
    features.extend(_one_hot(carc, _CARC_CODES))
    weights.extend([1.5] * len(_CARC_CODES))

    # Amount bucket (weight 0.5)
    features.extend(_amount_bucket(amount))
    weights.extend([0.5] * 5)

    # Exact procedure code — hash to single float (weight 2.0)
    proc_hash = float(hash(proc) % 1000) / 1000.0
    features.append(proc_hash)
    weights.append(2.0)

    # Diagnosis category prefix — hash to single float (weight 1.5)
    diag_hash = float(hash(diag) % 1000) / 1000.0
    features.append(diag_hash)
    weights.append(1.5)

    vec = np.array(features, dtype=np.float32)
    w = np.array(weights, dtype=np.float32)
    return vec * w


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _shared_features(query: JoinedClaim, candidate: JoinedClaim) -> list[str]:
    """Identify which features drove the similarity score."""
    shared = []
    if _normalize_payer(query.payer_name) == _normalize_payer(candidate.payer_name):
        shared.append(f"Same payer: {query.payer_name}")
    if query.insurance_type == candidate.insurance_type:
        shared.append(f"Same insurance type: {query.insurance_type}")
    if query.procedure_code == candidate.procedure_code:
        shared.append(f"Exact procedure match: CPT {query.procedure_code}")
    elif (
        _PROCEDURE_TO_GROUP.get(query.procedure_code, "Other")
        == _PROCEDURE_TO_GROUP.get(candidate.procedure_code, "Other")
    ):
        shared.append(
            f"Same procedure category: {_PROCEDURE_TO_GROUP.get(query.procedure_code, 'Other')}"
        )
    if query.carc_code and query.carc_code == candidate.carc_code:
        shared.append(f"Same denial reason: CARC {query.carc_code}")
    if (
        query.principal_diagnosis
        and candidate.principal_diagnosis
        and query.principal_diagnosis[:3] == candidate.principal_diagnosis[:3]
    ):
        shared.append(f"Same diagnosis category: {query.principal_diagnosis[:3]}xx")
    return shared


# ─── Pattern Matcher ──────────────────────────────────────────────────────────


class PatternMatcher:
    """Finds historically similar claims and detects systemic denial patterns.

    Design decision: We use weighted cosine similarity on interpretable feature vectors
    rather than neural embeddings. This gives us:
    - No additional API cost for embeddings
    - Full interpretability of similarity scores
    - Ability to explain exactly which features drove similarity
    - Consistent, deterministic results across runs
    """

    def __init__(self, historical_claims: list[JoinedClaim]):
        self.historical_claims = historical_claims
        self._vectors: dict[str, np.ndarray] = {}
        self._precompute_vectors()

    def _precompute_vectors(self) -> None:
        for claim in self.historical_claims:
            self._vectors[claim.claim_id] = featurize(claim)

    def _get_vector(self, claim: JoinedClaim) -> np.ndarray:
        if claim.claim_id not in self._vectors:
            self._vectors[claim.claim_id] = featurize(claim)
        return self._vectors[claim.claim_id]

    def find_similar(
        self,
        query: JoinedClaim,
        top_k: int = 5,
        min_similarity: float = 0.4,
        exclude_self: bool = True,
    ) -> tuple[list[SimilarClaim], list[SimilarClaim]]:
        """Return top-K similar paid and denied claims separately."""
        query_vec = self._get_vector(query)
        scored = []

        for candidate in self.historical_claims:
            if exclude_self and candidate.claim_id == query.claim_id:
                continue
            cand_vec = self._get_vector(candidate)
            score = cosine_similarity(query_vec, cand_vec)
            if score >= min_similarity:
                scored.append((score, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)

        paid_results: list[SimilarClaim] = []
        denied_results: list[SimilarClaim] = []

        for score, candidate in scored:
            outcome = "paid" if not candidate.is_denied else "denied"
            similar_claim = SimilarClaim(
                claim_id=candidate.claim_id,
                similarity_score=round(score, 3),
                outcome=outcome,
                payer_name=candidate.payer_name,
                procedure_code=candidate.procedure_code,
                principal_diagnosis=candidate.principal_diagnosis,
                claim_amount=candidate.claimed_amount,
                shared_features=_shared_features(query, candidate),
            )
            if outcome == "paid" and len(paid_results) < top_k:
                paid_results.append(similar_claim)
            elif outcome == "denied" and len(denied_results) < top_k:
                denied_results.append(similar_claim)

        return paid_results, denied_results

    def compute_payer_denial_rate(
        self, payer_name: str, procedure_code: str
    ) -> Optional[float]:
        """Compute denial rate for a specific payer+procedure combination."""
        matching = [
            c for c in self.historical_claims
            if (
                _normalize_payer(c.payer_name) == _normalize_payer(payer_name)
                and c.procedure_code == procedure_code
            )
        ]
        if len(matching) < 2:
            return None
        denied = sum(1 for c in matching if c.is_denied)
        return round(denied / len(matching), 2)

    def detect_systemic_pattern(
        self, payer_name: str, procedure_code: str, carc_code: str
    ) -> Optional[str]:
        """Detect if there is a systemic denial pattern for this payer+procedure+CARC combination."""
        rate = self.compute_payer_denial_rate(payer_name, procedure_code)
        if rate is None:
            return None

        payer_norm = _normalize_payer(payer_name)
        proc_group = _PROCEDURE_TO_GROUP.get(procedure_code, procedure_code)

        if rate >= 0.6:
            return (
                f"SYSTEMIC PATTERN: {payer_norm} denies {proc_group} (CPT {procedure_code}) "
                f"at {rate:.0%} rate. This is a systemic issue requiring process-level fix, "
                f"not just individual appeals."
            )
        elif rate >= 0.4:
            return (
                f"ELEVATED DENIAL RATE: {payer_norm} denies {proc_group} (CPT {procedure_code}) "
                f"at {rate:.0%} — above typical. Review documentation and coding practices."
            )
        else:
            return (
                f"Normal variation: {payer_norm} denial rate for CPT {procedure_code} "
                f"is {rate:.0%}. Individual appeals are appropriate."
            )

    def match(self, query: JoinedClaim, top_k: int = 5) -> PatternMatchResult:
        """Run full pattern matching analysis for a denied claim."""
        paid_similar, denied_similar = self.find_similar(query, top_k=top_k)

        payer_rate = self.compute_payer_denial_rate(query.payer_name, query.procedure_code)
        systemic_pattern = self.detect_systemic_pattern(
            query.payer_name, query.procedure_code, query.carc_code
        )

        # Determine recoverability adjustment based on historical evidence
        if paid_similar and not denied_similar:
            adjustment = "strengthened"
        elif denied_similar and not paid_similar:
            adjustment = "weakened"
        elif paid_similar and denied_similar:
            # Compare average similarity scores
            avg_paid = sum(c.similarity_score for c in paid_similar) / len(paid_similar)
            avg_denied = sum(c.similarity_score for c in denied_similar) / len(denied_similar)
            adjustment = "strengthened" if avg_paid > avg_denied else "weakened"
        else:
            adjustment = "neutral"

        # Build pattern summary
        top_paid_id = paid_similar[0].claim_id if paid_similar else None
        summary_parts = []
        if paid_similar:
            summary_parts.append(
                f"Found {len(paid_similar)} similar PAID claims (highest similarity: "
                f"{paid_similar[0].similarity_score:.0%} — {paid_similar[0].payer_name}, "
                f"CPT {paid_similar[0].procedure_code})"
            )
        if denied_similar:
            summary_parts.append(
                f"{len(denied_similar)} similar DENIED claims with CARC "
                f"{denied_similar[0].shared_features[-1] if denied_similar[0].shared_features else 'N/A'}"
            )
        if payer_rate is not None:
            summary_parts.append(f"Payer denial rate for this procedure: {payer_rate:.0%}")
        if systemic_pattern:
            summary_parts.append(systemic_pattern)
        if not summary_parts:
            summary_parts.append("Insufficient historical data for pattern analysis.")

        pattern_summary = " | ".join(summary_parts)

        return PatternMatchResult(
            claim_id=query.claim_id,
            similar_paid_claims=paid_similar,
            similar_denied_claims=denied_similar,
            pattern_summary=pattern_summary,
            payer_denial_rate=payer_rate,
            procedure_denial_pattern=systemic_pattern,
            recoverability_adjustment=adjustment,
            top_matching_paid_claim_id=top_paid_id,
        )
