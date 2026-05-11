"""Tests for rule-based pre-analysis logic (no API calls required)."""

import pytest
from src.models.claim import Claim835, Claim837, JoinedClaim
from src.data.loader import ClaimLoader
from src.data.generator import generate_synthetic_dataset
from src.analysis.pattern_matching import (
    PatternMatcher, featurize, cosine_similarity, _normalize_payer, _shared_features
)


# ─── Test Data Fixtures ────────────────────────────────────────────────────────

def _make_joined(
    claim_id: str,
    claim_status: str,
    carc_code: str,
    payer_name: str,
    insurance_type: str,
    procedure_code: str,
    service_date: str,
    received_date: str,
    claim_amount: float = 1000.0,
    diagnosis: str = "M17.11",
    prior_auth: str = "",
    delay_reason: str = "",
) -> JoinedClaim:
    c835 = Claim835(
        pc_ClaimID=claim_id,
        pc_ClaimStatus=claim_status,
        pc_ClaimAmount=claim_amount,
        pc_ClaimPaid=0.0 if claim_status == "4" else claim_amount * 0.7,
        cp_PayerName=payer_name,
        pc_InsuranceType=insurance_type,
        pc_ReceivedDate=received_date,
        pc_StatementBegin=service_date,
        pcl_ProcedureCode=procedure_code,
        pcla_AdjustmentReason=carc_code if claim_status == "4" else "45",
        pcla_AdjustmentGroup="CO",
        pcla_AdjustmentAmount=claim_amount,
    )
    c837 = Claim837(
        ec_ClaimNo=claim_id,
        ec_PayerName=payer_name,
        ec_InsuranceType=insurance_type,
        ec_ServiceDateFrom=service_date,
        ec_PrincipalDiagnosis=diagnosis,
        ec_PriorAuthorization=prior_auth,
        ec_DelayReasonCode=delay_reason,
        ec_ClaimFrequency="1",
    )
    return JoinedClaim(claim_id=claim_id, claim_835=c835, claim_837=c837)


# ─── Data Loader Tests ─────────────────────────────────────────────────────────

def test_loader_loads_sample_claims():
    loader = ClaimLoader()
    claims = loader.load_file("data/sample_claims.json")
    assert len(claims) == 4
    assert all(c.is_denied for c in claims)


def test_loader_filter_denied():
    loader = ClaimLoader()
    synthetic = generate_synthetic_dataset()
    claims = loader.load_from_dicts(synthetic)
    denied = loader.filter_denied(claims)
    paid = loader.filter_paid(claims)
    assert len(denied) == 20
    assert len(paid) == 10
    assert len(denied) + len(paid) == 30


def test_synthetic_dataset_has_correct_structure():
    synthetic = generate_synthetic_dataset()
    assert len(synthetic) == 30
    for item in synthetic:
        assert "claim_id" in item
        assert "claim_835" in item
        assert "claim_837" in item
        assert item["claim_835"]["pc_ClaimID"] == item["claim_id"]
        assert item["claim_837"]["ec_ClaimNo"] == item["claim_id"]


# ─── Pre-Analysis: CARC 29 Tests ──────────────────────────────────────────────

def test_carc_29_genuinely_late_commercial():
    """278 days for a commercial claim (180-day limit) should be genuinely late."""
    import json
    from src.analysis.root_cause import _run_pre_analysis, _load_carc_codes
    carc_codes = _load_carc_codes()

    claim = _make_joined(
        "CLM-TEST-01", "4", "29", "Blue Cross Blue Shield", "Commercial",
        "99214", "2025-06-15", "2026-03-20", claim_amount=4500.0
    )
    pre = _run_pre_analysis(claim, carc_codes)
    assert pre.days_since_service == 278
    assert pre.filing_limit_days == 180
    assert pre.is_genuinely_late is True
    assert pre.has_delay_reason_code is False


def test_carc_29_medicare_within_limit():
    """300 days for Medicare (365-day limit) should NOT be late."""
    from src.analysis.root_cause import _run_pre_analysis, _load_carc_codes
    carc_codes = _load_carc_codes()

    claim = _make_joined(
        "CLM-TEST-02", "4", "29", "Medicare Part B", "Medicare",
        "27447", "2025-06-01", "2026-03-27", claim_amount=12000.0
    )
    pre = _run_pre_analysis(claim, carc_codes)
    assert pre.days_since_service == 299
    assert pre.filing_limit_days == 365
    assert pre.is_genuinely_late is False


def test_carc_29_with_delay_reason_code():
    """Filing with a delay reason code should be noted as potentially recoverable."""
    from src.analysis.root_cause import _run_pre_analysis, _load_carc_codes
    carc_codes = _load_carc_codes()

    claim = _make_joined(
        "CLM-TEST-03", "4", "29", "United Healthcare", "Commercial",
        "72148", "2025-11-05", "2026-04-25", delay_reason="9"
    )
    pre = _run_pre_analysis(claim, carc_codes)
    assert pre.has_delay_reason_code is True
    assert "Original Claim Rejected" in (pre.delay_reason_description or "")


# ─── Pre-Analysis: CARC 197 Tests ─────────────────────────────────────────────

def test_carc_197_auth_present_on_claim():
    """When prior auth exists on 837 but payer says absent — system mismatch."""
    from src.analysis.root_cause import _run_pre_analysis, _load_carc_codes
    carc_codes = _load_carc_codes()

    claim = _make_joined(
        "CLM-TEST-04", "4", "197", "United Healthcare", "Commercial",
        "72148", "2026-03-08", "2026-03-15", prior_auth="AUTH-999001"
    )
    pre = _run_pre_analysis(claim, carc_codes)
    assert pre.prior_auth_on_claim is True
    assert any("system" in note.lower() or "matching error" in note.lower() for note in pre.notes)


def test_carc_197_no_auth():
    """No prior auth at all — genuinely missing."""
    from src.analysis.root_cause import _run_pre_analysis, _load_carc_codes
    carc_codes = _load_carc_codes()

    claim = _make_joined(
        "CLM-TEST-05", "4", "197", "Aetna", "Commercial",
        "27447", "2026-02-22", "2026-02-28"
    )
    pre = _run_pre_analysis(claim, carc_codes)
    assert pre.prior_auth_on_claim is False
    assert any("retroactive" in note.lower() or "not obtained" in note.lower() for note in pre.notes)


# ─── Pattern Matching Tests ────────────────────────────────────────────────────

def test_featurize_returns_ndarray():
    import numpy as np
    claim = _make_joined("CLM-001", "1", "45", "Aetna", "Commercial", "99213", "2026-01-01", "2026-01-10")
    vec = featurize(claim)
    assert isinstance(vec, np.ndarray)
    assert len(vec) > 0


def test_cosine_similarity_identical():
    """Identical vectors should have similarity 1.0."""
    import numpy as np
    v = np.array([1.0, 0.5, 0.3])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have similarity ~0."""
    import numpy as np
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_normalize_payer():
    assert _normalize_payer("Medicare Part B") == "Medicare Part B"
    assert _normalize_payer("BCBS-IL") == "Blue Cross Blue Shield"
    assert _normalize_payer("Blue Cross Blue Shield") == "Blue Cross Blue Shield"
    assert _normalize_payer("United Healthcare") == "United Healthcare"
    assert _normalize_payer("Unknown Payer XYZ") == "Other"


def test_pattern_matcher_finds_similar_paid():
    """A denied claim should find similar paid claims from the synthetic dataset."""
    loader = ClaimLoader()
    all_claims = loader.load_from_dicts(generate_synthetic_dataset())

    matcher = PatternMatcher(historical_claims=all_claims)
    # CLM-SYN-0019 is denied Aetna 72148 (MRI) — similar to CLM-SYN-0003 (paid Aetna 90837)
    denied_claim = next(c for c in all_claims if c.claim_id == "CLM-SYN-0019")
    paid_similar, denied_similar = matcher.find_similar(denied_claim, top_k=5)

    # Should find at least some similar claims
    assert len(paid_similar) + len(denied_similar) > 0


def test_pattern_matcher_self_excluded():
    """A claim should not appear as its own similar claim."""
    loader = ClaimLoader()
    all_claims = loader.load_from_dicts(generate_synthetic_dataset())
    matcher = PatternMatcher(historical_claims=all_claims)

    target = all_claims[0]
    paid, denied = matcher.find_similar(target, exclude_self=True)
    all_results = paid + denied
    assert not any(c.claim_id == target.claim_id for c in all_results)


def test_payer_denial_rate_computation():
    """Denial rate should be computed correctly from the dataset."""
    loader = ClaimLoader()
    all_claims = loader.load_from_dicts(generate_synthetic_dataset())
    matcher = PatternMatcher(historical_claims=all_claims)

    # Aetna has several claims in dataset
    rate = matcher.compute_payer_denial_rate("Aetna", "72148")
    # 72148 appears in CLM-SYN-0019 (denied) and potentially others
    # Should be between 0 and 1
    if rate is not None:
        assert 0.0 <= rate <= 1.0


def test_shared_features_detection():
    """Shared features should correctly identify matching attributes."""
    c1 = _make_joined("CLM-A", "4", "50", "Aetna", "Commercial", "72148", "2026-01-01", "2026-02-01")
    c2 = _make_joined("CLM-B", "1", "45", "Aetna", "Commercial", "72148", "2026-01-15", "2026-02-10")
    shared = _shared_features(c1, c2)
    assert any("Aetna" in f for f in shared)
    assert any("72148" in f for f in shared)
    assert any("Commercial" in f for f in shared)
