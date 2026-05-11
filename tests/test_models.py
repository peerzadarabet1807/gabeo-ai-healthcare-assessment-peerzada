"""Tests for Pydantic data models."""

import pytest
from src.models.claim import Claim835, Claim837, JoinedClaim
from src.models.analysis import RecoverabilityVerdict, SupportingEvidence, RootCauseAnalysis


def test_claim_835_is_denied():
    claim = Claim835(pc_ClaimID="CLM-001", pc_ClaimStatus="4", pc_ClaimAmount=1000.0, pc_ClaimPaid=0.0)
    assert claim.is_denied is True


def test_claim_835_is_not_denied():
    claim = Claim835(pc_ClaimID="CLM-001", pc_ClaimStatus="1", pc_ClaimAmount=1000.0, pc_ClaimPaid=800.0)
    assert claim.is_denied is False


def test_claim_837_all_diagnoses():
    claim = Claim837(
        ec_ClaimNo="CLM-001",
        ec_PrincipalDiagnosis="M17.11",
        ec_Diag2="I10",
        ec_Diag3="",
    )
    assert claim.all_diagnoses == ["M17.11", "I10"]


def test_claim_837_has_prior_auth():
    claim = Claim837(ec_ClaimNo="CLM-001", ec_PriorAuthorization="AUTH-123")
    assert claim.has_prior_authorization is True


def test_claim_837_no_prior_auth():
    claim = Claim837(ec_ClaimNo="CLM-001", ec_PriorAuthorization="")
    assert claim.has_prior_authorization is False


def test_joined_claim_properties():
    c835 = Claim835(
        pc_ClaimID="CLM-001",
        pc_ClaimStatus="4",
        pc_ClaimAmount=5000.0,
        pc_ClaimPaid=0.0,
        cp_PayerName="Aetna",
        pc_InsuranceType="Commercial",
        pcl_ProcedureCode="72148",
        pcla_AdjustmentReason="50",
    )
    c837 = Claim837(
        ec_ClaimNo="CLM-001",
        ec_PrincipalDiagnosis="M54.5",
        ec_InsuranceType="Commercial",
    )
    claim = JoinedClaim(claim_id="CLM-001", claim_835=c835, claim_837=c837)

    assert claim.is_denied is True
    assert claim.payer_name == "Aetna"
    assert claim.insurance_type == "Commercial"
    assert claim.procedure_code == "72148"
    assert claim.carc_code == "50"
    assert claim.principal_diagnosis == "M54.5"
    assert claim.claimed_amount == 5000.0


def test_recoverability_verdict_enum():
    assert RecoverabilityVerdict.RECOVERABLE.value == "recoverable"
    assert RecoverabilityVerdict.NOT_RECOVERABLE.value == "not_recoverable"
    assert RecoverabilityVerdict.NEEDS_REVIEW.value == "needs_review"


def test_root_cause_analysis_model():
    rca = RootCauseAnalysis(
        claim_id="CLM-001",
        denial_root_cause="Test root cause",
        carc_code="29",
        carc_interpretation="Test interpretation",
        recoverability_verdict=RecoverabilityVerdict.NOT_RECOVERABLE,
        confidence_score=0.95,
        supporting_evidence=[
            SupportingEvidence(
                field_name="ec_ServiceDateFrom",
                field_value="2025-06-15",
                significance="Service was 278 days before received date",
            )
        ],
        recommended_action="Write off the claim",
    )
    assert rca.claim_id == "CLM-001"
    assert rca.confidence_score == 0.95
    assert len(rca.supporting_evidence) == 1
