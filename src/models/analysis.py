"""Output models for the three analysis modules."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RecoverabilityVerdict(str, Enum):
    RECOVERABLE = "recoverable"
    NOT_RECOVERABLE = "not_recoverable"
    NEEDS_REVIEW = "needs_review"


class SupportingEvidence(BaseModel):
    """A single piece of evidence from the claim data supporting the analysis."""

    field_name: str = Field(description="The EDI field name (e.g. ec_ServiceDateFrom)")
    field_value: str = Field(description="The actual value found in the claim")
    significance: str = Field(description="Why this value matters to the denial analysis")


class PreAnalysisFindings(BaseModel):
    """Rule-based pre-analysis results fed into the LLM as grounded context."""

    carc_code: str
    carc_description: str
    carc_category: str
    days_since_service: Optional[int] = None
    filing_limit_days: Optional[int] = None
    is_genuinely_late: Optional[bool] = None
    has_delay_reason_code: Optional[bool] = None
    delay_reason_description: Optional[str] = None
    prior_auth_on_claim: Optional[bool] = None
    prior_auth_on_remittance: Optional[bool] = None
    auth_number: Optional[str] = None
    remark_codes: Optional[list[str]] = None
    rarc_descriptions: Optional[list[str]] = None
    lcd_referenced: Optional[bool] = None
    notes: list[str] = Field(default_factory=list)


class RootCauseAnalysis(BaseModel):
    """Structured output for Problem 1: per-claim denial root cause analysis."""

    claim_id: str
    denial_root_cause: str = Field(
        description="Human-readable explanation of WHY this claim was denied (beyond the CARC code)"
    )
    carc_code: str
    carc_interpretation: str = Field(
        description="What the CARC code means in the context of THIS specific claim"
    )
    rarc_codes: list[str] = Field(default_factory=list)
    rarc_interpretation: Optional[str] = None
    recoverability_verdict: RecoverabilityVerdict
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in the recoverability verdict (0=uncertain, 1=highly confident)"
    )
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)
    recommended_action: str = Field(
        description="Specific next step the billing team should take"
    )
    appeal_strategy: Optional[str] = Field(
        default=None,
        description="How to frame the appeal if applicable"
    )
    appeal_deadline_estimate: Optional[str] = Field(
        default=None,
        description="Estimated appeal deadline based on payer type and denial date"
    )
    pre_analysis: Optional[PreAnalysisFindings] = Field(
        default=None,
        description="Rule-based findings that grounded the LLM analysis"
    )


class SimilarClaim(BaseModel):
    """A historically similar claim used for pattern matching."""

    claim_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    outcome: str = Field(description="paid or denied")
    payer_name: str
    procedure_code: str
    principal_diagnosis: str
    claim_amount: float
    shared_features: list[str] = Field(
        description="Which features drove the similarity score"
    )


class PatternMatchResult(BaseModel):
    """Structured output for Problem 2: historical pattern matching for a single denied claim."""

    claim_id: str
    similar_paid_claims: list[SimilarClaim] = Field(default_factory=list)
    similar_denied_claims: list[SimilarClaim] = Field(default_factory=list)
    pattern_summary: str = Field(
        description="Natural language summary of patterns found"
    )
    payer_denial_rate: Optional[float] = Field(
        default=None,
        description="This payer's denial rate for this procedure+diagnosis combination"
    )
    procedure_denial_pattern: Optional[str] = Field(
        default=None,
        description="Detected systemic pattern for this procedure code"
    )
    recoverability_adjustment: str = Field(
        description="strengthened / weakened / neutral — how historical data adjusts recoverability"
    )
    top_matching_paid_claim_id: Optional[str] = Field(
        default=None,
        description="The single most similar paid claim to reference in appeal"
    )


class DenialCluster(BaseModel):
    """A group of similar denied claims identified by the clustering module."""

    cluster_id: str
    cluster_label: str = Field(
        description="Human-readable label e.g. 'Aetna MRI Medical Necessity Denials'"
    )
    claim_ids: list[str]
    claim_count: int
    total_denied_amount: float
    primary_carc_code: str
    primary_carc_description: str
    primary_payer: str
    procedure_codes: list[str]
    historical_appeal_success_rate: float = Field(
        ge=0.0, le=1.0,
        description="Estimated success rate based on similar historical paid claims"
    )
    recoverable_amount_estimate: float = Field(
        description="total_denied_amount * historical_appeal_success_rate"
    )
    priority_score: float = Field(
        description="Composite score for billing team prioritization"
    )
    recommended_batch_action: str = Field(
        description="What the billing team should do for this entire cluster at once"
    )
    billing_team_summary: str = Field(
        description="Plain English summary for a billing manager"
    )


class BatchIntelligenceReport(BaseModel):
    """Structured output for Problem 3: full batch analysis of all denied claims."""

    total_claims_analyzed: int
    total_denied_claims: int
    total_denied_amount: float
    total_recoverable_estimate: float
    clusters: list[DenialCluster]
    top_priority_cluster_id: str
    executive_summary: str = Field(
        description="One-paragraph summary for a billing director"
    )
    quick_wins: list[str] = Field(
        description="3-5 highest ROI actions the team should take first"
    )

    def total_claimed_amount_range_ok(self) -> bool:
        """Sanity check: recoverable estimate should not exceed total denied amount."""
        return self.total_recoverable_estimate <= self.total_denied_amount + 0.01
