from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RecoverabilityVerdict(str, Enum):
    RECOVERABLE = "recoverable"
    NOT_RECOVERABLE = "not_recoverable"
    NEEDS_REVIEW = "needs_review"


class SupportingEvidence(BaseModel):
    field_name: str
    field_value: str
    significance: str


class PreAnalysisFindings(BaseModel):
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
    claim_id: str
    denial_root_cause: str
    carc_code: str
    carc_interpretation: str
    rarc_codes: list[str] = Field(default_factory=list)
    rarc_interpretation: Optional[str] = None
    recoverability_verdict: RecoverabilityVerdict
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)
    recommended_action: str
    appeal_strategy: Optional[str] = None
    appeal_deadline_estimate: Optional[str] = None
    pre_analysis: Optional[PreAnalysisFindings] = None


class SimilarClaim(BaseModel):
    claim_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    outcome: str  # "paid" or "denied"
    payer_name: str
    procedure_code: str
    principal_diagnosis: str
    claim_amount: float
    shared_features: list[str]


class PatternMatchResult(BaseModel):
    claim_id: str
    similar_paid_claims: list[SimilarClaim] = Field(default_factory=list)
    similar_denied_claims: list[SimilarClaim] = Field(default_factory=list)
    pattern_summary: str
    payer_denial_rate: Optional[float] = None
    procedure_denial_pattern: Optional[str] = None
    recoverability_adjustment: str  # strengthened / weakened / neutral
    top_matching_paid_claim_id: Optional[str] = None


class DenialCluster(BaseModel):
    cluster_id: str
    cluster_label: str
    claim_ids: list[str]
    claim_count: int
    total_denied_amount: float
    primary_carc_code: str
    primary_carc_description: str
    primary_payer: str
    procedure_codes: list[str]
    historical_appeal_success_rate: float = Field(ge=0.0, le=1.0)
    recoverable_amount_estimate: float
    priority_score: float
    recommended_batch_action: str
    billing_team_summary: str


class BatchIntelligenceReport(BaseModel):
    total_claims_analyzed: int
    total_denied_claims: int
    total_denied_amount: float
    total_recoverable_estimate: float
    clusters: list[DenialCluster]
    top_priority_cluster_id: str
    executive_summary: str
    quick_wins: list[str]

    def total_claimed_amount_range_ok(self) -> bool:
        return self.total_recoverable_estimate <= self.total_denied_amount + 0.01
