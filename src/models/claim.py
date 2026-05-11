"""Pydantic models for EDI 835 (remittance) and EDI 837 (claim submission) data."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Claim835(BaseModel):
    """EDI 835 Remittance Advice — the payer's response to a submitted claim."""

    # Payment-level fields (cp_ prefix) — top-level remittance info
    cp_PayerName: Optional[str] = None
    cp_PayerID: Optional[str] = None
    cp_PayeeName: Optional[str] = None
    cp_PaymentAmount: Optional[float] = None
    cp_TotalClaimCount: Optional[int] = None
    cp_TotalDeniedChargeAmount: Optional[float] = None
    cp_TotalClaimChargeAmount: Optional[float] = None
    cp_EffectiveDate: Optional[str] = None

    # Claim-level fields (pc_ prefix) — per-claim adjudication details
    pc_ClaimID: str
    pc_ClaimStatus: str = Field(
        description="1=Primary, 2=Secondary, 4=Denied, 19=Primary+Forwarded, 22=Reversal"
    )
    pc_ClaimAmount: float
    pc_ClaimPaid: float
    pc_PatientResponsibility: Optional[float] = None
    pc_InsuranceType: Optional[str] = None
    pc_StatementBegin: Optional[str] = None
    pc_StatementEnd: Optional[str] = None
    pc_ReceivedDate: Optional[str] = None
    pc_PriorAuthNum: Optional[str] = None
    pc_PatientLast: Optional[str] = None
    pc_PatientFirst: Optional[str] = None
    pc_RenderingID: Optional[str] = None

    # Line-level fields (pcl_ prefix) — individual service line details
    pcl_ProcedureCode: Optional[str] = None
    pcl_ProcedureModifier1: Optional[str] = None
    pcl_ProcedureModifier2: Optional[str] = None
    pcl_ProcedureModifier3: Optional[str] = None
    pcl_ProcedureModifier4: Optional[str] = None
    pcl_ChargedAmount: Optional[float] = None
    pcl_PaidAmount: Optional[float] = None
    pcl_AllowedAmount: Optional[float] = None
    pcl_ServiceDate: Optional[str] = None
    pcl_RemarkCodes: Optional[str] = None

    # Adjustment fields (pcla_ prefix) — reason codes for payment adjustments
    pcla_AdjustmentGroup: Optional[str] = Field(
        default=None,
        description="CO=Contractual, PR=Patient Responsibility, OA=Other, PI=Payer Initiated"
    )
    pcla_AdjustmentReason: Optional[str] = Field(
        default=None,
        description="CARC code explaining WHY the adjustment was made"
    )
    pcla_AdjustmentAmount: Optional[float] = None
    pcla_AdjustmentQty: Optional[float] = None

    @property
    def is_denied(self) -> bool:
        return self.pc_ClaimStatus == "4"

    @property
    def denial_amount(self) -> float:
        return self.pc_ClaimAmount if self.is_denied else 0.0


class Claim837(BaseModel):
    """EDI 837 Claim Submission — what the provider originally billed."""

    # Claim-level fields (ec_ prefix)
    ec_ClaimNo: str
    ec_Amount: Optional[float] = None
    ec_PlaceOfService: Optional[str] = Field(
        default=None,
        description="11=Office, 21=Inpatient Hospital, 22=Outpatient Hospital, 23=ER"
    )
    ec_PayerName: Optional[str] = None
    ec_PayerID: Optional[str] = None
    ec_InsuranceType: Optional[str] = None
    ec_PrincipalDiagnosis: Optional[str] = Field(
        default=None, description="Primary ICD-10 diagnosis code"
    )
    ec_Diag2: Optional[str] = None
    ec_Diag3: Optional[str] = None
    ec_Diag4: Optional[str] = None
    ec_Diag5: Optional[str] = None
    ec_BillProvNPI: Optional[str] = None
    ec_RendProvNPI: Optional[str] = None
    ec_RendProvSpecialty: Optional[str] = None
    ec_ServiceDateFrom: Optional[str] = None
    ec_ServiceDateTo: Optional[str] = None
    ec_PriorAuthorization: Optional[str] = None
    ec_TypeOfBill: Optional[str] = None
    ec_ClaimFrequency: Optional[str] = Field(
        default=None, description="1=Original, 7=Replacement, 8=Void"
    )
    ec_DelayReasonCode: Optional[str] = Field(
        default=None,
        description="Reason code explaining why claim was submitted late"
    )
    ec_PatientRelationship: Optional[str] = None
    ec_SubscriberID: Optional[str] = None

    @property
    def all_diagnoses(self) -> list[str]:
        """Return all non-empty diagnosis codes."""
        codes = [
            self.ec_PrincipalDiagnosis,
            self.ec_Diag2,
            self.ec_Diag3,
            self.ec_Diag4,
            self.ec_Diag5,
        ]
        return [c for c in codes if c and c.strip()]

    @property
    def has_prior_authorization(self) -> bool:
        return bool(self.ec_PriorAuthorization and self.ec_PriorAuthorization.strip())


class JoinedClaim(BaseModel):
    """Combined 835 + 837 data — full picture of what was billed and how it was adjudicated."""

    claim_id: str
    claim_835: Claim835
    claim_837: Claim837
    label: Optional[str] = None  # human-readable scenario label for synthetic data

    @property
    def is_denied(self) -> bool:
        return self.claim_835.is_denied

    @property
    def payer_name(self) -> str:
        return (
            self.claim_835.cp_PayerName
            or self.claim_837.ec_PayerName
            or "Unknown Payer"
        )

    @property
    def insurance_type(self) -> str:
        return (
            self.claim_835.pc_InsuranceType
            or self.claim_837.ec_InsuranceType
            or "Unknown"
        )

    @property
    def procedure_code(self) -> str:
        return self.claim_835.pcl_ProcedureCode or ""

    @property
    def carc_code(self) -> str:
        return self.claim_835.pcla_AdjustmentReason or ""

    @property
    def principal_diagnosis(self) -> str:
        return self.claim_837.ec_PrincipalDiagnosis or ""

    @property
    def claimed_amount(self) -> float:
        return self.claim_835.pc_ClaimAmount
