from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

from src.models.claim import JoinedClaim
from src.models.analysis import (
    PreAnalysisFindings,
    RootCauseAnalysis,
    RecoverabilityVerdict,
    SupportingEvidence,
)


CARC_CODES_PATH = Path(__file__).parent.parent.parent / "data" / "carc_codes.json"

_CARC_CODES: Optional[dict] = None

_SYSTEM_PROMPT = """You are a senior healthcare revenue cycle management (RCM) specialist with 15+ years of experience analyzing insurance claim denials. You have deep expertise in:
- EDI 835 (remittance advice) and EDI 837 (claim submission) data formats
- CARC (Claim Adjustment Reason Code) and RARC (Remittance Advice Remark Code) interpretation
- Payer-specific policies: Medicare (CMS), Medicaid, and major commercial payers (BCBS, Aetna, UHC, Cigna, Humana)
- Federal timely filing deadlines (Medicare = 365 days, Medicaid = 365 days, Commercial = typically 90-180 days)
- Prior authorization requirements and retroactive authorization processes
- Medical necessity criteria for common procedures (E&M, radiology, surgery)
- NCCI (National Correct Coding Initiative) bundling rules
- ICD-10 diagnosis codes and CPT/HCPCS procedure codes

CARC REFERENCE (for adjudication context):
- CARC 4: Modifier inconsistency or missing required modifier → Review CPT/modifier pairing and resubmit
- CARC 16: Missing/incomplete information → Identify via RARC and resubmit corrected claim
- CARC 18: Exact duplicate claim → Verify original adjudication; if genuinely different, appeal with proof
- CARC 29: Timely filing expired → Analyze filing gap vs. payer limit; check for delay reason codes
- CARC 45: Contractual adjustment → Write-off per contract, cannot bill patient (CO group)
- CARC 50: Not medically necessary → Gather clinical documentation, appeal with physician attestation
- CARC 96: Non-covered service → Verify plan coverage; ABN notice may allow patient billing
- CARC 97: Bundled into another service → Check NCCI edits; separate payability requires modifiers
- CARC 197: Missing prior authorization → Request retroactive auth; emergency exceptions may apply
- CARC 252: Attachment required → Submit operative report, clinical notes, or other documentation

RARC REFERENCE:
- N20: Missing/incomplete HCPCS code
- N30: Patient identity cannot be confirmed
- N56: Procedure code invalid for the date of service
- N115/N386: Denial based on Local Coverage Determination (LCD) — clinical documentation required
- MA01: Appeal rights notification — 6 months from date of notice
- MA04: Secondary payment requires primary payer EOB
- M86: Service already paid or adjusted in prior claim

TIMELY FILING RULES:
- Medicare (Part A & B): 365 days from date of service
- Medicaid: 365 days (varies by state; IL Medicaid = 180 days)
- Commercial payers: typically 90-180 days from date of service (varies by contract)
- Secondary claims: filing window may start from primary payer's EOB date, not service date

PRIOR AUTH GUIDANCE:
- High-cost surgeries (knee replacement, shoulder arthroscopy, spinal surgery) almost always require auth
- Advanced imaging (MRI, CT with contrast) typically requires auth from commercial payers
- Mental health outpatient usually does not require auth but has session limits
- Emergency services are generally exempt from prior auth requirements

MEDICAL NECESSITY STANDARDS:
- MRI for back pain (M54.x) requires documentation of 4-6 weeks conservative treatment failure
- Inpatient admission must meet InterQual or Milliman criteria for the condition
- Repeated procedures (injections, therapy) require ongoing documentation of functional improvement

Your task is to analyze a denied healthcare claim and provide a structured, evidence-based assessment.

ANALYSIS METHODOLOGY:
1. Identify the CARC code and what it means for THIS specific claim (not just the generic definition)
2. Examine the rule-based pre-analysis findings — these are computed facts about the claim (filing days, auth presence, etc.)
3. Cross-reference 835 and 837 fields to find inconsistencies or supporting evidence
4. Determine recoverability based on the specific combination of factors
5. Assign a confidence score reflecting how certain you are, based on the evidence available
6. Recommend a specific, actionable next step (not generic advice)

RECOVERABILITY FRAMEWORK:
- "recoverable": Strong evidence the claim can be re-billed, corrected, or successfully appealed
- "not_recoverable": The denial is valid and there is no viable appeal pathway
- "needs_review": Borderline case requiring human review (e.g., partial evidence, conflicting signals)

CONFIDENCE SCORING:
- 0.9-1.0: All evidence clearly supports the verdict; unambiguous
- 0.7-0.89: Strong evidence with minor uncertainties
- 0.5-0.69: Mixed signals; verdict is best estimate but requires verification
- 0.3-0.49: Significant uncertainty; verdict is tentative
- Below 0.3: Insufficient evidence to make a reliable determination"""


def _load_carc_codes() -> dict:
    global _CARC_CODES
    if _CARC_CODES is None:
        with open(CARC_CODES_PATH, "r", encoding="utf-8") as f:
            _CARC_CODES = json.load(f)
    return _CARC_CODES


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str or not date_str.strip():
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _preanalyze_carc_29(claim: JoinedClaim, carc_codes: dict) -> dict:
    service_date = _parse_date(claim.claim_837.ec_ServiceDateFrom)
    received_date = _parse_date(claim.claim_835.pc_ReceivedDate)

    days_elapsed = None
    if service_date and received_date:
        days_elapsed = (received_date - service_date).days

    insurance_type = claim.insurance_type
    filing_limit = carc_codes["filing_limits_days"].get(insurance_type, 180)
    is_late = (days_elapsed > filing_limit) if days_elapsed is not None else None

    delay_code = (claim.claim_837.ec_DelayReasonCode or "").strip()
    delay_desc = carc_codes["delay_reason_codes"].get(delay_code) if delay_code else None

    notes = []
    if days_elapsed is not None:
        notes.append(
            f"Filed {days_elapsed} days after service date; {insurance_type} filing limit is {filing_limit} days."
        )
    if delay_code:
        notes.append(f"Delay reason code {delay_code} present: '{delay_desc}' — may justify late filing.")
    if is_late is False:
        notes.append("Claim was NOT actually filed late — denial may be erroneous; appeal immediately.")
    if is_late and not delay_code:
        notes.append("Filing was genuinely late with no documented delay reason — low recoverability.")

    remark = (claim.claim_835.pcl_RemarkCodes or "").upper()
    if "MA04" in remark:
        notes.append(
            "RARC MA04 present — this may be a secondary claim; timely filing may start from primary EOB date."
        )

    return {
        "days_since_service": days_elapsed,
        "filing_limit_days": filing_limit,
        "is_genuinely_late": is_late,
        "has_delay_reason_code": bool(delay_code),
        "delay_reason_description": delay_desc,
        "notes": notes,
    }


def _preanalyze_carc_16(claim: JoinedClaim, carc_codes: dict) -> dict:
    remark_raw = (claim.claim_835.pcl_RemarkCodes or "").strip()
    remark_codes = [r.strip() for r in remark_raw.replace(",", " ").split() if r.strip()]
    rarc_ref = carc_codes.get("rarc_codes", {})
    rarc_descriptions = [
        f"{r}: {rarc_ref[r]['description']}" for r in remark_codes if r in rarc_ref
    ]

    notes = []
    if "N20" in remark_codes:
        notes.append("N20: HCPCS/procedure code is missing or invalid — resubmit with correct procedure code.")
    if "N56" in remark_codes:
        notes.append("N56: Procedure code invalid for this date of service — check code validity timeline.")
    if "MA01" in remark_codes:
        notes.append("MA01: Appeal rights noted — 6 months from denial date to file appeal.")
    if not remark_codes:
        notes.append("No RARC codes present — contact payer to identify the specific missing information.")

    if not claim.claim_837.ec_RendProvNPI or not claim.claim_837.ec_RendProvNPI.strip():
        notes.append("Rendering provider NPI is missing on the 837 — likely cause of CARC 16.")

    return {
        "remark_codes": remark_codes,
        "rarc_descriptions": rarc_descriptions,
        "notes": notes,
    }


def _preanalyze_carc_50(claim: JoinedClaim, carc_codes: dict) -> dict:
    remark_raw = (claim.claim_835.pcl_RemarkCodes or "").strip()
    remark_codes = [r.strip() for r in remark_raw.replace(",", " ").split() if r.strip()]
    lcd_referenced = any(c in remark_codes for c in ("N386", "N115"))

    notes = []
    if lcd_referenced:
        notes.append(
            "LCD referenced — payer is using a Local Coverage Determination. "
            "Documentation must meet LCD criteria for this procedure."
        )

    proc = claim.claim_835.pcl_ProcedureCode or ""
    diag = claim.claim_837.ec_PrincipalDiagnosis or ""
    if proc in ("72148", "72141", "72156") and diag.startswith("M54"):
        notes.append(
            "MRI for back pain typically requires 4-6 weeks of documented conservative treatment "
            "(PT, NSAIDs, chiropractic). Without this documentation in the chart, appeal is difficult."
        )

    has_auth = claim.claim_837.has_prior_authorization
    if has_auth:
        notes.append(
            "Prior authorization was obtained — if payer pre-approved the service, "
            "a medical necessity denial may be erroneous and appealable."
        )
    else:
        notes.append("No prior authorization on file — provider must demonstrate medical necessity retroactively.")

    return {
        "remark_codes": remark_codes,
        "lcd_referenced": lcd_referenced,
        "prior_auth_on_claim": has_auth,
        "notes": notes,
    }


def _preanalyze_carc_197(claim: JoinedClaim, carc_codes: dict) -> dict:
    prior_auth_837 = (claim.claim_837.ec_PriorAuthorization or "").strip()
    prior_auth_835 = (claim.claim_835.pc_PriorAuthNum or "").strip()
    has_auth_837 = bool(prior_auth_837)
    has_auth_835 = bool(prior_auth_835)

    notes = []
    if has_auth_837 and not has_auth_835:
        notes.append(
            f"Auth number '{prior_auth_837}' is present on the 837 but not reflected in the 835. "
            "This may be a payer system matching error — appeal with the authorization number and date."
        )
    elif not has_auth_837:
        notes.append(
            "No prior authorization was obtained before service. "
            "Options: (1) request retroactive authorization with clinical documentation, "
            "(2) check if emergency exception applies, "
            "(3) verify if this procedure requires auth under the patient's specific plan."
        )

    proc = claim.claim_835.pcl_ProcedureCode or ""
    if proc in ("27447", "29827", "23472", "22612"):
        notes.append(
            f"Procedure {proc} is an elective surgery that virtually all commercial payers require "
            "prior authorization for. Retroactive auth approval is possible but not guaranteed."
        )

    return {
        "prior_auth_on_claim": has_auth_837,
        "auth_number": prior_auth_837 or prior_auth_835 or None,
        "prior_auth_on_remittance": has_auth_835,
        "notes": notes,
    }


def _preanalyze_carc_18(claim: JoinedClaim, carc_codes: dict) -> dict:
    remark_raw = (claim.claim_835.pcl_RemarkCodes or "").strip()
    remark_codes = [r.strip() for r in remark_raw.replace(",", " ").split() if r.strip()]

    notes = [
        "Verify whether the original claim (same service, provider, date) was actually paid. "
        "If original was also denied or is still pending, this denial may be incorrect.",
        "Check claim frequency code on the 837 — if ec_ClaimFrequency='7' (replacement) "
        "or '8' (void), it is not a true duplicate.",
    ]

    freq = (claim.claim_837.ec_ClaimFrequency or "").strip()
    if freq in ("7", "8"):
        notes.insert(0, f"ec_ClaimFrequency='{freq}' indicates this is a replacement/void — not a true duplicate. Appeal immediately.")

    return {
        "remark_codes": remark_codes,
        "claim_frequency": freq,
        "notes": notes,
    }


def _preanalyze_carc_4(claim: JoinedClaim, carc_codes: dict) -> dict:
    modifier = (claim.claim_835.pcl_ProcedureModifier1 or "").strip()
    proc = claim.claim_835.pcl_ProcedureCode or ""
    notes = []

    if modifier == "25":
        notes.append(
            "Modifier 25 is used to indicate a significant, separately identifiable E&M service "
            "on the same day as a procedure. The payer may be questioning whether the E&M was "
            "truly separate from the procedure. Provide documentation showing the E&M addressed a "
            "different condition or was medically necessary."
        )
    elif modifier:
        notes.append(
            f"Modifier '{modifier}' may not be appropriate for procedure {proc} under this payer's policy. "
            "Review the CPT code's modifier requirements."
        )
    else:
        notes.append(f"A required modifier is missing for procedure {proc}.")

    return {"modifier": modifier, "procedure_code": proc, "notes": notes}


def _run_pre_analysis(claim: JoinedClaim, carc_codes: dict) -> PreAnalysisFindings:
    carc = claim.carc_code
    carc_ref = carc_codes["carc_codes"].get(carc, {})

    base = {
        "carc_code": carc,
        "carc_description": carc_ref.get("description", "Unknown CARC code"),
        "carc_category": carc_ref.get("category", "Unknown"),
    }

    if carc == "29":
        extra = _preanalyze_carc_29(claim, carc_codes)
    elif carc == "16":
        extra = _preanalyze_carc_16(claim, carc_codes)
    elif carc == "50":
        extra = _preanalyze_carc_50(claim, carc_codes)
    elif carc == "197":
        extra = _preanalyze_carc_197(claim, carc_codes)
    elif carc == "18":
        extra = _preanalyze_carc_18(claim, carc_codes)
    elif carc == "4":
        extra = _preanalyze_carc_4(claim, carc_codes)
    else:
        remark_raw = (claim.claim_835.pcl_RemarkCodes or "").strip()
        remark_codes = [r.strip() for r in remark_raw.replace(",", " ").split() if r.strip()]
        extra = {
            "remark_codes": remark_codes,
            "notes": [f"CARC {carc}: {carc_ref.get('recovery_strategy', 'Review claim details and payer policy.')}"],
        }

    return PreAnalysisFindings(**{**base, **extra})


def _build_tool_definition() -> dict:
    return {
        "name": "record_denial_analysis",
        "description": "Record the structured root cause analysis for a denied healthcare claim.",
        "input_schema": {
            "type": "object",
            "properties": {
                "denial_root_cause": {
                    "type": "string",
                    "description": "Human-readable explanation of WHY this claim was denied beyond just reading the CARC code. Include specific claim data in your explanation."
                },
                "carc_interpretation": {
                    "type": "string",
                    "description": "What this CARC code means in the context of THIS specific claim, not the generic definition."
                },
                "rarc_interpretation": {
                    "type": "string",
                    "description": "What the RARC remark codes mean in context. Leave empty if no RARC codes."
                },
                "recoverability_verdict": {
                    "type": "string",
                    "enum": ["recoverable", "not_recoverable", "needs_review"],
                    "description": "Whether this denial can be reversed through appeal or correction."
                },
                "confidence_score": {
                    "type": "number",
                    "description": "Your confidence in the verdict (0.0 = very uncertain, 1.0 = certain)."
                },
                "supporting_evidence": {
                    "type": "array",
                    "description": "Specific fields from the claim data that support your analysis.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field_name": {"type": "string"},
                            "field_value": {"type": "string"},
                            "significance": {"type": "string"}
                        },
                        "required": ["field_name", "field_value", "significance"]
                    }
                },
                "recommended_action": {
                    "type": "string",
                    "description": "Specific, actionable next step for the billing team."
                },
                "appeal_strategy": {
                    "type": "string",
                    "description": "How to frame the appeal if this is recoverable. Include what documentation to gather."
                },
                "appeal_deadline_estimate": {
                    "type": "string",
                    "description": "Estimated appeal deadline based on payer type (e.g., '120 days from denial date for Medicare')."
                }
            },
            "required": [
                "denial_root_cause",
                "carc_interpretation",
                "recoverability_verdict",
                "confidence_score",
                "supporting_evidence",
                "recommended_action"
            ]
        }
    }


def _format_claim_for_prompt(claim: JoinedClaim, pre: PreAnalysisFindings) -> str:
    c835 = claim.claim_835
    c837 = claim.claim_837

    return f"""
CLAIM TO ANALYZE: {claim.claim_id}
{"=" * 60}

EDI 835 — REMITTANCE ADVICE (Payer's Response):
  Claim Status: {c835.pc_ClaimStatus} (4 = Denied)
  Payer: {claim.payer_name}
  Insurance Type: {claim.insurance_type}
  Claim Amount: ${c835.pc_ClaimAmount:,.2f}
  Amount Paid: ${c835.pc_ClaimPaid:,.2f}
  Statement Period: {c835.pc_StatementBegin} to {c835.pc_StatementEnd}
  Date Received by Payer: {c835.pc_ReceivedDate}
  Procedure Code: {c835.pcl_ProcedureCode}
  Procedure Modifier(s): {c835.pcl_ProcedureModifier1 or '(none)'}
  Remark Codes (RARC): {c835.pcl_RemarkCodes or '(none)'}
  Adjustment Group: {c835.pcla_AdjustmentGroup} (CO = Contractual Obligation)
  CARC Code: {c835.pcla_AdjustmentReason}
  Adjustment Amount: ${c835.pcla_AdjustmentAmount:,.2f}
  Prior Auth (835): {c835.pc_PriorAuthNum or '(none)'}

EDI 837 — ORIGINAL CLAIM SUBMISSION:
  Claim No: {c837.ec_ClaimNo}
  Payer: {c837.ec_PayerName}
  Insurance Type: {c837.ec_InsuranceType}
  Service Date From: {c837.ec_ServiceDateFrom}
  Service Date To: {c837.ec_ServiceDateTo}
  Place of Service: {c837.ec_PlaceOfService or '(not specified)'}
  Principal Diagnosis: {c837.ec_PrincipalDiagnosis} {('| Secondary: ' + c837.ec_Diag2) if c837.ec_Diag2 else ''}
  Billing Provider NPI: {c837.ec_BillProvNPI or '(missing)'}
  Rendering Provider NPI: {c837.ec_RendProvNPI or '(missing — potential issue)'}
  Prior Authorization: {c837.ec_PriorAuthorization or '(none)'}
  Delay Reason Code: {c837.ec_DelayReasonCode or '(none)'}
  Claim Frequency: {c837.ec_ClaimFrequency} (1=Original, 7=Replacement, 8=Void)
  Type of Bill: {c837.ec_TypeOfBill or '(not specified)'}

RULE-BASED PRE-ANALYSIS FINDINGS:
  CARC Code: {pre.carc_code} — {pre.carc_description}
  Category: {pre.carc_category}
{"  Days since service: " + str(pre.days_since_service) + " days" if pre.days_since_service is not None else "  Days since service: (dates unavailable)"}
{"  Filing limit (" + claim.insurance_type + "): " + str(pre.filing_limit_days) + " days" if pre.filing_limit_days is not None else ""}
{"  Is genuinely late: " + str(pre.is_genuinely_late) if pre.is_genuinely_late is not None else ""}
{"  Has delay reason code: " + str(pre.has_delay_reason_code) if pre.has_delay_reason_code is not None else ""}
{"  Delay reason: " + str(pre.delay_reason_description) if pre.delay_reason_description else ""}
{"  Prior auth on 837: " + str(pre.prior_auth_on_claim) if pre.prior_auth_on_claim is not None else ""}
{"  Auth number: " + str(pre.auth_number) if pre.auth_number else ""}
{"  LCD referenced: " + str(pre.lcd_referenced) if pre.lcd_referenced is not None else ""}

Pre-Analysis Notes:
{chr(10).join("  • " + note for note in pre.notes)}
""".strip()


class RootCauseAnalyzer:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._carc_codes = _load_carc_codes()
        self._system_prompt = _SYSTEM_PROMPT
        self._tool = _build_tool_definition()

    def analyze(self, claim: JoinedClaim) -> RootCauseAnalysis:
        pre = _run_pre_analysis(claim, self._carc_codes)
        claim_text = _format_claim_for_prompt(claim, pre)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Analyze this denied claim and call the record_denial_analysis tool "
                        f"with your structured findings.\n\n{claim_text}"
                    ),
                }
            ],
            tools=[self._tool],
            tool_choice={"type": "tool", "name": "record_denial_analysis"},
            betas=["prompt-caching-2024-07-31"],
        )

        tool_result = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if not tool_result:
            raise RuntimeError(f"LLM did not return tool use for claim {claim.claim_id}")

        result = tool_result.input

        remark_raw = (claim.claim_835.pcl_RemarkCodes or "").strip()
        rarc_codes = [r.strip() for r in remark_raw.replace(",", " ").split() if r.strip()]

        appeal_deadline = result.get("appeal_deadline_estimate")
        if not appeal_deadline:
            appeal_windows = self._carc_codes["appeal_windows_days"]
            window = appeal_windows.get(claim.insurance_type, appeal_windows["default"])
            appeal_deadline = f"{window} days from denial date (estimate for {claim.insurance_type})"

        return RootCauseAnalysis(
            claim_id=claim.claim_id,
            denial_root_cause=result["denial_root_cause"],
            carc_code=claim.carc_code,
            carc_interpretation=result["carc_interpretation"],
            rarc_codes=rarc_codes,
            rarc_interpretation=result.get("rarc_interpretation"),
            recoverability_verdict=RecoverabilityVerdict(result["recoverability_verdict"]),
            confidence_score=float(result["confidence_score"]),
            supporting_evidence=[
                SupportingEvidence(**e) for e in result.get("supporting_evidence", [])
            ],
            recommended_action=result["recommended_action"],
            appeal_strategy=result.get("appeal_strategy"),
            appeal_deadline_estimate=appeal_deadline,
            pre_analysis=pre,
        )

    def analyze_batch(self, claims: list[JoinedClaim]) -> list[RootCauseAnalysis]:
        results = []
        for claim in claims:
            if not claim.is_denied:
                continue
            try:
                results.append(self.analyze(claim))
            except Exception as e:
                print(f"[ERROR] Failed to analyze {claim.claim_id}: {e}")
        return results
