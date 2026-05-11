"""Generates a synthetic dataset of 30 realistic healthcare claims (mix of paid and denied)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _make_claim(
    claim_id: str,
    label: str,
    # 835 fields
    claim_status: str,
    claim_amount: float,
    claim_paid: float,
    insurance_type: str,
    payer_name: str,
    payer_id: str,
    received_date: str,
    statement_begin: str,
    statement_end: str,
    procedure_code: str,
    adjustment_group: str,
    adjustment_reason: str,
    adjustment_amount: float,
    remark_codes: str,
    # 837 fields
    service_date_from: str,
    service_date_to: str,
    principal_diagnosis: str,
    bill_prov_npi: str,
    rend_prov_npi: str,
    prior_authorization: str,
    delay_reason_code: str,
    claim_frequency: str,
    subscriber_id: str,
    place_of_service: str,
    type_of_bill: str,
    diag2: str = "",
    patient_last: str = "",
    patient_first: str = "",
    modifier1: str = "",
    paid_amount: float = 0.0,
    allowed_amount: float = 0.0,
    patient_responsibility: float = 0.0,
) -> dict:
    return {
        "claim_id": claim_id,
        "label": label,
        "claim_835": {
            "pc_ClaimID": claim_id,
            "pc_ClaimStatus": claim_status,
            "pc_ClaimAmount": claim_amount,
            "pc_ClaimPaid": claim_paid,
            "pc_PatientResponsibility": patient_responsibility,
            "pc_InsuranceType": insurance_type,
            "cp_PayerName": payer_name,
            "cp_PayerID": payer_id,
            "pc_ReceivedDate": received_date,
            "pc_StatementBegin": statement_begin,
            "pc_StatementEnd": statement_end,
            "pc_PatientLast": patient_last,
            "pc_PatientFirst": patient_first,
            "pcl_ProcedureCode": procedure_code,
            "pcl_ProcedureModifier1": modifier1,
            "pcl_ChargedAmount": claim_amount,
            "pcl_PaidAmount": paid_amount,
            "pcl_AllowedAmount": allowed_amount,
            "pcl_RemarkCodes": remark_codes,
            "pcla_AdjustmentGroup": adjustment_group,
            "pcla_AdjustmentReason": adjustment_reason,
            "pcla_AdjustmentAmount": adjustment_amount,
        },
        "claim_837": {
            "ec_ClaimNo": claim_id,
            "ec_Amount": claim_amount,
            "ec_PayerName": payer_name,
            "ec_PayerID": payer_id,
            "ec_InsuranceType": insurance_type,
            "ec_ServiceDateFrom": service_date_from,
            "ec_ServiceDateTo": service_date_to,
            "ec_PrincipalDiagnosis": principal_diagnosis,
            "ec_Diag2": diag2,
            "ec_BillProvNPI": bill_prov_npi,
            "ec_RendProvNPI": rend_prov_npi,
            "ec_PriorAuthorization": prior_authorization,
            "ec_DelayReasonCode": delay_reason_code,
            "ec_ClaimFrequency": claim_frequency,
            "ec_SubscriberID": subscriber_id,
            "ec_PlaceOfService": place_of_service,
            "ec_TypeOfBill": type_of_bill,
        },
    }


def generate_synthetic_dataset() -> list[dict]:
    """Generate 30 synthetic claims: 10 paid + 20 denied across realistic scenarios."""

    claims = []

    # ─── PAID CLAIMS (10) ────────────────────────────────────────────────────
    # status "1" = Processed as Primary; adjustment_reason "45" = contractual write-off

    claims.append(_make_claim(
        claim_id="CLM-SYN-0001",
        label="PAID: Medicare office visit (routine)",
        claim_status="1", claim_amount=180.00, claim_paid=120.00,
        insurance_type="Medicare", payer_name="Medicare Part B", payer_id="MEDICARE",
        received_date="2026-01-15", statement_begin="2026-01-10", statement_end="2026-01-10",
        procedure_code="99213", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=60.00, remark_codes="",
        service_date_from="2026-01-10", service_date_to="2026-01-10",
        principal_diagnosis="J06.9", bill_prov_npi="1112223330",
        rend_prov_npi="1112223330", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="1EG4-TE5-MK72", place_of_service="11",
        type_of_bill="", paid_amount=120.00, allowed_amount=142.00, patient_responsibility=22.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0002",
        label="PAID: BCBS knee replacement (with prior auth)",
        claim_status="1", claim_amount=22000.00, claim_paid=8800.00,
        insurance_type="Commercial", payer_name="Blue Cross Blue Shield", payer_id="BCBS-IL",
        received_date="2026-01-25", statement_begin="2026-01-20", statement_end="2026-01-20",
        procedure_code="27447", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=13200.00, remark_codes="",
        service_date_from="2026-01-20", service_date_to="2026-01-20",
        principal_diagnosis="M17.11", bill_prov_npi="2223334440",
        rend_prov_npi="2223334440", prior_authorization="AUTH-112233", delay_reason_code="",
        claim_frequency="1", subscriber_id="BCBS789012", place_of_service="21",
        type_of_bill="131", paid_amount=8800.00, allowed_amount=9500.00, patient_responsibility=700.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0003",
        label="PAID: Aetna mental health therapy",
        claim_status="1", claim_amount=200.00, claim_paid=130.00,
        insurance_type="Commercial", payer_name="Aetna", payer_id="AETNA-01",
        received_date="2026-02-01", statement_begin="2026-01-28", statement_end="2026-01-28",
        procedure_code="90837", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=55.00, remark_codes="",
        service_date_from="2026-01-28", service_date_to="2026-01-28",
        principal_diagnosis="F32.1", bill_prov_npi="3334445550",
        rend_prov_npi="3334445550", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="AET456789", place_of_service="11",
        type_of_bill="", paid_amount=130.00, allowed_amount=145.00, patient_responsibility=15.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0004",
        label="PAID: UHC EKG cardiology",
        claim_status="1", claim_amount=150.00, claim_paid=95.00,
        insurance_type="Commercial", payer_name="United Healthcare", payer_id="UHC-01",
        received_date="2026-02-08", statement_begin="2026-02-03", statement_end="2026-02-03",
        procedure_code="93000", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=45.00, remark_codes="",
        service_date_from="2026-02-03", service_date_to="2026-02-03",
        principal_diagnosis="I10", bill_prov_npi="4445556660",
        rend_prov_npi="4445556660", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="UHC123456", place_of_service="11",
        type_of_bill="", paid_amount=95.00, allowed_amount=105.00, patient_responsibility=10.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0005",
        label="PAID: Cigna chest X-ray (pneumonia)",
        claim_status="1", claim_amount=450.00, claim_paid=280.00,
        insurance_type="Commercial", payer_name="Cigna", payer_id="CIGNA-01",
        received_date="2026-02-12", statement_begin="2026-02-07", statement_end="2026-02-07",
        procedure_code="71046", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=120.00, remark_codes="",
        service_date_from="2026-02-07", service_date_to="2026-02-07",
        principal_diagnosis="J18.9", bill_prov_npi="5556667770",
        rend_prov_npi="5556667770", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="CIG789012", place_of_service="22",
        type_of_bill="", paid_amount=280.00, allowed_amount=330.00, patient_responsibility=50.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0006",
        label="PAID: Medicare inpatient follow-up visit (diabetes)",
        claim_status="1", claim_amount=280.00, claim_paid=180.00,
        insurance_type="Medicare", payer_name="Medicare Part B", payer_id="MEDICARE",
        received_date="2026-02-20", statement_begin="2026-02-14", statement_end="2026-02-14",
        procedure_code="99232", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=100.00, remark_codes="",
        service_date_from="2026-02-14", service_date_to="2026-02-14",
        principal_diagnosis="E11.9", bill_prov_npi="1112223330",
        rend_prov_npi="1112223330", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="1EG4-TE5-AB11", place_of_service="21",
        type_of_bill="", paid_amount=180.00, allowed_amount=195.00, patient_responsibility=15.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0007",
        label="PAID: Medicaid preventive new patient visit",
        claim_status="1", claim_amount=160.00, claim_paid=95.00,
        insurance_type="Medicaid", payer_name="State Medicaid", payer_id="MEDICAID-IL",
        received_date="2026-03-01", statement_begin="2026-02-25", statement_end="2026-02-25",
        procedure_code="99203", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=65.00, remark_codes="",
        service_date_from="2026-02-25", service_date_to="2026-02-25",
        principal_diagnosis="Z23", bill_prov_npi="6667778880",
        rend_prov_npi="6667778880", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="MCDIL567890", place_of_service="11",
        type_of_bill="", paid_amount=95.00, allowed_amount=95.00, patient_responsibility=0.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0008",
        label="PAID: BCBS endoscopy (GERD)",
        claim_status="1", claim_amount=2800.00, claim_paid=1200.00,
        insurance_type="Commercial", payer_name="Blue Cross Blue Shield", payer_id="BCBS-IL",
        received_date="2026-03-10", statement_begin="2026-03-05", statement_end="2026-03-05",
        procedure_code="43239", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=1400.00, remark_codes="",
        service_date_from="2026-03-05", service_date_to="2026-03-05",
        principal_diagnosis="K21.0", bill_prov_npi="2223334440",
        rend_prov_npi="2223334440", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="BCBS890123", place_of_service="22",
        type_of_bill="", paid_amount=1200.00, allowed_amount=1350.00, patient_responsibility=150.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0009",
        label="PAID: Humana shoulder arthroscopy (with prior auth)",
        claim_status="1", claim_amount=4200.00, claim_paid=1800.00,
        insurance_type="Commercial", payer_name="Humana", payer_id="HUMANA-01",
        received_date="2026-03-18", statement_begin="2026-03-12", statement_end="2026-03-12",
        procedure_code="29827", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=2200.00, remark_codes="",
        service_date_from="2026-03-12", service_date_to="2026-03-12",
        principal_diagnosis="M75.1", bill_prov_npi="7778889990",
        rend_prov_npi="7778889990", prior_authorization="AUTH-334455", delay_reason_code="",
        claim_frequency="1", subscriber_id="HUM456789", place_of_service="21",
        type_of_bill="131", paid_amount=1800.00, allowed_amount=2000.00, patient_responsibility=200.00,
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0010",
        label="PAID: Aetna complex office visit (hypertension)",
        claim_status="1", claim_amount=420.00, claim_paid=250.00,
        insurance_type="Commercial", payer_name="Aetna", payer_id="AETNA-01",
        received_date="2026-03-22", statement_begin="2026-03-18", statement_end="2026-03-18",
        procedure_code="99215", adjustment_group="CO", adjustment_reason="45",
        adjustment_amount=135.00, remark_codes="",
        service_date_from="2026-03-18", service_date_to="2026-03-18",
        principal_diagnosis="I10", diag2="E11.9", bill_prov_npi="3334445550",
        rend_prov_npi="3334445550", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="AET012345", place_of_service="11",
        type_of_bill="", paid_amount=250.00, allowed_amount=275.00, patient_responsibility=25.00,
    ))

    # ─── DENIED: CARC 29 — Timely Filing (4) ─────────────────────────────────

    claims.append(_make_claim(
        claim_id="CLM-SYN-0011",
        label="DENIED CARC29: BCBS — genuinely late (278 days, limit 180)",
        claim_status="4", claim_amount=4500.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Blue Cross Blue Shield", payer_id="BCBS-IL",
        received_date="2026-03-20", statement_begin="2025-06-15", statement_end="2025-06-15",
        procedure_code="99214", adjustment_group="CO", adjustment_reason="29",
        adjustment_amount=4500.00, remark_codes="",
        service_date_from="2025-06-15", service_date_to="2025-06-15",
        principal_diagnosis="J06.9", bill_prov_npi="1234567890",
        rend_prov_npi="1234567890", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="XYZ123456", place_of_service="11", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0012",
        label="DENIED CARC29: Aetna — borderline (95 days, limit 90 per contract)",
        claim_status="4", claim_amount=320.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Aetna", payer_id="AETNA-01",
        received_date="2026-04-15", statement_begin="2026-01-09", statement_end="2026-01-09",
        procedure_code="93000", adjustment_group="CO", adjustment_reason="29",
        adjustment_amount=320.00, remark_codes="",
        service_date_from="2026-01-09", service_date_to="2026-01-09",
        principal_diagnosis="I25.10", bill_prov_npi="4445556660",
        rend_prov_npi="4445556660", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="AET234567", place_of_service="11", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0013",
        label="DENIED CARC29: Medicare — filed 380 days late (limit 365)",
        claim_status="4", claim_amount=22000.00, claim_paid=0.00,
        insurance_type="Medicare", payer_name="Medicare Part B", payer_id="MEDICARE",
        received_date="2026-02-10", statement_begin="2025-02-05", statement_end="2025-02-05",
        procedure_code="27447", adjustment_group="CO", adjustment_reason="29",
        adjustment_amount=22000.00, remark_codes="",
        service_date_from="2025-02-05", service_date_to="2025-02-05",
        principal_diagnosis="M17.11", bill_prov_npi="2223334440",
        rend_prov_npi="2223334440", prior_authorization="AUTH-555666", delay_reason_code="",
        claim_frequency="1", subscriber_id="1EG4-TE5-CD22", place_of_service="21", type_of_bill="131",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0014",
        label="DENIED CARC29: UHC — 170 days WITH delay reason code (potentially recoverable)",
        claim_status="4", claim_amount=8200.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="United Healthcare", payer_id="UHC-01",
        received_date="2026-04-25", statement_begin="2025-11-05", statement_end="2025-11-05",
        procedure_code="72148", adjustment_group="CO", adjustment_reason="29",
        adjustment_amount=8200.00, remark_codes="",
        service_date_from="2025-11-05", service_date_to="2025-11-05",
        principal_diagnosis="M54.5", bill_prov_npi="5678901234",
        rend_prov_npi="5678901234", prior_authorization="", delay_reason_code="9",
        claim_frequency="1", subscriber_id="UHC567890", place_of_service="11", type_of_bill="",
    ))

    # ─── DENIED: CARC 16 — Missing Information (4) ────────────────────────────

    claims.append(_make_claim(
        claim_id="CLM-SYN-0015",
        label="DENIED CARC16: Medicare — missing modifier on knee replacement",
        claim_status="4", claim_amount=12800.00, claim_paid=0.00,
        insurance_type="Medicare", payer_name="Medicare Part B", payer_id="MEDICARE",
        received_date="2026-02-10", statement_begin="2026-01-08", statement_end="2026-01-08",
        procedure_code="27447", adjustment_group="CO", adjustment_reason="16",
        adjustment_amount=12800.00, remark_codes="N20",
        service_date_from="2026-01-08", service_date_to="2026-01-08",
        principal_diagnosis="M17.11", bill_prov_npi="9876543210",
        rend_prov_npi="9876543210", prior_authorization="AUTH-998877", delay_reason_code="",
        claim_frequency="1", subscriber_id="1EG4-TE5-MK72", place_of_service="21",
        type_of_bill="131",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0016",
        label="DENIED CARC16: BCBS — missing HCPCS code on X-ray",
        claim_status="4", claim_amount=450.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Blue Cross Blue Shield", payer_id="BCBS-IL",
        received_date="2026-03-05", statement_begin="2026-02-28", statement_end="2026-02-28",
        procedure_code="71046", adjustment_group="CO", adjustment_reason="16",
        adjustment_amount=450.00, remark_codes="N20",
        service_date_from="2026-02-28", service_date_to="2026-02-28",
        principal_diagnosis="J18.9", bill_prov_npi="5556667770",
        rend_prov_npi="5556667770", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="BCBS901234", place_of_service="22", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0017",
        label="DENIED CARC16: Aetna — missing rendering provider NPI",
        claim_status="4", claim_amount=200.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Aetna", payer_id="AETNA-01",
        received_date="2026-03-12", statement_begin="2026-03-07", statement_end="2026-03-07",
        procedure_code="90837", adjustment_group="CO", adjustment_reason="16",
        adjustment_amount=200.00, remark_codes="N56",
        service_date_from="2026-03-07", service_date_to="2026-03-07",
        principal_diagnosis="F32.1", bill_prov_npi="3334445550",
        rend_prov_npi="", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="AET678901", place_of_service="11", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0018",
        label="DENIED CARC16: UHC — missing operative report attachment",
        claim_status="4", claim_amount=2800.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="United Healthcare", payer_id="UHC-01",
        received_date="2026-03-20", statement_begin="2026-03-14", statement_end="2026-03-14",
        procedure_code="43239", adjustment_group="CO", adjustment_reason="16",
        adjustment_amount=2800.00, remark_codes="MA01",
        service_date_from="2026-03-14", service_date_to="2026-03-14",
        principal_diagnosis="K21.0", bill_prov_npi="4445556660",
        rend_prov_npi="4445556660", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="UHC890123", place_of_service="22", type_of_bill="",
    ))

    # ─── DENIED: CARC 50 — Medical Necessity (4) ──────────────────────────────

    claims.append(_make_claim(
        claim_id="CLM-SYN-0019",
        label="DENIED CARC50: Aetna — MRI lumbar without prior conservative treatment",
        claim_status="4", claim_amount=8200.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Aetna", payer_id="AETNA-01",
        received_date="2026-03-05", statement_begin="2026-02-20", statement_end="2026-02-20",
        procedure_code="72148", adjustment_group="CO", adjustment_reason="50",
        adjustment_amount=8200.00, remark_codes="N386",
        service_date_from="2026-02-20", service_date_to="2026-02-20",
        principal_diagnosis="M54.5", diag2="M51.16", bill_prov_npi="5678901234",
        rend_prov_npi="5678901234", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="AET789012", place_of_service="11", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0020",
        label="DENIED CARC50: BCBS — CT chest without documented indication",
        claim_status="4", claim_amount=3200.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Blue Cross Blue Shield", payer_id="BCBS-IL",
        received_date="2026-04-01", statement_begin="2026-03-25", statement_end="2026-03-25",
        procedure_code="71250", adjustment_group="CO", adjustment_reason="50",
        adjustment_amount=3200.00, remark_codes="N386",
        service_date_from="2026-03-25", service_date_to="2026-03-25",
        principal_diagnosis="C34.10", bill_prov_npi="5556667770",
        rend_prov_npi="5556667770", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="BCBS234567", place_of_service="22", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0021",
        label="DENIED CARC50: Cigna — inpatient admission not medically necessary",
        claim_status="4", claim_amount=18500.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Cigna", payer_id="CIGNA-01",
        received_date="2026-04-08", statement_begin="2026-04-01", statement_end="2026-04-03",
        procedure_code="99233", adjustment_group="CO", adjustment_reason="50",
        adjustment_amount=18500.00, remark_codes="MA01",
        service_date_from="2026-04-01", service_date_to="2026-04-03",
        principal_diagnosis="F32.1", bill_prov_npi="3334445550",
        rend_prov_npi="3334445550", prior_authorization="AUTH-667788", delay_reason_code="",
        claim_frequency="1", subscriber_id="CIG012345", place_of_service="21", type_of_bill="131",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0022",
        label="DENIED CARC50: Medicare — joint injection non-covered (exceeded frequency)",
        claim_status="4", claim_amount=350.00, claim_paid=0.00,
        insurance_type="Medicare", payer_name="Medicare Part B", payer_id="MEDICARE",
        received_date="2026-04-15", statement_begin="2026-04-08", statement_end="2026-04-08",
        procedure_code="20610", adjustment_group="CO", adjustment_reason="50",
        adjustment_amount=350.00, remark_codes="N115",
        service_date_from="2026-04-08", service_date_to="2026-04-08",
        principal_diagnosis="M17.11", bill_prov_npi="9876543210",
        rend_prov_npi="9876543210", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="1EG4-TE5-EF33", place_of_service="11", type_of_bill="",
    ))

    # ─── DENIED: CARC 197 — Prior Auth Missing (3) ────────────────────────────

    claims.append(_make_claim(
        claim_id="CLM-SYN-0023",
        label="DENIED CARC197: Aetna — knee replacement without prior auth (elective surgery)",
        claim_status="4", claim_amount=22000.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Aetna", payer_id="AETNA-01",
        received_date="2026-02-28", statement_begin="2026-02-22", statement_end="2026-02-22",
        procedure_code="27447", adjustment_group="CO", adjustment_reason="197",
        adjustment_amount=22000.00, remark_codes="",
        service_date_from="2026-02-22", service_date_to="2026-02-22",
        principal_diagnosis="M17.11", bill_prov_npi="2223334440",
        rend_prov_npi="2223334440", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="AET345678", place_of_service="21", type_of_bill="131",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0024",
        label="DENIED CARC197: UHC — MRI without prior auth (auth number on claim, payer says absent)",
        claim_status="4", claim_amount=8200.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="United Healthcare", payer_id="UHC-01",
        received_date="2026-03-15", statement_begin="2026-03-08", statement_end="2026-03-08",
        procedure_code="72148", adjustment_group="CO", adjustment_reason="197",
        adjustment_amount=8200.00, remark_codes="",
        service_date_from="2026-03-08", service_date_to="2026-03-08",
        principal_diagnosis="M54.5", bill_prov_npi="5678901234",
        rend_prov_npi="5678901234", prior_authorization="AUTH-999001", delay_reason_code="",
        claim_frequency="1", subscriber_id="UHC234567", place_of_service="11", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0025",
        label="DENIED CARC197: BCBS — therapy exceeded authorized 12 sessions",
        claim_status="4", claim_amount=200.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Blue Cross Blue Shield", payer_id="BCBS-IL",
        received_date="2026-04-05", statement_begin="2026-03-28", statement_end="2026-03-28",
        procedure_code="90837", adjustment_group="CO", adjustment_reason="197",
        adjustment_amount=200.00, remark_codes="",
        service_date_from="2026-03-28", service_date_to="2026-03-28",
        principal_diagnosis="F32.1", bill_prov_npi="3334445550",
        rend_prov_npi="3334445550", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="BCBS678901", place_of_service="11", type_of_bill="",
    ))

    # ─── DENIED: CARC 18 — Duplicate (2) ─────────────────────────────────────

    claims.append(_make_claim(
        claim_id="CLM-SYN-0026",
        label="DENIED CARC18: UHC — true duplicate (same claim submitted twice)",
        claim_status="4", claim_amount=3200.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="United Healthcare", payer_id="UHC-01",
        received_date="2026-02-15", statement_begin="2026-01-10", statement_end="2026-01-10",
        procedure_code="99213", adjustment_group="CO", adjustment_reason="18",
        adjustment_amount=3200.00, remark_codes="M86",
        service_date_from="2026-01-10", service_date_to="2026-01-10",
        principal_diagnosis="J20.9", bill_prov_npi="1234567890",
        rend_prov_npi="1234567890", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="UHC456789", place_of_service="11", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0027",
        label="DENIED CARC18: Medicare — flagged as duplicate, different DOS (needs review)",
        claim_status="4", claim_amount=280.00, claim_paid=0.00,
        insurance_type="Medicare", payer_name="Medicare Part B", payer_id="MEDICARE",
        received_date="2026-03-01", statement_begin="2026-02-14", statement_end="2026-02-14",
        procedure_code="99232", adjustment_group="CO", adjustment_reason="18",
        adjustment_amount=280.00, remark_codes="",
        service_date_from="2026-02-14", service_date_to="2026-02-14",
        principal_diagnosis="E11.9", bill_prov_npi="1112223330",
        rend_prov_npi="1112223330", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="1EG4-TE5-GH44", place_of_service="21", type_of_bill="",
    ))

    # ─── DENIED: OTHER CARC CODES (3) ─────────────────────────────────────────

    claims.append(_make_claim(
        claim_id="CLM-SYN-0028",
        label="DENIED CARC4: Aetna — modifier 25 inconsistency with E&M code",
        claim_status="4", claim_amount=420.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Aetna", payer_id="AETNA-01",
        received_date="2026-04-10", statement_begin="2026-04-03", statement_end="2026-04-03",
        procedure_code="99214", adjustment_group="CO", adjustment_reason="4",
        adjustment_amount=420.00, remark_codes="N56", modifier1="25",
        service_date_from="2026-04-03", service_date_to="2026-04-03",
        principal_diagnosis="I10", bill_prov_npi="3334445550",
        rend_prov_npi="3334445550", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="AET901234", place_of_service="11", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0029",
        label="DENIED CARC97: BCBS — minor visit bundled with same-day comprehensive visit",
        claim_status="4", claim_amount=75.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Blue Cross Blue Shield", payer_id="BCBS-IL",
        received_date="2026-04-18", statement_begin="2026-04-10", statement_end="2026-04-10",
        procedure_code="99211", adjustment_group="CO", adjustment_reason="97",
        adjustment_amount=75.00, remark_codes="",
        service_date_from="2026-04-10", service_date_to="2026-04-10",
        principal_diagnosis="I10", bill_prov_npi="1234567890",
        rend_prov_npi="1234567890", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="BCBS345678", place_of_service="11", type_of_bill="",
    ))

    claims.append(_make_claim(
        claim_id="CLM-SYN-0030",
        label="DENIED CARC96: Cigna — non-covered DME service excluded from plan",
        claim_status="4", claim_amount=1200.00, claim_paid=0.00,
        insurance_type="Commercial", payer_name="Cigna", payer_id="CIGNA-01",
        received_date="2026-04-25", statement_begin="2026-04-18", statement_end="2026-04-18",
        procedure_code="S9088", adjustment_group="CO", adjustment_reason="96",
        adjustment_amount=1200.00, remark_codes="",
        service_date_from="2026-04-18", service_date_to="2026-04-18",
        principal_diagnosis="M17.11", bill_prov_npi="7778889990",
        rend_prov_npi="7778889990", prior_authorization="", delay_reason_code="",
        claim_frequency="1", subscriber_id="CIG567890", place_of_service="12", type_of_bill="",
    ))

    return claims


def save_synthetic_dataset(output_path: str) -> list[dict]:
    """Generate and save the synthetic dataset to a JSON file."""
    claims = generate_synthetic_dataset()
    data = {"description": "Synthetic dataset: 30 claims (10 paid + 20 denied)", "claims": claims}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return claims
