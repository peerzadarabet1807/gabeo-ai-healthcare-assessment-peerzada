"""Problem 3: Denial Clustering & Batch Intelligence using KMeans + LLM summaries."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

import anthropic

from src.models.claim import JoinedClaim
from src.models.analysis import (
    BatchIntelligenceReport,
    DenialCluster,
    RootCauseAnalysis,
)
from src.analysis.pattern_matching import featurize, _normalize_payer

PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "cluster_summary.txt"

_CLUSTER_SYSTEM_PROMPT: Optional[str] = None


def _load_cluster_prompt() -> str:
    global _CLUSTER_SYSTEM_PROMPT
    if _CLUSTER_SYSTEM_PROMPT is None:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            _CLUSTER_SYSTEM_PROMPT = f.read()
    return _CLUSTER_SYSTEM_PROMPT


# ─── Rule-Based Clustering ────────────────────────────────────────────────────


def _rule_based_key(claim: JoinedClaim) -> str:
    """Create a cluster key from (payer, CARC code) — the most actionable grouping."""
    payer = _normalize_payer(claim.payer_name)
    carc = claim.carc_code or "unknown"
    return f"{payer}||{carc}"


def _cluster_by_rules(denied_claims: list[JoinedClaim]) -> dict[str, list[JoinedClaim]]:
    """Group denied claims by (payer, CARC code) combination."""
    clusters: dict[str, list[JoinedClaim]] = {}
    for claim in denied_claims:
        key = _rule_based_key(claim)
        clusters.setdefault(key, []).append(claim)
    return clusters


# ─── ML-Based Clustering (KMeans) ─────────────────────────────────────────────


def _kmeans_cluster(
    denied_claims: list[JoinedClaim], n_clusters: int
) -> dict[int, list[JoinedClaim]]:
    """Cluster denied claims using KMeans on feature vectors."""
    if len(denied_claims) <= n_clusters:
        return {i: [c] for i, c in enumerate(denied_claims)}

    vectors = np.array([featurize(c) for c in denied_claims], dtype=np.float32)

    # Simple KMeans implementation to avoid sklearn version issues
    np.random.seed(42)
    idx = np.random.choice(len(vectors), n_clusters, replace=False)
    centroids = vectors[idx].copy()

    for _ in range(50):  # max iterations
        # Assign each claim to nearest centroid
        dists = np.array([
            [np.linalg.norm(v - c) for c in centroids] for v in vectors
        ])
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.array([
            vectors[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    clusters: dict[int, list[JoinedClaim]] = {}
    for i, claim in enumerate(denied_claims):
        k = int(labels[i])
        clusters.setdefault(k, []).append(claim)
    return clusters


def _merge_clusters(
    rule_clusters: dict[str, list[JoinedClaim]],
    ml_clusters: dict[int, list[JoinedClaim]],
    min_cluster_size: int = 2,
) -> dict[str, list[JoinedClaim]]:
    """Prefer rule-based clusters (more actionable) and merge small ones via ML labels."""
    merged: dict[str, list[JoinedClaim]] = {}

    # Keep rule-based clusters that have enough claims
    claim_to_key: dict[str, str] = {}
    for key, claims in rule_clusters.items():
        if len(claims) >= min_cluster_size:
            merged[key] = claims
            for c in claims:
                claim_to_key[c.claim_id] = key

    # For singletons, absorb into nearest ML cluster
    singleton_claims = [
        c for c in (
            claim
            for claims in rule_clusters.values()
            for claim in claims
            if claim.claim_id not in claim_to_key
        )
    ]
    if singleton_claims:
        for i, ml_group in ml_clusters.items():
            singleton_ids = {c.claim_id for c in singleton_claims}
            ml_singletons = [c for c in ml_group if c.claim_id in singleton_ids]
            if ml_singletons:
                key = f"ML_CLUSTER_{i}"
                merged[key] = ml_singletons

    return merged if merged else {k: v for k, v in rule_clusters.items()}


# ─── Cluster Metrics ──────────────────────────────────────────────────────────


def _estimate_success_rate(cluster_claims: list[JoinedClaim], all_historical: list[JoinedClaim]) -> float:
    """Estimate historical appeal success rate based on similar paid claims in dataset."""
    carc = cluster_claims[0].carc_code if cluster_claims else ""
    payer = _normalize_payer(cluster_claims[0].payer_name) if cluster_claims else ""

    # Find historical claims with same payer + same CARC that were eventually paid
    # In a real system this would use appeal outcome data; here we use paid claims
    # with same payer as a proxy for "what this payer is willing to pay"
    same_payer_paid = sum(
        1 for c in all_historical
        if not c.is_denied and _normalize_payer(c.payer_name) == payer
    )
    same_payer_total = sum(
        1 for c in all_historical
        if _normalize_payer(c.payer_name) == payer
    )

    if same_payer_total == 0:
        base_rate = 0.5
    else:
        base_rate = same_payer_paid / same_payer_total

    # Adjust by CARC code recoverability
    carc_adjustments = {
        "16": 0.20,   # Missing info — almost always recoverable
        "4": 0.15,    # Coding error — easily corrected
        "197": 0.10,  # Prior auth — often retroactively approvable
        "50": 0.00,   # Medical necessity — neutral (documentation dependent)
        "29": -0.15,  # Timely filing — harder to recover
        "18": -0.10,  # Duplicate — usually not recoverable
        "97": -0.20,  # Bundled — usually not recoverable
        "96": -0.25,  # Non-covered — rarely recoverable
    }
    adjustment = carc_adjustments.get(carc, 0.0)
    return max(0.05, min(0.95, base_rate + adjustment))


def _compute_priority_score(cluster: DenialCluster) -> float:
    """Priority = recoverable amount × confidence in success, normalized."""
    return cluster.recoverable_amount_estimate * cluster.historical_appeal_success_rate


def _label_cluster(claims: list[JoinedClaim]) -> str:
    """Generate a human-readable cluster label."""
    if not claims:
        return "Unknown Cluster"
    payer = _normalize_payer(claims[0].payer_name)
    carc = claims[0].carc_code

    carc_labels = {
        "16": "Missing Information",
        "29": "Timely Filing",
        "50": "Medical Necessity",
        "197": "Missing Prior Auth",
        "18": "Duplicate",
        "4": "Modifier Coding Error",
        "97": "Bundled Service",
        "96": "Non-Covered Service",
    }
    carc_label = carc_labels.get(carc, f"CARC {carc}")

    proc_codes = list({c.procedure_code for c in claims if c.procedure_code})
    proc_str = ", ".join(proc_codes[:2])
    if len(proc_codes) > 2:
        proc_str += f" +{len(proc_codes)-2} more"

    return f"{payer} — {carc_label} — CPT {proc_str}"


def _batch_action(carc: str, payer: str, claim_count: int) -> str:
    """Recommend a batch action for the billing team."""
    actions = {
        "16": (
            f"Prepare corrected claim template for {claim_count} claims. "
            "Identify specific missing fields from RARC codes on each claim. "
            "Submit corrected claims within 30 days."
        ),
        "4": (
            f"Review modifier usage for {claim_count} claims against CPT guidelines and payer policy. "
            "Correct modifier and resubmit with addendum explaining medical necessity."
        ),
        "29": (
            f"Pull original submission dates for {claim_count} claims. "
            "Document any delay reason codes. "
            "File formal appeal with timeline justification for borderline cases."
        ),
        "50": (
            f"Request ordering physician to complete {claim_count} Letters of Medical Necessity (LMN). "
            "Gather clinical notes showing conservative treatment and functional limitations. "
            "Submit appeals with LOMS and clinical documentation package."
        ),
        "197": (
            f"Contact {payer} Provider Relations to request retroactive authorization "
            f"for {claim_count} claims. "
            "Submit clinical justification for each. Check if emergency exception applies."
        ),
        "18": (
            f"Pull original claim EOBs for {claim_count} duplicate-flagged claims. "
            "Confirm original payment status. Appeal those where original was not paid "
            "or where service dates differ."
        ),
        "97": (
            f"Review NCCI edits for all {claim_count} bundled claims. "
            "For separately payable services, add appropriate modifiers and resubmit."
        ),
        "96": (
            f"Verify benefit coverage for {claim_count} non-covered claims. "
            "Ensure patient ABN was obtained. Bill patient if ABN was signed."
        ),
    }
    return actions.get(carc, f"Review {claim_count} denied claims and contact {payer} for guidance.")


# ─── LLM Cluster Summary ──────────────────────────────────────────────────────


def _generate_cluster_summary(
    cluster_label: str,
    claims: list[JoinedClaim],
    success_rate: float,
    recoverable_amount: float,
    root_cause_analyses: list[RootCauseAnalysis],
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
) -> str:
    """Use Claude to generate a plain-English billing team summary for this cluster."""
    carc = claims[0].carc_code if claims else "unknown"
    payer = _normalize_payer(claims[0].payer_name) if claims else "Unknown"
    total = sum(c.claimed_amount for c in claims)
    proc_codes = list({c.procedure_code for c in claims if c.procedure_code})

    # Build representative analysis summaries
    analysis_snippets = []
    for rca in root_cause_analyses[:3]:
        analysis_snippets.append(
            f"  • {rca.claim_id}: {rca.denial_root_cause[:120]}... "
            f"[verdict: {rca.recoverability_verdict.value}, confidence: {rca.confidence_score:.0%}]"
        )

    prompt = f"""
Generate a concise billing team summary for this denial cluster.

CLUSTER: {cluster_label}
Claims: {len(claims)}
Payer: {payer}
CARC Code: {carc}
Procedure codes: {', '.join(proc_codes)}
Total denied amount: ${total:,.2f}
Estimated recoverable: ${recoverable_amount:,.2f}
Historical success rate: {success_rate:.0%}

Representative denial analyses:
{chr(10).join(analysis_snippets) if analysis_snippets else "  (No individual analyses available)"}

Write ONE paragraph summary in the format:
"You have [N] claims from [PAYER] denied for [REASON], totaling $[AMOUNT].
Based on historical data, [X]% of similar claims were successfully recovered.
Recommended action: [SPECIFIC ACTION]. Estimated recovery: $[AMOUNT]."

Be specific, action-oriented, and concise. Max 4 sentences.
""".strip()

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
        betas=["prompt-caching-2024-07-31"],
    )
    return response.content[0].text.strip()


# ─── Clusterer ────────────────────────────────────────────────────────────────


class DenialClusterer:
    """Clusters denied claims into actionable groups and generates batch intelligence reports.

    Approach: Rule-based clustering by (payer, CARC code) as the primary grouping,
    with ML KMeans as a fallback for singletons. This gives billing teams the most
    actionable view — they can work ALL claims in a cluster with one batch action.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._system_prompt = _load_cluster_prompt()

    def cluster_and_report(
        self,
        all_claims: list[JoinedClaim],
        root_cause_analyses: Optional[list[RootCauseAnalysis]] = None,
        use_llm_summaries: bool = True,
    ) -> BatchIntelligenceReport:
        """Run full clustering and generate a batch intelligence report."""
        denied = [c for c in all_claims if c.is_denied]
        if not denied:
            raise ValueError("No denied claims to cluster.")

        # Cluster by rules first
        rule_clusters = _cluster_by_rules(denied)

        # KMeans as fallback for small datasets
        n_ml_clusters = max(2, min(5, len(denied) // 3))
        ml_clusters = _kmeans_cluster(denied, n_clusters=n_ml_clusters)

        # Merge
        merged = _merge_clusters(rule_clusters, ml_clusters)

        # Build RCA lookup
        rca_by_id: dict[str, RootCauseAnalysis] = {}
        if root_cause_analyses:
            for rca in root_cause_analyses:
                rca_by_id[rca.claim_id] = rca

        # Build DenialCluster objects
        denial_clusters: list[DenialCluster] = []
        for cluster_idx, (key, claims) in enumerate(merged.items()):
            carc = claims[0].carc_code if claims else "unknown"
            payer = _normalize_payer(claims[0].payer_name) if claims else "Unknown"
            total_amount = sum(c.claimed_amount for c in claims)
            proc_codes = list({c.procedure_code for c in claims if c.procedure_code})
            label = _label_cluster(claims)
            success_rate = _estimate_success_rate(claims, all_claims)
            recoverable = round(total_amount * success_rate, 2)
            cluster_rca = [rca_by_id[c.claim_id] for c in claims if c.claim_id in rca_by_id]
            batch_action = _batch_action(carc, payer, len(claims))

            # LLM summary (can be disabled for cost control)
            if use_llm_summaries and self.client:
                try:
                    summary = _generate_cluster_summary(
                        label, claims, success_rate, recoverable,
                        cluster_rca, self.client, self.model, self._system_prompt,
                    )
                except Exception:
                    summary = (
                        f"You have {len(claims)} claims from {payer} denied for CARC {carc} "
                        f"(total: ${total_amount:,.2f}). Estimated {success_rate:.0%} recovery rate "
                        f"= ${recoverable:,.2f} recoverable. Batch action: {batch_action}"
                    )
            else:
                summary = (
                    f"You have {len(claims)} claims from {payer} denied for CARC {carc} "
                    f"(total: ${total_amount:,.2f}). Estimated {success_rate:.0%} recovery rate "
                    f"= ${recoverable:,.2f} recoverable."
                )

            dc = DenialCluster(
                cluster_id=f"CLUSTER-{cluster_idx+1:02d}",
                cluster_label=label,
                claim_ids=[c.claim_id for c in claims],
                claim_count=len(claims),
                total_denied_amount=round(total_amount, 2),
                primary_carc_code=carc,
                primary_carc_description=_carc_description(carc),
                primary_payer=payer,
                procedure_codes=proc_codes,
                historical_appeal_success_rate=round(success_rate, 2),
                recoverable_amount_estimate=recoverable,
                priority_score=0.0,  # filled below
                recommended_batch_action=batch_action,
                billing_team_summary=summary,
            )
            # Set priority score after cluster is created
            dc.priority_score = round(_compute_priority_score(dc), 2)
            denial_clusters.append(dc)

        # Sort by priority descending
        denial_clusters.sort(key=lambda c: c.priority_score, reverse=True)

        total_denied = sum(c.claimed_amount for c in denied)
        total_recoverable = sum(c.recoverable_amount_estimate for c in denial_clusters)
        top_cluster = denial_clusters[0].cluster_id if denial_clusters else ""

        # Executive summary
        quick_wins = [
            f"Priority 1: {denial_clusters[0].cluster_label} — ${denial_clusters[0].recoverable_amount_estimate:,.0f} recoverable"
            if denial_clusters else "",
        ]
        for dc in denial_clusters[1:4]:
            quick_wins.append(
                f"{dc.cluster_label}: ${dc.recoverable_amount_estimate:,.0f} at {dc.historical_appeal_success_rate:.0%} success rate"
            )
        quick_wins = [w for w in quick_wins if w]

        executive_summary = (
            f"Analysis of {len(denied)} denied claims totaling ${total_denied:,.2f}. "
            f"Across {len(denial_clusters)} clusters, estimated ${total_recoverable:,.2f} "
            f"({total_recoverable/total_denied:.0%}) is potentially recoverable through appeals and corrections. "
            f"Highest priority: {denial_clusters[0].cluster_label if denial_clusters else 'N/A'} "
            f"(${denial_clusters[0].recoverable_amount_estimate:,.0f} recoverable). "
            f"Immediate batch actions recommended for top 3 clusters."
        )

        return BatchIntelligenceReport(
            total_claims_analyzed=len(all_claims),
            total_denied_claims=len(denied),
            total_denied_amount=round(total_denied, 2),
            total_recoverable_estimate=round(total_recoverable, 2),
            clusters=denial_clusters,
            top_priority_cluster_id=top_cluster,
            executive_summary=executive_summary,
            quick_wins=quick_wins,
        )


def _carc_description(carc: str) -> str:
    descriptions = {
        "4": "Modifier inconsistency or missing modifier",
        "16": "Missing or incomplete claim information",
        "18": "Exact duplicate claim",
        "29": "Timely filing deadline expired",
        "45": "Contractual fee schedule adjustment",
        "50": "Service not medically necessary",
        "96": "Non-covered service",
        "97": "Service bundled into another payment",
        "197": "Prior authorization absent",
        "252": "Attachment/documentation required",
    }
    return descriptions.get(carc, f"CARC {carc}")
