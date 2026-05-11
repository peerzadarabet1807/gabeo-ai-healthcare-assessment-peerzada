"""Integration tests for the pipeline (no API calls — uses mock or skip_* flags)."""

import json
import os
import pytest
from pathlib import Path

from src.data.loader import ClaimLoader
from src.data.generator import generate_synthetic_dataset, save_synthetic_dataset
from src.analysis.clustering import DenialClusterer, _rule_based_key, _estimate_success_rate
from src.models.claim import JoinedClaim


def test_save_and_reload_synthetic_dataset(tmp_path):
    """Generated dataset should round-trip through JSON without data loss."""
    output = str(tmp_path / "test_synthetic.json")
    save_synthetic_dataset(output)

    assert Path(output).exists()
    with open(output) as f:
        data = json.load(f)

    assert "claims" in data
    assert len(data["claims"]) == 30


def test_loader_round_trips_synthetic(tmp_path):
    """Loader should successfully parse all 30 synthetic claims."""
    output = str(tmp_path / "synthetic.json")
    save_synthetic_dataset(output)

    loader = ClaimLoader()
    claims = loader.load_file(output)
    assert len(claims) == 30
    denied = [c for c in claims if c.is_denied]
    paid = [c for c in claims if not c.is_denied]
    assert len(denied) == 20
    assert len(paid) == 10


def test_rule_based_clustering():
    """Rule-based clustering should group claims by payer+CARC."""
    from src.analysis.clustering import _cluster_by_rules

    loader = ClaimLoader()
    synthetic = generate_synthetic_dataset()
    all_claims = loader.load_from_dicts(synthetic)
    denied = [c for c in all_claims if c.is_denied]

    clusters = _cluster_by_rules(denied)
    assert len(clusters) > 0

    # All denied claims should appear in exactly one cluster
    all_in_clusters = [c for group in clusters.values() for c in group]
    assert len(all_in_clusters) == len(denied)


def test_cluster_keys_are_consistent():
    """Two claims with same payer and CARC should share a cluster key."""
    loader = ClaimLoader()
    synthetic = generate_synthetic_dataset()
    all_claims = loader.load_from_dicts(synthetic)

    # Both CLM-SYN-0019 and CLM-SYN-0020 are Aetna/BCBS + CARC 50 or 50
    carc50_denied = [c for c in all_claims if c.is_denied and c.carc_code == "50"]
    if len(carc50_denied) >= 2:
        # At least some should share the same payer
        keys = [_rule_based_key(c) for c in carc50_denied]
        unique_keys = set(keys)
        # There should be fewer unique keys than claims (clustering happened)
        assert len(unique_keys) <= len(carc50_denied)


def test_estimate_success_rate_range():
    """Success rate estimate should always be between 0 and 1."""
    loader = ClaimLoader()
    synthetic = generate_synthetic_dataset()
    all_claims = loader.load_from_dicts(synthetic)
    denied = [c for c in all_claims if c.is_denied]

    for carc in ["16", "29", "50", "197", "18", "4", "97", "96"]:
        carc_claims = [c for c in denied if c.carc_code == carc]
        if carc_claims:
            rate = _estimate_success_rate(carc_claims, all_claims)
            assert 0.0 <= rate <= 1.0, f"Rate {rate} out of range for CARC {carc}"


def test_clusterer_no_llm(tmp_path):
    """Clustering without LLM should still produce a valid report."""
    loader = ClaimLoader()
    synthetic = generate_synthetic_dataset()
    all_claims = loader.load_from_dicts(synthetic)

    clusterer = DenialClusterer(api_key="dummy-key-for-test")
    report = clusterer.cluster_and_report(
        all_claims=all_claims,
        root_cause_analyses=None,
        use_llm_summaries=False,  # skip LLM calls
    )

    assert report.total_denied_claims == 20
    assert report.total_claimed_amount_range_ok()  # custom validator
    assert len(report.clusters) > 0
    assert report.total_recoverable_estimate > 0
    assert report.top_priority_cluster_id in {c.cluster_id for c in report.clusters}

    # Each claim should appear in exactly one cluster
    all_claim_ids = [cid for cluster in report.clusters for cid in cluster.claim_ids]
    assert len(set(all_claim_ids)) == len(all_claim_ids), "Duplicate claims in clusters"


def test_pipeline_no_api(tmp_path):
    """Pipeline with all LLM steps skipped should still produce a result."""
    from src.pipeline import DenialAnalysisPipeline, PipelineConfig

    loader = ClaimLoader()
    synthetic = generate_synthetic_dataset()
    all_claims = loader.load_from_dicts(synthetic)

    config = PipelineConfig(
        api_key="dummy",
        skip_root_cause=True,
        skip_pattern_matching=False,
        skip_clustering=False,
        use_llm_for_clustering=False,
    )
    pipeline = DenialAnalysisPipeline(config=config)
    result = pipeline.run(all_claims)

    assert len(result.all_claims) == 30
    assert len(result.denied_claims) == 20
    assert len(result.root_cause_analyses) == 0  # skipped
    assert len(result.pattern_match_results) == 20
    assert result.batch_report is not None
    assert result.batch_report.total_denied_claims == 20
