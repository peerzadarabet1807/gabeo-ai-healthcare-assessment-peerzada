"""Main orchestration pipeline: load claims → root cause → pattern matching → clustering."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data.loader import ClaimLoader
from src.models.claim import JoinedClaim
from src.models.analysis import (
    BatchIntelligenceReport,
    PatternMatchResult,
    RootCauseAnalysis,
)
from src.analysis.root_cause import RootCauseAnalyzer
from src.analysis.pattern_matching import PatternMatcher
from src.analysis.clustering import DenialClusterer


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""

    api_key: Optional[str] = None
    model: str = "claude-sonnet-4-6"
    top_k_similar: int = 5
    min_cluster_size: int = 2
    use_llm_for_clustering: bool = True
    output_dir: str = "outputs"
    skip_root_cause: bool = False
    skip_pattern_matching: bool = False
    skip_clustering: bool = False


@dataclass
class PipelineResult:
    """Full pipeline output for a batch of claims."""

    all_claims: list[JoinedClaim] = field(default_factory=list)
    denied_claims: list[JoinedClaim] = field(default_factory=list)
    root_cause_analyses: list[RootCauseAnalysis] = field(default_factory=list)
    pattern_match_results: list[PatternMatchResult] = field(default_factory=list)
    batch_report: Optional[BatchIntelligenceReport] = None

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_claims": len(self.all_claims),
                "denied_claims": len(self.denied_claims),
                "analyzed_claims": len(self.root_cause_analyses),
            },
            "root_cause_analyses": [r.model_dump() for r in self.root_cause_analyses],
            "pattern_match_results": [p.model_dump() for p in self.pattern_match_results],
            "batch_intelligence_report": (
                self.batch_report.model_dump() if self.batch_report else None
            ),
        }


class DenialAnalysisPipeline:
    """End-to-end pipeline for healthcare claim denial analysis.

    Architecture:
    1. Load & join EDI 835/837 data  (deterministic)
    2. Rule-based pre-analysis       (deterministic — grounded facts)
    3. LLM root cause analysis       (1 API call per denied claim, cached system prompt)
    4. Feature-based pattern matching (deterministic cosine similarity)
    5. LLM pattern interpretation    (1 API call per denied claim, cached system prompt)
    6. Rule + ML clustering          (deterministic KMeans on feature vectors)
    7. LLM cluster summarization     (1 API call per cluster, cached system prompt)

    Design rationale: LLM is used only where it adds unique value — interpreting ambiguous
    denial contexts, generating natural language summaries. All factual computations (filing
    days, similarity scores, clustering) are deterministic, cheap, and explainable.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        model = self.config.model

        self.loader = ClaimLoader()
        self.root_cause_analyzer = RootCauseAnalyzer(api_key=api_key, model=model)
        self.clusterer = DenialClusterer(api_key=api_key, model=model)

    def run_from_file(self, claims_path: str) -> PipelineResult:
        """Run full pipeline from a JSON claims file."""
        claims = self.loader.load_file(claims_path)
        return self.run(claims)

    def run(self, claims: list[JoinedClaim]) -> PipelineResult:
        """Run full pipeline on a list of JoinedClaim objects."""
        result = PipelineResult(all_claims=claims)
        result.denied_claims = [c for c in claims if c.is_denied]

        print(f"[Pipeline] {len(claims)} total claims | {len(result.denied_claims)} denied")

        # Problem 1: Root Cause Analysis
        if not self.config.skip_root_cause and result.denied_claims:
            print(f"[Pipeline] Running root cause analysis on {len(result.denied_claims)} denied claims...")
            result.root_cause_analyses = self.root_cause_analyzer.analyze_batch(
                result.denied_claims
            )
            print(f"[Pipeline] Root cause analysis complete: {len(result.root_cause_analyses)} analyzed")

        # Problem 2: Pattern Matching
        if not self.config.skip_pattern_matching and result.denied_claims:
            print("[Pipeline] Running pattern matching...")
            matcher = PatternMatcher(historical_claims=claims)
            for claim in result.denied_claims:
                try:
                    match_result = matcher.match(claim, top_k=self.config.top_k_similar)
                    result.pattern_match_results.append(match_result)
                except Exception as e:
                    print(f"[WARN] Pattern match failed for {claim.claim_id}: {e}")
            print(f"[Pipeline] Pattern matching complete: {len(result.pattern_match_results)} matched")

        # Problem 3: Clustering & Batch Intelligence
        if not self.config.skip_clustering and result.denied_claims:
            print("[Pipeline] Running clustering and batch intelligence...")
            try:
                result.batch_report = self.clusterer.cluster_and_report(
                    all_claims=claims,
                    root_cause_analyses=result.root_cause_analyses or None,
                    use_llm_summaries=self.config.use_llm_for_clustering,
                )
                print(
                    f"[Pipeline] Clustering complete: {len(result.batch_report.clusters)} clusters | "
                    f"${result.batch_report.total_recoverable_estimate:,.0f} estimated recoverable"
                )
            except Exception as e:
                print(f"[ERROR] Clustering failed: {e}")

        return result

    def save_result(self, result: PipelineResult, output_path: Optional[str] = None) -> str:
        """Save pipeline result to JSON file."""
        if output_path is None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(self.config.output_dir) / "pipeline_result.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        print(f"[Pipeline] Results saved to: {output_path}")
        return output_path
