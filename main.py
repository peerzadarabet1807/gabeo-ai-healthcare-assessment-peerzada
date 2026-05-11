"""CLI entry point for the Gabeo AI Claim Denial Analysis System."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

app = typer.Typer(
    name="claim-denial-analyzer",
    help="AI-powered healthcare claim denial analysis system.",
    add_completion=False,
)
console = Console()


def _get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        rprint("[red]Error: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.[/red]")
        raise typer.Exit(1)
    return key


# ─── Commands ─────────────────────────────────────────────────────────────────


@app.command("generate-dataset")
def generate_dataset(
    output: str = typer.Option(
        "data/synthetic_dataset.json",
        "--output", "-o",
        help="Output path for the synthetic dataset JSON file.",
    )
):
    """Generate 30 synthetic claims (10 paid + 20 denied) for testing."""
    from src.data.generator import save_synthetic_dataset

    rprint(f"[cyan]Generating synthetic dataset...[/cyan]")
    claims = save_synthetic_dataset(output)
    rprint(f"[green]✓ Generated {len(claims)} synthetic claims → {output}[/green]")

    denied = [c for c in claims if c.get("claim_835", {}).get("pc_ClaimStatus") == "4"]
    paid = [c for c in claims if c.get("claim_835", {}).get("pc_ClaimStatus") != "4"]
    rprint(f"  Paid: {len(paid)} | Denied: {len(denied)}")


@app.command("analyze")
def analyze_single(
    claim_id: str = typer.Argument(help="Claim ID to analyze (must exist in the input file)"),
    input_file: str = typer.Option(
        "data/sample_claims.json",
        "--input", "-i",
        help="JSON file containing claims data.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Save result to this JSON file.",
    ),
):
    """Analyze a single denied claim (root cause + pattern matching)."""
    api_key = _get_api_key()

    from src.data.loader import ClaimLoader
    from src.analysis.root_cause import RootCauseAnalyzer
    from src.analysis.pattern_matching import PatternMatcher

    loader = ClaimLoader()
    all_claims = loader.load_file(input_file)
    target = next((c for c in all_claims if c.claim_id == claim_id), None)

    if not target:
        rprint(f"[red]Claim '{claim_id}' not found in {input_file}[/red]")
        raise typer.Exit(1)

    if not target.is_denied:
        rprint(f"[yellow]Claim '{claim_id}' is not denied (status: {target.claim_835.pc_ClaimStatus})[/yellow]")
        raise typer.Exit(1)

    rprint(f"\n[bold cyan]Analyzing claim: {claim_id}[/bold cyan]")
    rprint(f"  Payer: {target.payer_name}")
    rprint(f"  Amount: ${target.claimed_amount:,.2f}")
    rprint(f"  CARC: {target.carc_code}")

    # Root cause analysis
    rprint("\n[cyan]Running root cause analysis...[/cyan]")
    analyzer = RootCauseAnalyzer(api_key=api_key)
    rca = analyzer.analyze(target)

    # Pattern matching
    rprint("[cyan]Running pattern matching...[/cyan]")
    matcher = PatternMatcher(historical_claims=all_claims)
    pattern = matcher.match(target)

    # Display results
    _display_root_cause(rca)
    _display_pattern_match(pattern)

    if output:
        result = {
            "claim_id": claim_id,
            "root_cause_analysis": rca.model_dump(),
            "pattern_match": pattern.model_dump(),
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        rprint(f"\n[green]✓ Result saved to {output}[/green]")


@app.command("analyze-batch")
def analyze_batch(
    input_file: str = typer.Option(
        "data/synthetic_dataset.json",
        "--input", "-i",
        help="JSON file containing claims (paid + denied).",
    ),
    output: str = typer.Option(
        "outputs/pipeline_result.json",
        "--output", "-o",
        help="Output path for full pipeline results.",
    ),
    skip_clustering: bool = typer.Option(False, "--skip-clustering"),
    no_llm_clustering: bool = typer.Option(False, "--no-llm-clustering"),
):
    """Run the full pipeline on a batch of claims."""
    api_key = _get_api_key()

    from src.pipeline import DenialAnalysisPipeline, PipelineConfig

    config = PipelineConfig(
        api_key=api_key,
        skip_clustering=skip_clustering,
        use_llm_for_clustering=not no_llm_clustering,
    )
    pipeline = DenialAnalysisPipeline(config=config)

    rprint(f"\n[bold cyan]Running full pipeline on: {input_file}[/bold cyan]")
    result = pipeline.run_from_file(input_file)
    saved_path = pipeline.save_result(result, output)

    # Display summary
    _display_batch_summary(result)
    rprint(f"\n[green]✓ Full results saved to: {saved_path}[/green]")


@app.command("cluster")
def cluster_only(
    input_file: str = typer.Option(
        "data/synthetic_dataset.json",
        "--input", "-i",
        help="JSON file containing denied claims.",
    ),
    output: str = typer.Option(
        "outputs/cluster_report.json",
        "--output", "-o",
        help="Output path for cluster report.",
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM-generated summaries."),
):
    """Cluster denied claims and generate a batch intelligence report."""
    api_key = _get_api_key()

    from src.data.loader import ClaimLoader
    from src.analysis.clustering import DenialClusterer

    loader = ClaimLoader()
    all_claims = loader.load_file(input_file)
    denied = [c for c in all_claims if c.is_denied]

    rprint(f"\n[bold cyan]Clustering {len(denied)} denied claims from {input_file}[/bold cyan]")

    clusterer = DenialClusterer(api_key=api_key)
    report = clusterer.cluster_and_report(all_claims, use_llm_summaries=not no_llm)

    _display_cluster_report(report)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
    rprint(f"\n[green]✓ Cluster report saved to: {output}[/green]")


# ─── Display Helpers ──────────────────────────────────────────────────────────


def _display_root_cause(rca):
    verdict_color = {
        "recoverable": "green",
        "not_recoverable": "red",
        "needs_review": "yellow",
    }.get(rca.recoverability_verdict.value, "white")

    console.print(
        Panel(
            f"[bold]Root Cause:[/bold] {rca.denial_root_cause}\n\n"
            f"[bold]CARC {rca.carc_code} Interpretation:[/bold] {rca.carc_interpretation}\n\n"
            f"[bold]Verdict:[/bold] [{verdict_color}]{rca.recoverability_verdict.value.upper()}[/{verdict_color}] "
            f"(confidence: {rca.confidence_score:.0%})\n\n"
            f"[bold]Recommended Action:[/bold] {rca.recommended_action}\n\n"
            f"[bold]Appeal Strategy:[/bold] {rca.appeal_strategy or 'N/A'}\n"
            f"[bold]Appeal Deadline:[/bold] {rca.appeal_deadline_estimate or 'N/A'}",
            title=f"[bold]Root Cause Analysis — {rca.claim_id}[/bold]",
            border_style=verdict_color,
        )
    )

    if rca.supporting_evidence:
        console.print("\n[bold]Supporting Evidence:[/bold]")
        for e in rca.supporting_evidence[:4]:
            console.print(f"  • {e.field_name} = '{e.field_value}' → {e.significance}")


def _display_pattern_match(pattern):
    adj_color = {"strengthened": "green", "weakened": "red", "neutral": "yellow"}.get(
        pattern.recoverability_adjustment, "white"
    )
    console.print(
        Panel(
            f"[bold]Pattern Summary:[/bold] {pattern.pattern_summary}\n\n"
            f"[bold]Recoverability Adjustment:[/bold] [{adj_color}]{pattern.recoverability_adjustment.upper()}[/{adj_color}]\n"
            f"[bold]Payer Denial Rate:[/bold] {pattern.payer_denial_rate:.0%}" if pattern.payer_denial_rate else "",
            title="[bold]Historical Pattern Analysis[/bold]",
            border_style=adj_color,
        )
    )


def _display_batch_summary(result):
    table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Total Claims Analyzed", str(len(result.all_claims)))
    table.add_row("Denied Claims", str(len(result.denied_claims)))
    table.add_row("Root Cause Analyses", str(len(result.root_cause_analyses)))
    table.add_row("Pattern Matches", str(len(result.pattern_match_results)))

    if result.batch_report:
        table.add_row("Clusters Found", str(len(result.batch_report.clusters)))
        table.add_row(
            "Total Denied Amount", f"${result.batch_report.total_denied_amount:,.2f}"
        )
        table.add_row(
            "Estimated Recoverable", f"${result.batch_report.total_recoverable_estimate:,.2f}"
        )

    console.print(table)

    if result.batch_report:
        rprint(f"\n[bold]Executive Summary:[/bold]")
        rprint(result.batch_report.executive_summary)

        rprint(f"\n[bold]Quick Wins:[/bold]")
        for win in result.batch_report.quick_wins:
            rprint(f"  • {win}")


def _display_cluster_report(report):
    console.print(
        Panel(
            report.executive_summary,
            title="[bold]Batch Intelligence Report — Executive Summary[/bold]",
            border_style="cyan",
        )
    )

    table = Table(title="Denial Clusters (by Priority)", show_header=True, header_style="bold")
    table.add_column("Cluster", style="cyan")
    table.add_column("Claims", justify="right")
    table.add_column("Denied Amount", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Recoverable", justify="right", style="green")

    for cluster in report.clusters[:10]:
        table.add_row(
            cluster.cluster_label[:50],
            str(cluster.claim_count),
            f"${cluster.total_denied_amount:,.0f}",
            f"{cluster.historical_appeal_success_rate:.0%}",
            f"${cluster.recoverable_amount_estimate:,.0f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
