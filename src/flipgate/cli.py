"""FlipGate CLI: Release gate for quantised/re-served LLMs."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .manifest import load_manifest
from .store import ResultsStore, compute_manifest_sha

console = Console()


@click.group()
@click.version_option()
def main():
    """FlipGate: Release gate for quantised/re-served LLMs.
    
    Counts per-item right-to-wrong answer flips against a measured
    bf16 noise floor, using paired statistics instead of aggregate accuracy.
    """
    pass


@main.command()
@click.option("--manifest", "-m", type=click.Path(exists=True), help="Path to manifest.yaml")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output path for report")
def info(manifest, output):
    """Show manifest and environment info."""
    manifest_path = Path(manifest) if manifest else None
    m = load_manifest(manifest_path)
    
    console.print(f"\n[bold cyan]FlipGate[/bold cyan] v0.1.0")
    console.print(f"Manifest version: {m['manifest_version']}")
    
    # Models
    table = Table(title="Models")
    table.add_column("Key", style="cyan")
    table.add_column("Model ID", style="green")
    table.add_column("Dtype")
    table.add_column("Quantization")
    
    for key, model in m["models"].items():
        table.add_row(
            key,
            model.get("id", "N/A"),
            model.get("dtype", "auto"),
            model.get("quantization", "-"),
        )
    console.print(table)
    
    # Datasets
    table = Table(title="Datasets")
    table.add_column("Key", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Items")
    table.add_column("Scorer")
    
    for key, ds in m["datasets"].items():
        table.add_row(
            key,
            ds["source"],
            str(ds["num_items"]),
            ds["scorer"],
        )
    console.print(table)
    
    # Sampling
    console.print(f"\n[bold]Sampling:[/bold] temp={m['sampling']['temperature']}, "
                  f"max_tokens={m['sampling']['max_tokens']}")


@main.command()
@click.option("--manifest", "-m", type=click.Path(exists=True), help="Path to manifest.yaml")
@click.option("--results-dir", "-r", type=click.Path(), default="data/results",
              help="Results directory")
def list_runs(manifest, results_dir):
    """List all evaluation runs."""
    store = ResultsStore(results_dir)
    runs = store.list_runs()
    
    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return
    
    table = Table(title="Evaluation Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Dataset")
    table.add_column("Batch Size")
    table.add_column("Created")
    
    for run_id in sorted(runs):
        try:
            meta = store.load_run_metadata(run_id)
            table.add_row(
                run_id[:40] + "..." if len(run_id) > 40 else run_id,
                meta.get("model", "N/A"),
                meta.get("engine", "N/A"),
                meta.get("dataset", "N/A"),
                str(meta.get("batch_size", "-")),
                meta.get("created", "N/A")[:19],
            )
        except FileNotFoundError:
            table.add_row(run_id[:40], "???", "???", "???", "???", "???")
    
    console.print(table)


@main.command()
@click.option("--baseline", "-b", required=True, help="Baseline run ID")
@click.option("--candidate", "-c", required=True, help="Candidate run ID")
@click.option("--dataset", "-d", required=True, help="Dataset name")
@click.option("--results-dir", "-r", type=click.Path(), default="data/results",
              help="Results directory")
@click.option("--alpha", "-a", default=0.05, help="Significance level")
@click.option("--noise-floor", "-n", default=0.0, help="Known noise floor flip rate")
@click.option("--margin", default=2.0, help="Fail if flip rate > margin * noise floor")
@click.option("--report", "report_path", type=click.Path(), default=None,
              help="Write Markdown report to this path")
def check(baseline, candidate, dataset, results_dir, alpha, noise_floor, margin, report_path):
    """Check if candidate passes the gate vs baseline.
    
    Compares right-to-wrong flips between baseline and candidate using
    McNemar's test and paired bootstrap CIs. Fails if flip rate exceeds
    the noise floor by the specified margin.
    """
    from .stats import mcnemar_test, paired_bootstrap_ci, compare_to_floor
    from .stats.mcnemar import count_flips
    
    store = ResultsStore(results_dir)
    
    # Load items
    baseline_items = {item["item_id"]: item for item in store.iter_items(baseline, dataset)}
    candidate_items = {item["item_id"]: item for item in store.iter_items(candidate, dataset)}
    
    # Find common items
    common_ids = sorted(set(baseline_items.keys()) & set(candidate_items.keys()))
    if not common_ids:
        console.print("[red]No common items found between runs.[/red]")
        sys.exit(1)
    
    # Build paired arrays
    baseline_correct = [baseline_items[iid]["score"] == 1.0 for iid in common_ids]
    candidate_correct = [candidate_items[iid]["score"] == 1.0 for iid in common_ids]
    
    # Statistics
    mcnemar = mcnemar_test(baseline_correct, candidate_correct, alpha=alpha)
    flips = count_flips(baseline_correct, candidate_correct)
    
    bootstrap_r2w = paired_bootstrap_ci(
        baseline_correct, candidate_correct,
        direction="right_to_wrong", seed=42
    )
    bootstrap_any = paired_bootstrap_ci(
        baseline_correct, candidate_correct,
        direction="any", seed=42
    )
    
    # Compare to noise floor
    floor_comparison = compare_to_floor(
        candidate_flip_rate=bootstrap_r2w.estimate,
        floor_estimate=noise_floor,
        floor_ci_upper=noise_floor,  # simplified
        margin=margin,
    )
    
    # Display results
    console.print(f"\n[bold cyan]FlipGate Check[/bold cyan]")
    console.print(f"Baseline:  {baseline}")
    console.print(f"Candidate: {candidate}")
    console.print(f"Dataset:   {dataset}")
    console.print(f"Items:     {len(common_ids)}\n")
    
    # Accuracy comparison
    baseline_acc = sum(baseline_correct) / len(baseline_correct)
    candidate_acc = sum(candidate_correct) / len(candidate_correct)
    acc_delta = candidate_acc - baseline_acc
    
    # Flip analysis table
    table = Table(title="Flip Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Baseline accuracy", f"{baseline_acc:.1%}")
    table.add_row("Candidate accuracy", f"{candidate_acc:.1%}")
    table.add_row("Accuracy delta", f"{acc_delta:+.1%}")
    table.add_row("", "")
    table.add_row("Right-to-wrong flips", str(flips["right_to_wrong"]))
    table.add_row("Wrong-to-right flips", str(flips["wrong_to_right"]))
    table.add_row("Total flips", str(flips["total_flips"]))
    table.add_row("", "")
    table.add_row("R-to-W flip rate", f"{bootstrap_r2w.estimate:.4f}")
    table.add_row("95% CI", f"[{bootstrap_r2w.ci_lower:.4f}, {bootstrap_r2w.ci_upper:.4f}]")
    table.add_row("McNemar p-value", f"{mcnemar.p_value:.4f}")
    table.add_row("Significant (p<{alpha})", "YES" if mcnemar.significant else "no")
    
    if noise_floor > 0:
        table.add_row("", "")
        table.add_row("Noise floor", f"{noise_floor:.4f}")
        table.add_row(f"Threshold ({margin}x floor)", f"{floor_comparison['threshold']:.4f}")
        table.add_row("Ratio (rate/floor)", f"{floor_comparison['ratio']:.1f}x")
    
    console.print(table)
    
    # Determine verdict
    fails = []
    if mcnemar.significant and mcnemar.n_01 > mcnemar.n_10:
        fails.append("significant right-to-wrong flip asymmetry")
    if noise_floor > 0 and not floor_comparison["passes"]:
        fails.append(f"flip rate exceeds {margin}x noise floor")
    
    # Verdict
    if fails:
        console.print(f"\n[bold red]FAIL[/bold red]: {', '.join(fails)}")
        if report_path:
            _write_report(report_path, baseline, candidate, dataset, common_ids,
                         baseline_acc, candidate_acc, flips, bootstrap_r2w, 
                         mcnemar, noise_floor, floor_comparison, "FAIL", fails)
        sys.exit(1)
    elif mcnemar.significant:
        console.print(f"\n[bold yellow]WARN[/bold yellow]: Significant difference, "
                      f"but candidate improves on baseline.")
        if report_path:
            _write_report(report_path, baseline, candidate, dataset, common_ids,
                         baseline_acc, candidate_acc, flips, bootstrap_r2w,
                         mcnemar, noise_floor, floor_comparison, "WARN", [])
    else:
        console.print(f"\n[bold green]PASS[/bold green]: No significant regression detected.")
        if report_path:
            _write_report(report_path, baseline, candidate, dataset, common_ids,
                         baseline_acc, candidate_acc, flips, bootstrap_r2w,
                         mcnemar, noise_floor, floor_comparison, "PASS", [])


def _write_report(path, baseline, candidate, dataset, common_ids,
                  baseline_acc, candidate_acc, flips, bootstrap, mcnemar,
                  noise_floor, floor_comparison, verdict, failures):
    """Write a Markdown report of the gate check."""
    from .store import ResultsStore
    store = ResultsStore("data/results")

    # Find exact flipped items
    baseline_items = {it["item_id"]: it for it in store.iter_items(baseline, dataset)}
    candidate_items = {it["item_id"]: it for it in store.iter_items(candidate, dataset)}
    r2w_ids = [iid for iid in common_ids
               if baseline_items[iid]["score"] == 1.0 and candidate_items[iid]["score"] != 1.0]
    w2r_ids = [iid for iid in common_ids
               if baseline_items[iid]["score"] != 1.0 and candidate_items[iid]["score"] == 1.0]

    lines = [
        f"# FlipGate Report: {verdict}",
        "",
        f"- **Baseline:** {baseline}",
        f"- **Candidate:** {candidate}",
        f"- **Dataset:** {dataset}",
        f"- **Items:** {len(common_ids)}",
        "",
        "## Accuracy",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Baseline accuracy | {baseline_acc:.1%} |",
        f"| Candidate accuracy | {candidate_acc:.1%} |",
        f"| Delta | {candidate_acc - baseline_acc:+.1%} |",
        "",
        "## Flip Analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Right-to-wrong flips | {flips['right_to_wrong']} |",
        f"| Wrong-to-right flips | {flips['wrong_to_right']} |",
        f"| Total flips | {flips['total_flips']} |",
        f"| R-to-W flip rate | {bootstrap.estimate:.4f} |",
        f"| 95% CI | [{bootstrap.ci_lower:.4f}, {bootstrap.ci_upper:.4f}] |",
        f"| McNemar p-value | {mcnemar.p_value:.4f} |",
        "",
    ]
    
    if noise_floor > 0:
        lines.extend([
            "## Noise Floor Comparison",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Noise floor | {noise_floor:.4f} |",
            f"| Threshold | {floor_comparison['threshold']:.4f} |",
            f"| Ratio | {floor_comparison['ratio']:.1f}x |",
            "",
        ])
    
    if failures:
        lines.extend([
            "## Failures",
            "",
        ] + [f"- {f}" for f in failures] + [""])

    lines.extend([
        "## Flipped Items",
        "",
        f"### Right-to-wrong ({len(r2w_ids)})",
        "",
    ] + [f"- `{iid}`" for iid in r2w_ids] + [
        "",
        f"### Wrong-to-right ({len(w2r_ids)})",
        "",
    ] + [f"- `{iid}`" for iid in w2r_ids] + [""])

    lines.append(f"**Verdict: {verdict}**")
    
    with open(path, "w") as f:
        f.write("\n".join(lines))
    
    console.print(f"\n[dim]Report written to {path}[/dim]")


@main.command()
@click.option("--model", "-m", required=True, help="Model key from manifest")
@click.option("--engine", "-e", default="vllm", help="Engine to use")
@click.option("--dataset", "-d", required=True, help="Dataset key from manifest")
@click.option("--batch-sizes", "-b", default="1,8,32", help="Comma-separated batch sizes")
@click.option("--results-dir", "-r", type=click.Path(), default="data/results")
@click.option("--manifest-path", type=click.Path(exists=True))
def noise_floor(model, engine, dataset, batch_sizes, results_dir, manifest_path):
    """Measure bf16-vs-bf16 noise floor.
    
    Runs the same model multiple times at different batch sizes to measure
    the inherent nondeterminism of the inference engine.
    """
    console.print(f"\n[bold cyan]Noise Floor Measurement[/bold cyan]")
    console.print(f"Model: {model}")
    console.print(f"Engine: {engine}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Batch sizes: {batch_sizes}")
    console.print(f"\n[yellow]Note: This command requires torch and vllm to be installed.[/yellow]")
    console.print(f"Run with: pip install torch vllm")
    console.print(f"\nThis will measure bf16-vs-bf16 flip rates to establish the noise floor.")


@main.command()
@click.option("--results-dir", "-r", type=click.Path(), default="data/results")
def report(results_dir):
    """Generate a summary report of all runs."""
    store = ResultsStore(results_dir)
    runs = store.list_runs()
    
    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return
    
    console.print(f"\n[bold cyan]FlipGate Report[/bold cyan]\n")
    
    for run_id in sorted(runs):
        try:
            meta = store.load_run_metadata(run_id)
            console.print(f"[bold]{run_id}[/bold]")
            console.print(f"  Model: {meta.get('model', 'N/A')}")
            console.print(f"  Engine: {meta.get('engine', 'N/A')}")
            console.print(f"  Dataset: {meta.get('dataset', 'N/A')}")
            
            # Count items
            items = list(store.iter_items(run_id, meta.get("dataset", "")))
            if items:
                correct = sum(1 for i in items if i["score"] == 1.0)
                console.print(f"  Items: {len(items)}, Correct: {correct}, "
                              f"Accuracy: {correct/len(items):.2%}")
            console.print()
        except FileNotFoundError:
            continue


if __name__ == "__main__":
    main()
