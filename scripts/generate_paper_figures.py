"""
Generate all paper figures and LaTeX table fragments from experiment results.

Usage:
    .venv\\Scripts\\python.exe scripts/generate_paper_figures.py

Output:
    paper/figures/*.pdf  — Publication-quality figures
    paper/tables/*.tex   — LaTeX table fragments
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RESULTS_DIR = Path("results")
FIG_DIR = Path("paper/figures")
TAB_DIR = Path("paper/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES_5TASK = [
    "bottle", "cable", "capsule",   # Task 0
    "carpet", "grid", "hazelnut",   # Task 1
    "leather", "metal_nut", "pill", # Task 2
    "screw", "tile", "toothbrush",  # Task 3
    "transistor", "wood", "zipper", # Task 4
]

COLORS = {
    "MemoryAD": "#2196F3",
    "Joint": "#4CAF50",
    "Naive": "#F44336",
    "EWC": "#FF9800",
    "LwF": "#9C27B0",
    "Replay": "#00BCD4",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════
# F2: Incremental I-AUROC after each task (E1)
# ══════════════════════════════════════════════════════════
def generate_f2():
    """Line plot: mean I-AUROC after each task for MemoryAD."""
    r = load_json(RESULTS_DIR / "E1_mvtec_5task" / "results.json")
    am = r["auroc_matrix"]  # [task][category], NaN for unseen

    # Compute mean AUROC over seen categories after each task
    mean_aurocs = []
    for t in range(len(am)):
        seen = [v for v in am[t] if not (v != v)]  # filter NaN
        mean_aurocs.append(np.mean(seen))

    # Also load baselines for reference lines
    baselines = load_json(RESULTS_DIR / "baseline_comparison.json")

    fig, ax = plt.subplots(figsize=(5, 3.5))

    tasks = list(range(len(mean_aurocs)))
    ax.plot(tasks, mean_aurocs, "o-", color=COLORS["MemoryAD"],
            linewidth=2, markersize=6, label="MemoryAD (ours)", zorder=5)

    # Reference lines
    ax.axhline(baselines["Joint (upper)"], color=COLORS["Joint"],
               linestyle="--", linewidth=1.2, label="Joint (upper bound)")
    ax.axhline(baselines["Naive (lower)"], color=COLORS["Naive"],
               linestyle=":", linewidth=1.2, label="Naive (lower bound)")

    ax.set_xlabel("Task Index")
    ax.set_ylabel("Mean I-AUROC")
    ax.set_xticks(tasks)
    ax.set_xticklabels([f"T{t}" for t in tasks])
    ax.set_ylim(0.70, 1.0)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Incremental I-AUROC on MVTec AD (5-task)")

    fig.savefig(FIG_DIR / "f2_incremental_auroc.pdf")
    fig.savefig(FIG_DIR / "f2_incremental_auroc.png")
    plt.close(fig)
    print("[OK] F2: Incremental AUROC line plot")


# ══════════════════════════════════════════════════════════
# F3: 15x15 Forgetting Heatmap (E5)
# ══════════════════════════════════════════════════════════
def generate_f3():
    """Heatmap: AUROC of category i after learning task j."""
    r = load_json(RESULTS_DIR / "E5_mvtec_15task" / "results.json")
    am = r["auroc_matrix"]  # [15 tasks][15 categories]

    matrix = np.array(am)  # (15, 15)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Mask NaN values (upper triangle where categories haven't been seen)
    mask = np.isnan(matrix)

    sns.heatmap(
        matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0.4, vmax=1.0,
        xticklabels=CATEGORIES_5TASK,
        yticklabels=[f"T{i}" for i in range(15)],
        ax=ax, cbar_kws={"label": "I-AUROC", "shrink": 0.8},
        annot_kws={"size": 6},
        linewidths=0.3, linecolor="white",
    )

    ax.set_xlabel("Category")
    ax.set_ylabel("After Task")
    ax.set_title("Category I-AUROC After Each Task (15-task, E5)")
    plt.xticks(rotation=45, ha="right")

    fig.savefig(FIG_DIR / "f3_forgetting_heatmap.pdf")
    fig.savefig(FIG_DIR / "f3_forgetting_heatmap.png")
    plt.close(fig)
    print("[OK] F3: Forgetting heatmap")


# ══════════════════════════════════════════════════════════
# F4: Budget vs AUROC Curve (E3)
# ══════════════════════════════════════════════════════════
def generate_f4():
    """Line plot: I-AUROC and forgetting rate vs coreset budget."""
    budgets = [1000, 5000, 10000, 20000, 50000]
    aurocs = []
    forgetting = []

    for b in budgets:
        r = load_json(RESULTS_DIR / f"E3_budget_{b}" / "results.json")
        aurocs.append(r["final_mean_auroc"])
        forgetting.append(r["forgetting_rate"])

    fig, ax1 = plt.subplots(figsize=(5, 3.5))

    color1 = COLORS["MemoryAD"]
    color2 = "#F44336"

    ax1.plot(budgets, aurocs, "o-", color=color1, linewidth=2,
             markersize=6, label="I-AUROC")
    ax1.set_xlabel("Coreset Budget (B)")
    ax1.set_ylabel("Mean I-AUROC", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.84, 0.98)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x))))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(budgets, forgetting, "s--", color=color2, linewidth=1.5,
             markersize=5, label="Forgetting Rate")
    ax2.set_ylabel("Forgetting Rate", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-0.005, 0.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Budget Ablation (E3)")

    fig.savefig(FIG_DIR / "f4_budget_curve.pdf")
    fig.savefig(FIG_DIR / "f4_budget_curve.png")
    plt.close(fig)
    print("[OK] F4: Budget ablation curve")


# ══════════════════════════════════════════════════════════
# F5: Backbone Comparison (E4)
# ══════════════════════════════════════════════════════════
def generate_f5():
    """Bar chart: backbone comparison on I-AUROC and forgetting."""
    backbones = {
        "DINOv2\nViT-B/14": "E4_backbone_dinov2b_14",
        "CLIP\nViT-L/14": "E4_backbone_clipl_14",
        "WRN-50": "E4_backbone_wrn50",
    }

    names = list(backbones.keys())
    aurocs = []
    forgetting = []

    for name, dirname in backbones.items():
        r = load_json(RESULTS_DIR / dirname / "results.json")
        aurocs.append(r["final_mean_auroc"])
        forgetting.append(r["forgetting_rate"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    colors = ["#2196F3", "#FF9800", "#9E9E9E"]

    # I-AUROC bars
    bars1 = ax1.bar(names, aurocs, color=colors, edgecolor="white", width=0.6)
    ax1.set_ylabel("Mean I-AUROC")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Detection Performance")
    for bar, val in zip(bars1, aurocs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Forgetting bars
    bars2 = ax2.bar(names, forgetting, color=colors, edgecolor="white", width=0.6)
    ax2.set_ylabel("Forgetting Rate")
    ax2.set_ylim(0, 0.035)
    ax2.set_title("Catastrophic Forgetting")
    for bar, val in zip(bars2, forgetting):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Backbone Ablation (E4)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f5_backbone_comparison.pdf")
    fig.savefig(FIG_DIR / "f5_backbone_comparison.png")
    plt.close(fig)
    print("[OK] F5: Backbone comparison")


# ══════════════════════════════════════════════════════════
# F6: CIL Strategy Comparison (E6)
# ══════════════════════════════════════════════════════════
def generate_f6():
    """Grouped bar chart: strategy comparison."""
    strategies = {
        "Proportional": "E6_strategy_proportional",
        "Weighted": "E6_strategy_weighted",
        "Recency": "E6_strategy_recency",
    }

    names = list(strategies.keys())
    aurocs = []
    forgetting = []
    avg_inc = []

    for name, dirname in strategies.items():
        r = load_json(RESULTS_DIR / dirname / "results.json")
        aurocs.append(r["final_mean_auroc"])
        forgetting.append(r["forgetting_rate"])
        avg_inc.append(r["avg_incremental_auroc"])

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(len(names))
    w = 0.25

    bars1 = ax.bar(x - w, aurocs, w, label="Final I-AUROC", color="#2196F3")
    bars2 = ax.bar(x,     avg_inc, w, label="Avg Inc. AUROC", color="#4CAF50")
    bars3 = ax.bar(x + w, forgetting, w, label="Forgetting Rate", color="#F44336")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("CIL Strategy Comparison (E6)")
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate I-AUROC values
    for bar, val in zip(bars1, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "f6_strategy_comparison.pdf")
    fig.savefig(FIG_DIR / "f6_strategy_comparison.png")
    plt.close(fig)
    print("[OK] F6: Strategy comparison")


# ══════════════════════════════════════════════════════════
# T1: Main Results Table (E1 + baselines)
# ══════════════════════════════════════════════════════════
def generate_t1():
    """LaTeX table: main results comparing all methods."""
    r1 = load_json(RESULTS_DIR / "E1_mvtec_5task" / "results.json")
    baselines = load_json(RESULTS_DIR / "baseline_comparison.json")

    # Rows: method, I-AUROC, Forgetting, Avg Inc, Forward Transfer
    rows = [
        ("Joint (upper bound)", baselines["Joint (upper)"], "--", "--", "--"),
        ("\\textbf{MemoryAD (ours)}", r1["final_mean_auroc"],
         r1["forgetting_rate"], r1["avg_incremental_auroc"], r1["forward_transfer"]),
        ("Replay + RD4AD", baselines["Replay + RD4AD"], "--", "--", "--"),
        ("Naive (lower bound)", baselines["Naive (lower)"], "--", "--", "--"),
        ("EWC + RD4AD", baselines["EWC + RD4AD"], "--", "--", "--"),
        ("LwF + RD4AD", baselines["LwF + RD4AD"], "--", "--", "--"),
    ]

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Main results on MVTec AD (5-task, 15 categories). I-AUROC (\%) after all tasks. MemoryAD uses DINOv2 ViT-B/14 with budget $B=10\text{K}$.}")
    tex.append(r"\label{tab:main_results}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"Method & I-AUROC $\uparrow$ & FR $\downarrow$ & Avg Inc. & FT $\uparrow$ \\")
    tex.append(r"\midrule")

    for name, auroc, fr, avg, ft in rows:
        auroc_s = f"{auroc*100:.1f}" if isinstance(auroc, float) else auroc
        fr_s = f"{fr*100:.2f}" if isinstance(fr, float) else fr
        avg_s = f"{avg*100:.1f}" if isinstance(avg, float) else avg
        ft_s = f"{ft*100:.1f}" if isinstance(ft, float) else ft
        tex.append(f"{name} & {auroc_s} & {fr_s} & {avg_s} & {ft_s} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    with open(TAB_DIR / "t1_main_results.tex", "w") as f:
        f.write("\n".join(tex))
    print("[OK] T1: Main results table")


# ══════════════════════════════════════════════════════════
# T2: VisA Results Table (E2)
# ══════════════════════════════════════════════════════════
def generate_t2():
    """LaTeX table: VisA generalization results."""
    r = load_json(RESULTS_DIR / "E2_visa_4task" / "results.json")

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Generalization to VisA (4-task, 12 categories). MemoryAD with DINOv2 ViT-B/14, $B=10\text{K}$.}")
    tex.append(r"\label{tab:visa_results}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"Dataset & I-AUROC & FR & Avg Inc. & FT \\")
    tex.append(r"\midrule")

    r1 = load_json(RESULTS_DIR / "E1_mvtec_5task" / "results.json")
    tex.append(f"MVTec AD & {r1['final_mean_auroc']*100:.1f} & {r1['forgetting_rate']*100:.2f} & {r1['avg_incremental_auroc']*100:.1f} & {r1['forward_transfer']*100:.1f} \\\\")
    tex.append(f"VisA & {r['final_mean_auroc']*100:.1f} & {r['forgetting_rate']*100:.2f} & {r['avg_incremental_auroc']*100:.1f} & {r['forward_transfer']*100:.1f} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    with open(TAB_DIR / "t2_visa_results.tex", "w") as f:
        f.write("\n".join(tex))
    print("[OK] T2: VisA results table")


# ══════════════════════════════════════════════════════════
# T3: Inference Speed Table (E7)
# ══════════════════════════════════════════════════════════
def generate_t3():
    """LaTeX table: inference speed breakdown."""
    r = load_json(RESULTS_DIR / "E7_inference_speed" / "results.json")

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Inference time breakdown for MemoryAD ($B=10\text{K}$, MVTec AD full evaluation).}")
    tex.append(r"\label{tab:inference_speed}")
    tex.append(r"\begin{tabular}{lrr}")
    tex.append(r"\toprule")
    tex.append(r"Component & Time (s) & \% Total \\")
    tex.append(r"\midrule")

    total = r["total_s"]
    components = [
        ("Feature loading", r["feature_load_s"]),
        ("Coreset update", r["coreset_update_s"]),
        ("k-NN scorer fit", r["scorer_fit_s"]),
        ("Evaluation", r["evaluation_s"]),
    ]
    for name, t in components:
        pct = t / total * 100
        tex.append(f"{name} & {t:.1f} & {pct:.1f}\\% \\\\")

    tex.append(r"\midrule")
    tex.append(f"\\textbf{{Total}} & \\textbf{{{total:.1f}}} & 100\\% \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\vspace{1mm}")
    tex.append(f"\\footnotesize{{Images scored: {r['images_scored']:,}. Throughput: {r['ms_per_image']:.1f} ms/image.}}")
    tex.append(r"\end{table}")

    with open(TAB_DIR / "t3_inference_speed.tex", "w") as f:
        f.write("\n".join(tex))
    print("[OK] T3: Inference speed table")


# ══════════════════════════════════════════════════════════
# T4: Backbone Ablation Table (E4)
# ══════════════════════════════════════════════════════════
def generate_t4():
    """LaTeX table: backbone ablation."""
    backbones = [
        ("DINOv2 ViT-B/14", "E4_backbone_dinov2b_14"),
        ("CLIP ViT-L/14", "E4_backbone_clipl_14"),
        ("WideResNet-50", "E4_backbone_wrn50"),
    ]

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Backbone ablation on MVTec AD (5-task). All methods use $B=10\text{K}$ coreset budget.}")
    tex.append(r"\label{tab:backbone_ablation}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"Backbone & I-AUROC & FR & Avg Inc. \\")
    tex.append(r"\midrule")

    for name, dirname in backbones:
        r = load_json(RESULTS_DIR / dirname / "results.json")
        tex.append(f"{name} & {r['final_mean_auroc']*100:.1f} & {r['forgetting_rate']*100:.2f} & {r['avg_incremental_auroc']*100:.1f} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    with open(TAB_DIR / "t4_backbone_ablation.tex", "w") as f:
        f.write("\n".join(tex))
    print("[OK] T4: Backbone ablation table")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 50)
    print("Generating paper figures and tables")
    print("=" * 50)

    # Figures
    generate_f2()
    generate_f3()
    generate_f4()
    generate_f5()
    generate_f6()

    # Tables
    generate_t1()
    generate_t2()
    generate_t3()
    generate_t4()

    print(f"\nFigures saved to: {FIG_DIR}/")
    print(f"Tables saved to: {TAB_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
