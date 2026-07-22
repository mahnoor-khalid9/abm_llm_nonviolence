"""
visualize_attention.py - Attention Flow Visualization (ZS + CoT)
=================================================================
Consumes the .npz files produced by extract_attention.py (which now contains
both ZeroShot and CoT runs) and produces publication figures.

Data shape per model file:
    section_attention      [n_prompt_types, n_scenarios, n_layers, n_sections]
    layer_world_attention  [n_prompt_types, n_scenarios, n_layers]
    decisions              JSON list (len = n_prompt_types * n_scenarios)
    prompt_types           ["zeroshot", "cot"]

Figures produced:
  1. attention_decisions.{pdf,png}        Scenario x Model x Prompt -> target
  2. attention_doctrine_survival.{pdf,png}  Per-layer attention to <world>, ZS vs CoT
  3. attention_section_heatmap.{pdf,png}    Per-layer per-section heatmap (CoT)
  4. attention_doctrine_vs_context.{pdf,png}  Per-scenario doctrine ratio bars
  5. attention_difference_map.{pdf,png}     Qwen minus others (CoT)
  6. attention_ministral_failure.{pdf,png}  ZS vs CoT for Ministral (the key story)

Usage:
    python visualize_attention.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "figure.dpi": 150,
})

DATA_DIR = Path("attention_data")
OUT_DIR  = Path("attention_figures")
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    {"short": "llama",     "label": "Llama-3.1-8B"},
    {"short": "ministral", "label": "Ministral-8B"},
    {"short": "qwen",      "label": "Qwen3-8B"},
]

SECTION_DISPLAY = {
    "role":     ("Role",          "#9E9E9E"),
    "persona":  ("Persona",       "#7E57C2"),
    "world":    ("Doctrine",      "#2E7D32"),
    "scenario": ("Scenario",      "#1565C0"),
    "memory":   ("Memory",        "#F57F17"),
    "context":  ("Context",       "#E65100"),
    "task":     ("Task",          "#AD1457"),
    "output":   ("Output Schema", "#546E7A"),
}

MODEL_COLORS = {
    "llama":     "#1565C0",
    "ministral": "#E65100",
    "qwen":      "#2E7D32",
}

PROMPT_LINESTYLES = {"zeroshot": "--", "cot": "-"}

TARGET_COLORS = {
    "pillars":   "#2E7D32",
    "citizens":  "#1565C0",
    "police":    "#C62828",
    "activists": "#7E57C2",
    "unknown":   "#9E9E9E",
    "parse_error": "#F44336",
}


def load_data(short_name):
    path = DATA_DIR / f"{short_name}_attention.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "section_attn":   data["section_attention"],         # [pt, sc, layer, sec]
        "world":          data["layer_world_attention"],     # [pt, sc, layer]
        "head_wc":        data["head_world_vs_context"],     # [pt, sc, layer, head, 2]
        "section_names":  list(data["section_names"]),
        "prompt_types":   list(data["prompt_types"]),
        "labels":         list(data["scenario_labels"]),
        "descriptions":   list(data["scenario_descriptions"]),
        "n_layers":       int(data["n_layers"]),
        "n_heads":        int(data["n_heads"]),
        "decisions":      json.loads(str(data["decisions"])),
    }


def pt_idx(data, pt):
    """Get index of prompt type in this model's data."""
    return data["prompt_types"].index(pt)


def get_decision(data, prompt_type, scenario_idx):
    """Look up a decision dict by (prompt_type, scenario_idx)."""
    for d in data["decisions"]:
        if d["prompt_type"] == prompt_type and d["scenario_idx"] == scenario_idx:
            return d
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: DECISION MATRIX (ZS vs CoT side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def plot_decision_matrix(all_data):
    labels = descriptions = None
    for m in MODELS:
        d = all_data.get(m["short"])
        if d:
            labels, descriptions = d["labels"], d["descriptions"]
            break
    if not labels:
        return

    n_scenarios = len(labels)
    n_models = len(MODELS)

    fig, ax = plt.subplots(figsize=(13, 0.55 * n_scenarios + 1), facecolor="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Column layout: [scenario | (ZS,CoT) per model]
    label_x = 0.02
    col_x = []
    col_width = (1.0 - 0.30) / n_models
    for j in range(n_models):
        zs_x  = 0.30 + j * col_width + col_width * 0.25
        cot_x = 0.30 + j * col_width + col_width * 0.75
        col_x.append((zs_x, cot_x))

    # Headers
    ax.text(label_x, 0.97, "Scenario", fontsize=8.5, fontweight="bold")
    for j, m in enumerate(MODELS):
        zs_x, cot_x = col_x[j]
        cx = (zs_x + cot_x) / 2
        ax.text(cx, 0.985, m["label"], fontsize=8.5, fontweight="bold", ha="center")
        ax.text(zs_x,  0.955, "ZS",  fontsize=7, color="#666", ha="center")
        ax.text(cot_x, 0.955, "CoT", fontsize=7, color="#666", ha="center")

    row_h = 0.92 / n_scenarios
    for i in range(n_scenarios):
        y = 0.94 - (i + 1) * row_h + row_h * 0.45
        short_label = labels[i].split("_", 1)[-1].replace("_", " ").title()
        ax.text(label_x, y + 0.01, short_label, fontsize=7, fontweight="bold")
        ax.text(label_x, y - 0.012, descriptions[i][:65], fontsize=5.5, color="#666")

        for j, m in enumerate(MODELS):
            data = all_data.get(m["short"])
            if not data:
                continue
            zs_x, cot_x = col_x[j]
            for pt, x in [("zeroshot", zs_x), ("cot", cot_x)]:
                dec = get_decision(data, pt, i)
                if dec is None:
                    continue
                target = dec["target"]
                color = TARGET_COLORS.get(target, "#999")
                ax.text(x, y + 0.005, target.upper()[:9], fontsize=6.5,
                        fontweight="bold", color=color, ha="center")
                ax.text(x, y - 0.012, dec.get("attribute", "")[:10],
                        fontsize=5, color="#777", ha="center")

        # Row separator: use plot() not axhline (axhline doesn't accept transform)
        ysep = 0.94 - (i + 1) * row_h
        ax.plot([0, 1], [ysep, ysep], color="#EEE", linewidth=0.4,
                transform=ax.transAxes, clip_on=False)

    plt.suptitle("Decision Matrix: Same Scenario, Different Models, ZS vs CoT",
                 fontsize=10.5, fontweight="bold", y=1.01)
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_DIR / f"attention_decisions.{ext}", bbox_inches="tight", dpi=300)
    print("  Saved: attention_decisions.{pdf,png}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: DOCTRINE SURVIVAL BY LAYER (ZS vs CoT, all models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_doctrine_survival(all_data):
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="white")

    for m in MODELS:
        data = all_data.get(m["short"])
        if not data:
            continue
        for pt in ["zeroshot", "cot"]:
            try:
                idx = pt_idx(data, pt)
            except ValueError:
                continue
            # Mean across scenarios
            mean_world = data["world"][idx].mean(axis=0)  # [n_layers]
            layers = np.arange(1, len(mean_world) + 1)
            ax.plot(layers, mean_world,
                    color=MODEL_COLORS[m["short"]],
                    linestyle=PROMPT_LINESTYLES[pt],
                    linewidth=1.6, alpha=0.9,
                    label=f"{m['label']} ({pt.upper()})")

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Attention to Doctrine (<world>)", fontsize=9)
    ax.set_title("Does Doctrine Knowledge Survive Through Layers?",
                 fontsize=10.5, fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, ncol=3, loc="upper right", frameon=True, edgecolor="0.8")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_DIR / f"attention_doctrine_survival.{ext}", bbox_inches="tight", dpi=300)
    print("  Saved: attention_doctrine_survival.{pdf,png}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: SECTION HEATMAP (CoT only, all models side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def plot_section_heatmap(all_data):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 3.8),
                             sharey=True, facecolor="white")
    if len(MODELS) == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list(
        "section", ["#FFFFFF", "#C8E6C9", "#66BB6A", "#2E7D32", "#1B5E20"]
    )

    for idx, (m, ax) in enumerate(zip(MODELS, axes)):
        data = all_data.get(m["short"])
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        try:
            cot_idx = pt_idx(data, "cot")
        except ValueError:
            continue

        # Mean across scenarios -> [n_layers, n_sections]
        mean_attn = data["section_attn"][cot_idx].mean(axis=0)

        # Normalize per layer (so each row sums to 100%)
        row_sums = mean_attn.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized = mean_attn / row_sums * 100

        display = [SECTION_DISPLAY.get(s, (s, "#999"))[0] for s in data["section_names"]]

        im = ax.imshow(normalized.T, aspect="auto", cmap=cmap,
                       vmin=0, vmax=35, interpolation="nearest")

        ax.set_xlabel("Layer", fontsize=8)
        ax.set_title(f"{m['label']} (CoT)", fontsize=10, fontweight="bold", pad=6)

        if idx == 0:
            ax.set_yticks(range(len(display)))
            ax.set_yticklabels(display, fontsize=7)

        n_lay = normalized.shape[0]
        step = max(1, n_lay // 6)
        ax.set_xticks(range(0, n_lay, step))
        ax.set_xticklabels(range(1, n_lay + 1, step), fontsize=7)

    fig.colorbar(im, ax=axes, label="Attention %", shrink=0.85, pad=0.02)
    fig.suptitle("Per-Layer Attention to Prompt Sections (CoT, averaged across scenarios)",
                 fontsize=10.5, fontweight="bold", y=1.02)
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_DIR / f"attention_section_heatmap.{ext}", bbox_inches="tight", dpi=300)
    print("  Saved: attention_section_heatmap.{pdf,png}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: PER-SCENARIO DOCTRINE vs CONTEXT (CoT, all models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_scenario_attention(all_data):
    labels = None
    for m in MODELS:
        d = all_data.get(m["short"])
        if d:
            labels = d["labels"]
            break
    if not labels:
        return

    n_scenarios = len(labels)
    fig, ax = plt.subplots(figsize=(13, 3.8), facecolor="white")

    x = np.arange(n_scenarios)
    width = 0.25

    for j, m in enumerate(MODELS):
        data = all_data.get(m["short"])
        if not data:
            continue
        try:
            cot_idx = pt_idx(data, "cot")
        except ValueError:
            continue

        section_names = data["section_names"]
        if "world" not in section_names or "context" not in section_names:
            continue
        w_i = section_names.index("world")
        c_i = section_names.index("context")

        ratios = []
        for i in range(n_scenarios):
            w_attn = data["section_attn"][cot_idx, i, :, w_i].sum()
            c_attn = data["section_attn"][cot_idx, i, :, c_i].sum()
            total = w_attn + c_attn
            ratios.append(w_attn / total * 100 if total > 0 else 50)

        ax.bar(x + (j - 1) * width, ratios, width,
               label=m["label"], color=MODEL_COLORS[m["short"]],
               alpha=0.88, edgecolor="white", linewidth=0.4)

    short_labels = [l.split("_", 1)[-1].replace("_", "\n") for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=6.5)
    ax.set_ylabel("Doctrine / (Doctrine + Context) Attention %", fontsize=8.5)
    ax.axhline(y=50, color="#999", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title("Doctrine vs Context Attention by Scenario (CoT)",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="upper right", frameon=True, edgecolor="0.8")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_DIR / f"attention_doctrine_vs_context.{ext}", bbox_inches="tight", dpi=300)
    print("  Saved: attention_doctrine_vs_context.{pdf,png}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5: ATTENTION DIFFERENCE MAP (Qwen minus others, CoT)
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_difference(all_data):
    qwen = all_data.get("qwen")
    if not qwen:
        return
    try:
        qwen_cot = pt_idx(qwen, "cot")
    except ValueError:
        return

    diff_cmap = LinearSegmentedColormap.from_list(
        "diff", ["#C62828", "#FFCDD2", "#FFFFFF", "#C8E6C9", "#2E7D32"]
    )

    comparisons = [("llama", "Qwen minus Llama"),
                   ("ministral", "Qwen minus Ministral")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, facecolor="white")
    im = None
    for ax, (other_key, title) in zip(axes, comparisons):
        other = all_data.get(other_key)
        if not other:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        try:
            other_cot = pt_idx(other, "cot")
        except ValueError:
            continue

        qwen_mean  = qwen["section_attn"][qwen_cot].mean(axis=0)   # [layers, sections]
        other_mean = other["section_attn"][other_cot].mean(axis=0)

        # Per-layer normalize to %
        q_sums = qwen_mean.sum(axis=1, keepdims=True);  q_sums[q_sums == 0] = 1
        o_sums = other_mean.sum(axis=1, keepdims=True); o_sums[o_sums == 0] = 1
        qwen_pct  = qwen_mean  / q_sums * 100
        other_pct = other_mean / o_sums * 100

        diff = qwen_pct - other_pct
        display = [SECTION_DISPLAY.get(s, (s,))[0] for s in qwen["section_names"]]

        vmax = max(abs(diff.min()), abs(diff.max()), 8)
        im = ax.imshow(diff.T, aspect="auto", cmap=diff_cmap,
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_yticks(range(len(display)))
        ax.set_yticklabels(display, fontsize=7)
        n_lay = diff.shape[0]
        step = max(1, n_lay // 6)
        ax.set_xticks(range(0, n_lay, step))
        ax.set_xticklabels(range(1, n_lay + 1, step), fontsize=7)

    if im is not None:
        fig.colorbar(im, ax=axes, label="Δ Attention % (green = Qwen higher)",
                     shrink=0.85, pad=0.02)
    fig.suptitle("Where Does Qwen Pay More Attention Than Others? (CoT)",
                 fontsize=10.5, fontweight="bold", y=1.02)

    for ext in ["pdf", "png"]:
        plt.savefig(OUT_DIR / f"attention_difference_map.{ext}", bbox_inches="tight", dpi=300)
    print("  Saved: attention_difference_map.{pdf,png}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6: MINISTRAL FAILURE STORY (ZS vs CoT) — the key story
# ─────────────────────────────────────────────────────────────────────────────

def plot_ministral_failure(all_data):
    """
    Two-panel figure focused on Ministral:
      Left:  doctrine attention across layers, ZS vs CoT
             -> shows ZS drops, CoT recovers
      Right: per-scenario decisions, ZS vs CoT
             -> shows ZS picks citizens everywhere, CoT recovers on pillar scenarios
    """
    data = all_data.get("ministral")
    if not data:
        return

    try:
        zs_idx  = pt_idx(data, "zeroshot")
        cot_idx = pt_idx(data, "cot")
    except ValueError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                   facecolor="white",
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # ── LEFT: doctrine attention by layer, ZS vs CoT ──
    zs_world  = data["world"][zs_idx].mean(axis=0)
    cot_world = data["world"][cot_idx].mean(axis=0)
    layers = np.arange(1, len(zs_world) + 1)

    ax1.plot(layers, zs_world, color="#C62828", linestyle="--",
             linewidth=1.8, label="ZeroShot", alpha=0.9)
    ax1.fill_between(layers, zs_world, alpha=0.15, color="#C62828")
    ax1.plot(layers, cot_world, color="#2E7D32", linestyle="-",
             linewidth=1.8, label="Chain-of-Thought", alpha=0.9)
    ax1.fill_between(layers, cot_world, alpha=0.15, color="#2E7D32")

    ax1.set_xlabel("Layer", fontsize=9)
    ax1.set_ylabel("Attention to Doctrine (<world>)", fontsize=9)
    ax1.set_title("Ministral: Doctrine Attention Through Layers",
                  fontsize=10, fontweight="bold", pad=6)
    ax1.legend(fontsize=8, loc="upper right", frameon=True, edgecolor="0.8")
    ax1.grid(axis="y", linestyle=":", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── RIGHT: ZS vs CoT decisions per scenario ──
    labels = data["labels"]
    n = len(labels)
    y = np.arange(n)
    bar_h = 0.38

    def target_int(t):
        return {"pillars": 1, "citizens": -1, "police": 0, "activists": 0,
                "unknown": 0, "parse_error": 0}.get(t, 0)

    zs_vals  = [target_int(get_decision(data, "zeroshot", i)["target"]) for i in range(n)]
    cot_vals = [target_int(get_decision(data, "cot",      i)["target"]) for i in range(n)]

    bars_zs  = ax2.barh(y + bar_h/2, zs_vals,  bar_h, label="ZeroShot",
                        color=["#C62828" if v < 0 else "#2E7D32" if v > 0 else "#9E9E9E"
                               for v in zs_vals],
                        edgecolor="white", linewidth=0.4, alpha=0.85)
    bars_cot = ax2.barh(y - bar_h/2, cot_vals, bar_h, label="CoT",
                        color=["#C62828" if v < 0 else "#2E7D32" if v > 0 else "#9E9E9E"
                               for v in cot_vals],
                        edgecolor="white", linewidth=0.4, alpha=0.85)

    short_labels = [l.split("_", 1)[-1].replace("_", " ") for l in labels]
    ax2.set_yticks(y)
    ax2.set_yticklabels(short_labels, fontsize=7)
    ax2.set_xticks([-1, 0, 1])
    ax2.set_xticklabels(["citizens", "other", "pillars"], fontsize=7.5)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.set_title("Ministral: Decisions per Scenario (ZS top, CoT bottom)",
                  fontsize=10, fontweight="bold", pad=6)
    ax2.set_xlim(-1.3, 1.3)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", linestyle=":", alpha=0.3)

    fig.suptitle("Ministral-ZS Failure Mode: Doctrine Lost in Deep Layers, Recovered by CoT",
                 fontsize=11, fontweight="bold", y=1.02)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_DIR / f"attention_ministral_failure.{ext}", bbox_inches="tight", dpi=300)
    print("  Saved: attention_ministral_failure.{pdf,png}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_decision_table(all_data):
    labels = None
    for m in MODELS:
        d = all_data.get(m["short"])
        if d:
            labels = d["labels"]
            break
    if not labels:
        return

    print("\n" + "=" * 110)
    print("  DECISION COMPARISON - same scenario, all models, ZS vs CoT")
    print("=" * 110)

    for i, label in enumerate(labels):
        short = label.split("_", 1)[-1].replace("_", " ").title()
        print(f"\n  {short}")
        print("  " + "-" * 80)
        for m in MODELS:
            data = all_data.get(m["short"])
            if not data:
                continue
            for pt in ["zeroshot", "cot"]:
                dec = get_decision(data, pt, i)
                if dec is None:
                    continue
                tag = f"{m['label']:<18} {pt.upper():<8}"
                target = dec["target"].upper()
                attr = dec.get("attribute", "")[:14]
                intent = dec.get("strategic_intent", "")[:60] or "—"
                print(f"  {tag} -> {target:<10} | {attr:<14} | {intent}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_data = {}
    for m in MODELS:
        data = load_data(m["short"])
        if data:
            all_data[m["short"]] = data
            print(f"  Loaded: {m['short']} -- prompt_types={data['prompt_types']}, "
                  f"scenarios={len(data['labels'])}, layers={data['n_layers']}")

    if not all_data:
        print("No .npz files found in attention_data/")
        print("Run extract_attention.py on the cluster first.")
        exit(1)

    plot_decision_matrix(all_data)
    plot_doctrine_survival(all_data)
    plot_section_heatmap(all_data)
    plot_per_scenario_attention(all_data)
    plot_attention_difference(all_data)
    plot_ministral_failure(all_data)
    print_decision_table(all_data)

    print(f"\n  All figures written to: {OUT_DIR}")
