"""
plot_success_curves_vs_historical.py

Reads structured CSV model logs directly — no parsing needed.

CSV schema:
  step, elapsed_s, legitimacy, nvCitizen, protestingCitizen,
  protestingActivists, resistanceSize, campSize, jailedActivists,
  activists, police, pillars, victory, defected_count,
  defected_institutions, GrievanceLow, GrievanceMedium, GrievanceHigh,
  q33, q66

One CSV per run (model_state_v1.csv … model_state_v100.csv).
Peak participation = max(resistanceSize) across all steps in that run.
Victory = 1 if any row has victory == 2 (resistance wins).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ── Color Palette (matches samples exactly) ───────────────────────────────────
COLORS = {
    "low":    "#1f77b4",   # Blue
    "middle": "#ff7f0e",   # Orange
    "high":   "#2ca02c",   # Green
}

# ── Line Style Convention (Data = solid, Model/CI = dashed) ──────────────────
LINESTYLES = {
    "data":  "-",          # Solid  — observed/historical
    "model": "--",         # Dashed — simulated/model
    "ci":    ":",          # Dotted — confidence intervals
}

# ── Configuration ─────────────────────────────────────────────────────────────

SIM_POPULATION = 1217      # total agents (citizens + activists)
VERSION_START  = 1
VERSION_END    = 100

# Base directory — adjust to your project root
PROJECT_ROOT = Path.cwd().parent
PROVIDER = "vllm"
CONFIGS = [
    {
        "label":     "Gemini-2.5-Flash | ZeroShot | Internal Reasoning",
        "csv_dir":   PROJECT_ROOT / "SimulationLogs" / "gemini-2.5-flash" / "ZeroShot"
                     / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
    },
    {
        "label":   "Llama3.1-8B-Instruct | ZeroShot",
        "csv_dir": PROJECT_ROOT / "SimulationLogs" / "llama3.1-8B-Instruct_ZeroShot" / "ZeroShot"
                   / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
    },
    {
        "label":   "Llama3.1-8B-Instruct | CoT",
        "csv_dir": PROJECT_ROOT / "SimulationLogs" / "llama3.1-8B-Instruct_CoT" / "CoT"
                   / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
    },
    # {
    #     "label":   "Llama3.1-70B-Instruct_AWQ | ZeroShot",
    #     "csv_dir": PROJECT_ROOT / "SimulationLogs" / "llama-3.1-70B-Instruct-AWQ-INT4_ZeroShot" / "ZeroShot"
    #                / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
    # }, 
    {
        "label":   "Qwen3-8B | ZeroShot | Internal Reasoning",
        "csv_dir": PROJECT_ROOT / "SimulationLogs" / "qwen3-8B_ZeroShot" / "ZeroShot"
                   / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
    },
    # {
    #     "label":   "Qwen2.5-32B-Instruct-AWQ | ZeroShot",
    #     "csv_dir": PROJECT_ROOT / "SimulationLogs" / "Qwen2.5-32B-Instruct-AWQ_ZeroShot" / "ZeroShot"
    #                / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
    # },
]

SIM_COLORS  = ["#1565C0", "#E53935", "#43A047", "#CC00CC", "#FF8C00","#008080"]
SIM_MARKERS = ["o", "s", "^", "D", "v"]

OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "plots"


# ── NAVCO historical baseline ─────────────────────────────────────────────────
# sigmoid(0.6 + 0.6 * x) matches the Chenoweth reference curve:
#   x=-1  → 50%,  x=0  → 65%,  x=1  → 77%,  x=1.5 → 83%

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


# ── Per-run data extraction ───────────────────────────────────────────────────

def extract_run_data(
    csv_dir: Path,
    version_start: int = VERSION_START,
    version_end:   int = VERSION_END,
    sim_population: int = SIM_POPULATION,
) -> pd.DataFrame:
    """
    Read all run CSVs from csv_dir, return DataFrame with one row per run:
        log10_pct   : log10(peak_resistance / sim_population * 100)
        is_victory  : 1 if resistance won (victory == 2), else 0
    """
    rows = []

    for v in range(version_start, version_end + 1):
        csv_path = csv_dir / f"model_state_v{v}.csv"
        if not csv_path.exists():
            print(f"  [WARN] Missing file: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  [WARN] Could not read {csv_path.name}: {exc}")
            continue

        if df.empty:
            continue

        # Peak resistance size across all steps
        if "resistanceSize" in df.columns:
            max_resistance = df["resistanceSize"].max()
        else:
            # Fallback: sum of protesting agents
            max_resistance = (
                df.get("protestingCitizen",   pd.Series([0])).fillna(0).max()
                + df.get("protestingActivists", pd.Series([0])).fillna(0).max()
            )

        if max_resistance <= 0:
            print(f"  [SKIP] v{v}: resistanceSize = 0")
            continue

        pct       = max_resistance / sim_population * 100
        log10_pct = np.log10(pct)

        # Victory: resistance wins if any step has victory == 2
        is_victory = 0
        if "victory" in df.columns:
            is_victory = df["victory"].max() == 2

        rows.append({
            "version":    v,
            "max_res":    int(max_resistance),
            "pct":        round(pct, 4),
            "log10_pct":  round(log10_pct, 4),
            "is_victory": is_victory,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        print(f"  Loaded {len(result)} runs  |  "
              f"victories: {result['is_victory'].sum()}  |  "
              f"log10_pct range: [{result['log10_pct'].min():.2f}, "
              f"{result['log10_pct'].max():.2f}]")
    return result


# ── Helper Math ──────────────────────────────────────────────────────────────
def sigmoid(x):
    """Standard sigmoid function for probability mapping."""
    return 1 / (1 + np.exp(-x))

# ── Main Plotting Function ───────────────────────────────────────────────────

def plot_success_curves_vs_historical(
    output_dir:    Path = Path("./Visuals"),
    sim_population: int  = SIM_POPULATION,
    version_start:  int  = 0,
    version_end:    int  = 100,
) -> None:
    """
    Plots P(Success) vs log10(Peak Participation %) with 95% Confidence Intervals
    as dotted lines for both Historical and Simulated data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup X-range (log10 scale of participation %)
    x_range = np.linspace(-1, 1.5, 1000)
    x_range_reshaped = x_range.reshape(-1, 1)

    # 2. Historical NAVCO Baseline Calculation
    # Baseline: logit = 0.6 + 0.6 * x
    z_hist = 0.6 + 0.6 * x_range
    y_hist = sigmoid(z_hist) * 100
    
    # 95% CI for Historical (Dashed/Dotted Bounds)
    # We use a standard logit-space shift (~0.35) to represent historical variance
    y_hist_low = sigmoid(z_hist - 0.35) * 100
    y_hist_high = sigmoid(z_hist + 0.35) * 100

    # 3. Initialize Figure
    fig, ax = plt.subplots()  # ✅ remove figsize override

    # Plot Historical Mean (Solid)
    ax.plot(
        x_range, y_hist,
        color="black", linewidth=2.8, linestyle="-",
        zorder=10, label="Historical Data (NAVCO)", alpha=0.9
    )

    # Plot Historical 95% CI (Dashed)
    # ax.plot(x_range, y_hist_low, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    # ax.plot(x_range, y_hist_high, color="black", linestyle="--", linewidth=1.2, alpha=0.5, label="Historical 95% CI")

    # 4. Loop through Simulation Configurations
    # Assumes CONFIGS, SIM_COLORS, etc., are defined in your global scope
    for cfg, color in zip(CONFIGS, SIM_COLORS):
        # Data Extraction (Placeholder for your specific extraction logic)
        sim_df = extract_run_data(
            csv_dir=cfg["csv_dir"],
            version_start=version_start,
            version_end=version_end,
            sim_population=sim_population,
        )

        sim_df.to_csv("resistance_size_df.csv", index=False)
        
        if sim_df.empty or len(np.unique(sim_df["is_victory"])) < 2:
            print(f"  [SKIP] {cfg['label']}: Insufficient variance for logistic fit.")
            continue

        x_vals = sim_df["log10_pct"].values.reshape(-1, 1)
        y_vals = sim_df["is_victory"].values
        n_samples = len(y_vals)

        # Fit Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x_vals, y_vals)

        # Calculate Mean Probability
        # Predict_proba returns [P(0), P(1)], we take index 1
        y_sim = lr.predict_proba(x_range_reshaped)[:, 1] * 100
        
        # Calculate 95% CI using the Delta Method approximation
        # SE(logit) ≈ 1 / sqrt(n * p * (1-p))
        # For visualization, we use a conservative z-score of 1.96
        z_sim = lr.intercept_[0] + lr.coef_[0][0] * x_range
        se_logit = 1.96 * (1.0 / np.sqrt(n_samples * 0.25)) # Conservative logit SE
        
        y_sim_low = sigmoid(z_sim - se_logit) * 100
        y_sim_high = sigmoid(z_sim + se_logit) * 100

        # Plot Simulation Mean (Solid)
        ax.plot(
            x_range, y_sim,
            color=color, linewidth=2.2, linestyle="-",
            zorder=5, label=cfg["label"], alpha=0.85
        )

        # Plot Simulation 95% CI (Dotted - No Fill - Simulations)
        # ax.plot(x_range, y_sim_low, color=color, linestyle="--", linewidth=1.5, alpha=0.6)
        # ax.plot(x_range, y_sim_high, color=color, linestyle="--", linewidth=1.5, alpha=0.6)

    # 5. Visual Formatting
    ax.axhline(50, color="gray", linestyle=":", alpha=0.4, linewidth=1.2)
    
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-2, 105)

    # ax.set_xlabel("Peak Participation  log10(%)")
    # ax.set_ylabel("Probability of Success (%)")
    # ax.set_title("P(Success) vs Peak Participation — Simulation vs NAVCO")


    ax.set_xlabel("Peak Participation  log10(%)", labelpad=6)
    ax.set_ylabel("Probability of Success (%)", labelpad=6)
    
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.grid(True, linestyle="--", alpha=0.2)
    
    # ax.set_title(
        # "P(Success) vs Peak Participation — Simulation vs NAVCO",
        # fontsize=9, fontweight="bold", pad=10
    # )

    # Legend handling: Clean up duplicate CI entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", framealpha=0.95, ncol=1)



    # ── Save ─────────────────────────────────────────────────────────────────────
    # PDF is vector (DPI doesn't apply to vectors, but set high for any raster elements)
    # Also export TIFF @ 1200 dpi as Springer typesetters often request it
    fname_pdf  = output_dir / "success_curves_with_ci.pdf"
    fname_tiff = output_dir / "success_curves_with_ci.tif"

    plt.savefig(fname_pdf,  dpi=1200, bbox_inches="tight")   # was dpi=300
    plt.savefig(fname_tiff, dpi=1200, bbox_inches="tight",
                format="tiff", pil_kwargs={"compression": "tiff_lzw"})

    # plt.tight_layout()

    # # Save and Show
    # fname = output_dir / "success_curves_with_ci.pdf"
    # plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    # print(f"\nVisualization exported to: {fname}")

# ── Usage ────────────────────────────────────────────────────────────────────
# Ensure your CONFIGS and SIM_COLORS lists are set up before calling:
# plot_success_curves_vs_historical()



#  ========================================================================================= #
"""
table_success_summary.py
========================
Extracts per-model summary statistics from the logistic success-curve fits
and exports a publication-ready LaTeX table.

Columns produced
----------------
  Model                : display name
  N Runs               : number of runs loaded
  Victories            : count of resistance wins
  Win Rate (%)         : victories / N * 100
  P(success) min (%)   : lowest fitted probability across x_range, with 95% CI
  P(success) max (%)   : highest fitted probability across x_range, with 95% CI

Usage
-----
    python table_success_summary.py

Outputs
-------
    table_success_summary.tex   — drop straight into your LaTeX document
    table_success_summary.csv   — for inspection / supplementary material
"""



# =============================================================================
# SHARED SETTINGS  (keep in sync with your main plotting script)
# =============================================================================

SIM_POPULATION = 1217
VERSION_START  = 1
VERSION_END    = 100

# PROJECT_ROOT = Path.cwd().parent

# CONFIGS = [
#     {
#         "label":     "Gemini-2.5-Flash | ZeroShot | Internal Reasoning",
#         "csv_dir":   PROJECT_ROOT / "SimulationLogs" / "gemini-2.5-flash" / "ZeroShot"
#                      / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
#     },
#     {
#         "label":   "Llama3.1-8B-Instruct | ZeroShot",
#         "csv_dir": PROJECT_ROOT / "SimulationLogs" / "llama3.1-8B-Instruct_ZeroShot" / "ZeroShot"
#                    / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
#     },
#     {
#         "label":   "Llama3.1-8B-Instruct | CoT",
#         "csv_dir": PROJECT_ROOT / "SimulationLogs" / "llama3.1-8B-Instruct_CoT" / "CoT"
#                    / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
#     },
#     {
#         "label":   "Llama3.1-70B-Instruct-AWQ | ZeroShot",
#         "csv_dir": PROJECT_ROOT / "SimulationLogs" / "llama-3.1-70B-Instruct-AWQ-INT4_ZeroShot" / "ZeroShot"
#                    / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
#     },    {
#         "label":   "Qwen3-8B | ZeroShot | Internal Reasoning",
#         "csv_dir": PROJECT_ROOT / "SimulationLogs" / "qwen3-8B_ZeroShot_0-85" / "ZeroShot"
#                    / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
#     },
#     {
#         "label":   "Qwen2.5-32B-Instruct-AWQ | ZeroShot",
#         "csv_dir": PROJECT_ROOT / "SimulationLogs" / "Qwen2.5-32B-Instruct-AWQ_ZeroShot" / "ZeroShot"
#                    / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
#     },
# ]


X_RANGE = np.linspace(-1, 1.5, 1000)   # same as plotting script

# =============================================================================
# HELPERS (copied from plotting script — keep identical)
# =============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def extract_run_data(
    csv_dir: Path,
    version_start: int = VERSION_START,
    version_end:   int = VERSION_END,
    sim_population: int = SIM_POPULATION,
) -> pd.DataFrame:
    rows = []
    for v in range(version_start, version_end + 1):
        csv_path = csv_dir / f"model_state_v{v}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue

        if "resistanceSize" in df.columns:
            max_res = df["resistanceSize"].max()
        else:
            max_res = (
                df.get("protestingCitizen",   pd.Series([0])).fillna(0).max()
                + df.get("protestingActivists", pd.Series([0])).fillna(0).max()
            )

        if max_res <= 0:
            continue

        pct      = max_res / sim_population * 100
        is_win   = int(df["victory"].max() == 2) if "victory" in df.columns else 0
        rows.append({"version": v, "log10_pct": np.log10(pct), "is_victory": is_win})

    return pd.DataFrame(rows)


def fit_and_summarise(cfg: dict) -> dict | None:
    """
    Fit logistic regression, return a dict of summary statistics.
    Returns None if the config cannot be fitted (missing data / no variance).
    """
    sim_df = extract_run_data(cfg["csv_dir"])

    n      = len(sim_df)
    n_wins = int(sim_df["is_victory"].sum()) if not sim_df.empty else 0

    if sim_df.empty or len(np.unique(sim_df["is_victory"])) < 2:
        print(f"  [SKIP] {cfg['label']}: not enough variance.")
        return {
            "Model":             cfg["label"],
            "N Runs":            n,
            "Victories":         n_wins,
            "Win Rate (%)":      round(n_wins / n * 100, 1) if n > 0 else float("nan"),
            "P(min) (%)":        float("nan"),
            "P(min) 95% CI":     "---",
            "P(max) (%)":        float("nan"),
            "P(max) 95% CI":     "---",
        }

    x_vals = sim_df["log10_pct"].values.reshape(-1, 1)
    y_vals = sim_df["is_victory"].values

    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_vals, y_vals)

    # Logit scores over x_range
    z     = lr.intercept_[0] + lr.coef_[0][0] * X_RANGE
    y_fit = sigmoid(z) * 100

    # Conservative 95% CI using delta-method SE in logit space
    se    = 1.96 / np.sqrt(n * 0.25)
    y_lo  = sigmoid(z - se) * 100
    y_hi  = sigmoid(z + se) * 100

    # Min and max of fitted curve
    idx_min = int(np.argmin(y_fit))
    idx_max = int(np.argmax(y_fit))

    p_min     = round(y_fit[idx_min], 1)
    p_min_lo  = round(y_lo[idx_min],  1)
    p_min_hi  = round(y_hi[idx_min],  1)

    p_max     = round(y_fit[idx_max], 1)
    p_max_lo  = round(y_lo[idx_max],  1)
    p_max_hi  = round(y_hi[idx_max],  1)

    return {
        "Model":             cfg["label"],
        "N Runs":            n,
        "Victories":         n_wins,
        "Win Rate (%)":      round(n_wins / n * 100, 1),
        "P(min) (%)":        p_min,
        "P(min) 95% CI":     f"[{p_min_lo}, {p_min_hi}]",
        "P(max) (%)":        p_max,
        "P(max) 95% CI":     f"[{p_max_lo}, {p_max_hi}]",
    }


# =============================================================================
# HISTORICAL NAVCO BASELINE ROW
# =============================================================================

def navco_row() -> dict:
    z      = 0.6 + 0.6 * X_RANGE
    y_hist = sigmoid(z) * 100
    y_lo   = sigmoid(z - 0.35) * 100
    y_hi   = sigmoid(z + 0.35) * 100

    idx_min = int(np.argmin(y_hist))
    idx_max = int(np.argmax(y_hist))

    return {
        "Model":          "Historical (NAVCO)",
        "N Runs":         "---",
        "Victories":      "---",
        "Win Rate (%)":   "---",
        "P(min) (%)":     round(y_hist[idx_min], 1),
        "P(min) 95% CI":  f"[{round(y_lo[idx_min],1)}, {round(y_hi[idx_min],1)}]",
        "P(max) (%)":     round(y_hist[idx_max], 1),
        "P(max) 95% CI":  f"[{round(y_lo[idx_max],1)}, {round(y_hi[idx_max],1)}]",
    }


# =============================================================================
# BUILD DATAFRAME
# =============================================================================

def plot_success_curves_vs_historical_table():
    rows = [navco_row()]
    for cfg in CONFIGS:
        print(f"Processing: {cfg['label']}")
        row = fit_and_summarise(cfg)
        if row:
            rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Pretty-print to console
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       200)
    pd.set_option("display.max_colwidth", 40)
    print("\n" + "=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)

    # =============================================================================
    # EXPORT CSV
    # =============================================================================

    summary_df.to_csv("table_success_summary.csv", index=False)
    print("\n  Saved: table_success_summary.csv")

    # =============================================================================
    # EXPORT LaTeX
    # =============================================================================

    # Merge the two CI columns into combined cells for a cleaner table:
    #   P(min) %  [lo, hi]   →  one display column "P(success) at low participation"
    # We build the LaTeX manually for full control over multicolumn headers.

    def fmt_ci_cell(pct, ci):
        """Format a value+CI pair for a LaTeX cell."""
        if str(pct) == "nan":
            return r"\multicolumn{1}{c}{---}"
        return rf"{pct}\% \scriptsize{{[{ci.strip('[]')}]}}"


    lines = []
    lines.append(r"% ---------------------------------------------------------------")
    lines.append(r"% Auto-generated by table_success_summary.py")
    lines.append(r"% Requires: booktabs, multirow, siunitx (optional)")
    lines.append(r"% ---------------------------------------------------------------")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \renewcommand{\arraystretch}{1.25}")
    lines.append(
        r"  \begin{tabular}{l r r r l l}"
    )
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Model} & \textbf{$N$} & \textbf{Wins} & \textbf{Win\,\%}"
        r" & \textbf{$\hat{P}_{\min}$ [\,95\%\,CI\,]}"
        r" & \textbf{$\hat{P}_{\max}$ [\,95\%\,CI\,]} \\"
    )
    lines.append(r"    \midrule")

    for _, row in summary_df.iterrows():
        model    = row["Model"].replace("|", r"\textbar{}")
        n        = row["N Runs"]
        wins     = row["Victories"]
        win_rate = row["Win Rate (%)"]

        # P(min) cell
        if str(row["P(min) (%)"]) == "nan":
            p_min_cell = "---"
            p_max_cell = "---"
        else:
            p_min_ci   = row["P(min) 95% CI"].strip("[]").replace(",", ",\,")
            p_max_ci   = row["P(max) 95% CI"].strip("[]").replace(",", ",\,")
            p_min_cell = rf"{row['P(min) (%)']}\% {{\scriptsize [{p_min_ci}]}}"
            p_max_cell = rf"{row['P(max) (%)']}\% {{\scriptsize [{p_max_ci}]}}"

        # Format N/Wins/WinRate for non-data rows (NAVCO)
        n_str    = "---" if str(n)        == "---" else str(n)
        wins_str = "---" if str(wins)     == "---" else str(wins)
        wr_str   = "---" if str(win_rate) == "---" else rf"{win_rate}\%"

        lines.append(
            rf"    {model} & {n_str} & {wins_str} & {wr_str}"
            rf" & {p_min_cell} & {p_max_cell} \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Summary of logistic success-curve fits across LLM configurations")
    lines.append(r"    and the NAVCO historical baseline.")
    lines.append(r"    $N$: number of simulation runs.")
    lines.append(r"    Wins: runs where resistance succeeded (victory\,=\,2).")
    lines.append(r"    $\hat{P}_{\min}$ and $\hat{P}_{\max}$: lowest and highest fitted")
    lines.append(r"    probability of success across the participation range")
    lines.append(r"    $\log_{10}(\%) \in [-1,\, 1.5]$,")
    lines.append(r"    with 95\% confidence intervals derived from the delta method.}")
    lines.append(r"  \label{tab:success_summary}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    with open("table_success_summary.tex", "w", encoding="utf-8") as fh:
        fh.write(latex_str)

    print("\n  Saved: table_success_summary.tex")
    print("\n--- LaTeX preview ---\n")
    print(latex_str)


# =============================================================================
# ERROR METRICS vs NAVCO BASELINE
# =============================================================================

def compute_navco_baseline(x_range: np.ndarray) -> np.ndarray:
    """Returns NAVCO fitted P(success) curve over x_range."""
    z = 0.6 + 0.6 * x_range
    return sigmoid(z) * 100


def compute_model_curve(cfg: dict, x_range: np.ndarray) -> np.ndarray | None:
    """
    Fits logistic regression for a config and returns P(success) curve.
    Returns None if insufficient data.
    """
    sim_df = extract_run_data(cfg["csv_dir"])

    if sim_df.empty or len(np.unique(sim_df["is_victory"])) < 2:
        return None

    x_vals = sim_df["log10_pct"].values.reshape(-1, 1)
    y_vals = sim_df["is_victory"].values

    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_vals, y_vals)

    z = lr.intercept_[0] + lr.coef_[0][0] * x_range
    return sigmoid(z) * 100


def compute_error_metrics(
    y_model: np.ndarray,
    y_navco: np.ndarray,
    x_range: np.ndarray,
) -> dict:
    """
    Computes error metrics between a model curve and NAVCO baseline.

    Parameters
    ----------
    y_model : fitted P(success) for the model (%)
    y_navco : fitted P(success) for NAVCO baseline (%)
    x_range : shared x-axis (log10 participation %)
    """
    diff    = y_model - y_navco
    abs_diff = np.abs(diff)

    mae  = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(abs_diff))
    max_err_x = float(x_range[np.argmax(abs_diff)])

    # Area between curves (AUC diff) via trapezoidal integration
    # Positive = model overestimates NAVCO, negative = underestimates
    x_span = x_range[-1] - x_range[0]
    auc_diff = float(np.trapz(diff, x_range) / x_span)  # normalized by x range

    # Direction: does the model over or under-estimate NAVCO on average?
    bias = float(np.mean(diff))

    return {
        "MAE (%)":         round(mae,      2),
        "RMSE (%)":        round(rmse,     2),
        "Max Error (%)":   round(max_err,  2),
        "Max Error at x":  round(max_err_x, 3),
        "AUC Diff (%)":    round(auc_diff, 2),   # +ve = overestimates NAVCO
        "Bias (%)":        round(bias,     2),    # +ve = model predicts higher success
    }


def plot_error_metrics_table(x_range: np.ndarray = X_RANGE) -> pd.DataFrame:
    """
    Computes and prints error metrics for all CONFIGS vs NAVCO baseline.
    Also exports CSV and LaTeX.
    """
    y_navco = compute_navco_baseline(x_range)

    rows = []
    for cfg in CONFIGS:
        print(f"Computing errors: {cfg['label']}")
        y_model = compute_model_curve(cfg, x_range)

        if y_model is None:
            print(f"  [SKIP] {cfg['label']}: insufficient data.")
            rows.append({
                "Model":          cfg["label"],
                "MAE (%)":        float("nan"),
                "RMSE (%)":       float("nan"),
                "Max Error (%)":  float("nan"),
                "Max Error at x": float("nan"),
                "AUC Diff (%)":   float("nan"),
                "Bias (%)":       float("nan"),
            })
            continue

        metrics = compute_error_metrics(y_model, y_navco, x_range)
        rows.append({"Model": cfg["label"], **metrics})

    error_df = pd.DataFrame(rows)

    # ── Print ─────────────────────────────────────────────────────────────────
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("ERROR METRICS vs NAVCO HISTORICAL BASELINE")
    print("=" * 100)
    print(error_df.to_string(index=False))
    print("=" * 100)
    print("\nNote:")
    print("  MAE / RMSE  — lower is better (closer to NAVCO)")
    print("  Bias > 0    — model overestimates success probability vs NAVCO")
    print("  Bias < 0    — model underestimates success probability vs NAVCO")
    print("  AUC Diff    — signed area between curves (normalized by x range)")

    # ── CSV ───────────────────────────────────────────────────────────────────
    error_df.to_csv("table_error_metrics.csv", index=False)
    print("\n  Saved: table_error_metrics.csv")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    lines = []
    lines.append(r"% Auto-generated error metrics table")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \renewcommand{\arraystretch}{1.25}")
    lines.append(r"  \begin{tabular}{l r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Model} & \textbf{MAE} & \textbf{RMSE}"
        r" & \textbf{Max Err.} & \textbf{at $x$}"
        r" & \textbf{AUC Diff} & \textbf{Bias} \\"
    )
    lines.append(r"    \midrule")

    for _, row in error_df.iterrows():
        model = row["Model"].replace("|", r"\textbar{}")
        def fmt(v): return "---" if str(v) == "nan" else f"{v:.2f}"
        lines.append(
            rf"    {model} & {fmt(row['MAE (%)'])}\% & {fmt(row['RMSE (%)'])}\%"
            rf" & {fmt(row['Max Error (%)'])}\% & {fmt(row['Max Error at x'])}"
            rf" & {fmt(row['AUC Diff (%)'])}\% & {fmt(row['Bias (%)'])}\% \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(
        r"  \caption{Error metrics for each LLM configuration relative to the"
        r"  NAVCO historical baseline. MAE and RMSE are computed pointwise"
        r"  across $\log_{10}(\%) \in [-1,\,1.5]$."
        r"  Positive bias indicates the model overestimates P(success) vs.\ NAVCO.}"
    )
    lines.append(r"  \label{tab:error_metrics}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    with open("table_error_metrics.tex", "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    print("  Saved: table_error_metrics.tex")

    return error_df


# ── Also plot the error curves visually ──────────────────────────────────────

def plot_error_curves(
    x_range: np.ndarray = X_RANGE,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Plots the absolute deviation of each model from NAVCO across participation range.
    Useful for seeing WHERE each model diverges most.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    y_navco = compute_navco_baseline(x_range)

    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)

    for cfg, color in zip(CONFIGS, SIM_COLORS):
        y_model = compute_model_curve(cfg, x_range)
        if y_model is None:
            continue
        abs_err = np.abs(y_model - y_navco)
        ax.plot(x_range, abs_err, color=color, linewidth=2, label=cfg["label"])

    ax.set_xlabel("Peak Participation log10(%)", labelpad=6)
    ax.set_ylabel("Absolute Deviation from NAVCO (%)", labelpad=6)
    ax.set_title("Pointwise Absolute Error vs NAVCO Baseline")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2)

    fname_pdf  = output_dir / "error_curves.pdf"
    fname_tiff = output_dir / "error_curves.tif"
    plt.savefig(fname_pdf,  dpi=1200, bbox_inches="tight")
    plt.savefig(fname_tiff, dpi=1200, bbox_inches="tight",
                format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.show()
    print(f"  Saved: {fname_pdf}")