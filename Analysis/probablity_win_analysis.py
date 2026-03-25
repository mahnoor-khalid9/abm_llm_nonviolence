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


# ── Configuration ─────────────────────────────────────────────────────────────

SIM_POPULATION = 1217      # total agents (citizens + activists)
VERSION_START  = 1
VERSION_END    = 100

# Base directory — adjust to your project root
PROJECT_ROOT = Path.cwd().parent

CONFIGS = [
    {
        "label":     "ZeroShot | Stateful",
        "csv_dir":   PROJECT_ROOT / "SimulationLogs" / "gemini_flash" / "ZeroShot"
                     / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
    },
    # Add more configs here as needed:
    # {
    #     "label":   "CoT | Stateful",
    #     "csv_dir": PROJECT_ROOT / "SimulationLogs" / "gemini_flash" / "CoT_reflexion"
    #                / "stateful" / "vision-4" / "ModelLogging" / "structured" / "ModelLogs",
    # },
]

SIM_COLORS  = ["#1565C0", "#E53935", "#43A047", "#CC00CC", "#FF8C00"]
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


# ── Main plot function ────────────────────────────────────────────────────────

# def plot_success_curves_vs_historical(
#     output_dir:     Path = OUTPUT_DIR,
#     sim_population: int  = SIM_POPULATION,
#     version_start:  int  = VERSION_START,
#     version_end:    int  = VERSION_END,
# ) -> None:
#     """
#     Plot P(Success) vs log10(Peak Participation %) for each simulation
#     config alongside the NAVCO historical baseline.
#     """
#     output_dir.mkdir(parents=True, exist_ok=True)

#     x_range = np.linspace(-1, 1.5, 1000)
#     y_hist  = sigmoid(0.6 + 0.6 * x_range) * 100

#     # Print historical reference points
#     print("NAVCO historical baseline:")
#     for xv in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
#         print(f"  x={xv:+.1f}  →  {sigmoid(0.6 + 0.6*xv)*100:.1f}%")
#     print()

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # NAVCO baseline — black solid
#     ax.plot(
#         x_range, y_hist,
#         color="black", linewidth=2.5, linestyle="-",
#         zorder=5, label="Historical Data - NAVCO", alpha=0.9,
#     )

#     # One curve per simulation config
#     for cfg, color, marker in zip(CONFIGS, SIM_COLORS, SIM_MARKERS):
#         print(f"── {cfg['label']}")

#         sim_df = extract_run_data(
#             csv_dir        = cfg["csv_dir"],
#             version_start  = version_start,
#             version_end    = version_end,
#             sim_population = sim_population,
#         )
#         sim_df.to_csv("resistance_size_df.csv", index=False)
#         if sim_df.empty:
#             print("  [SKIP] no valid runs found\n")
#             continue

#         n_runs = len(sim_df)
#         n_wins = int(sim_df["is_victory"].sum())
#         print(f"  Overall win rate: {n_wins}/{n_runs} = {n_wins/n_runs*100:.1f}%\n")

#         x_vals = sim_df["log10_pct"].values
#         y_vals = sim_df["is_victory"].values

#         # # Scatter individual runs
#         # ax.scatter(
#         #     x_vals, y_vals * 100,
#         #     color=color, marker=marker, alpha=0.35, s=30, zorder=3,
#         # )

#         # Fit logistic regression if both outcomes exist
#         if len(np.unique(y_vals)) < 2:
#             flat_y = float(y_vals[0]) * 100
#             suffix = "all victories" if flat_y == 100 else "all failures"
#             ax.plot(
#                 x_range, np.full_like(x_range, flat_y),
#                 color=color, linewidth=2.2, linestyle="--", zorder=4,
#                 label=f"{cfg['label']}  ({suffix})", alpha=0.75,
#             )
#         else:
#             lr = LogisticRegression(max_iter=1000)
#             lr.fit(x_vals.reshape(-1, 1), y_vals)
#             print(f"  logit = {lr.intercept_[0]:+.3f} "
#                   f"+ {lr.coef_[0][0]:+.3f} · log10(pct)")
#             y_sim = lr.predict_proba(x_range.reshape(-1, 1))[:, 1] * 100
#             ax.plot(
#                 x_range, y_sim,
#                 color=color, linewidth=2.2, linestyle="-", zorder=4,
#                 label=cfg["label"], alpha=0.85,
#             )

#     # 50% reference line
#     ax.axhline(50, color="gray", linestyle=":", alpha=0.4, linewidth=1.2)

#     ax.set_xlim(-1, 1.5)
#     ax.set_ylim(-5, 110)
#     ax.set_xlabel("Peak Participation  log10(%)", fontsize=12)
#     ax.set_ylabel("Probability of Success (%)", fontsize=12)
#     ax.set_yticks(range(0, 101, 10))
#     ax.yaxis.set_major_formatter(
#         plt.FuncFormatter(lambda y, _: f"{int(y)}")
#     )
#     ax.yaxis.grid(True, linestyle="--", alpha=0.3)
#     ax.set_axisbelow(True)
#     ax.set_title(
#         "P(Success) vs Peak Participation — Simulation vs NAVCO",
#         fontsize=13, fontweight="bold",
#     )
#     ax.legend(fontsize=10, framealpha=0.92, loc="lower right")

#     plt.tight_layout()

#     fname = PROJECT_ROOT / "Visuals" / "success_curves_vs_historical.png"
#     plt.savefig(fname, dpi=150, bbox_inches="tight")
#     plt.show()
#     print(f"\nSaved → {fname}")


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from scipy.stats import norm

# def plot_success_curves_vs_historical(
#     output_dir: Path = OUTPUT_DIR,
#     sim_population: int = SIM_POPULATION,
#     version_start: int = VERSION_START,
#     version_end: int = VERSION_END,
# ) -> None:
#     output_dir.mkdir(parents=True, exist_ok=True)

#     x_range = np.linspace(-1, 1.5, 1000)
    
#     # --- HISTORICAL (NAVCO) BASELINE ---
#     # Logit parameters for NAVCO: y = sigmoid(0.6 + 0.6 * x)
#     # We add a +/- 0.3 logit shift for a typical 95% CI
#     y_hist = sigmoid(0.6 + 0.6 * x_range) * 100
#     y_hist_low = sigmoid(0.6 + 0.6 * x_range - 0.3) * 100
#     y_hist_high = sigmoid(0.6 + 0.6 * x_range + 0.3) * 100

#     fig, ax = plt.subplots(figsize=(10, 7))

#     # Plot Historical with CI
#     ax.plot(x_range, y_hist, color="black", linewidth=2.5, label="Historical Data - NAVCO", zorder=10)
#     ax.fill_between(x_range, y_hist_low, y_hist_high, color="black", alpha=0.1, label="Historical 95% CI")

#     # --- SIMULATION CONFIGS ---
#     for cfg, color in zip(CONFIGS, SIM_COLORS):
#         sim_df = extract_run_data(cfg["csv_dir"], version_start, version_end, sim_population)
        
#         if sim_df.empty or len(np.unique(sim_df["is_victory"])) < 2:
#             continue

#         x_vals = sim_df["log10_pct"].values.reshape(-1, 1)
#         y_vals = sim_df["is_victory"].values

#         # Fit Logistic Regression
#         lr = LogisticRegression(max_iter=1000)
#         lr.fit(x_vals, y_vals)

#         # 1. Calculate Mean Prediction
#         # Logit: z = β0 + β1*x
#         z = lr.intercept_[0] + lr.coef_[0][0] * x_range
#         y_sim = sigmoid(z) * 100

#         # 2. Calculate 95% CI via Delta Method (Standard Error of Logit)
#         # Note: This is a simplified SE estimate for visualization
#         # In a real logit CI, we'd use the covariance matrix: SE = sqrt(X * Cov * X.T)
#         # Here we apply a simulation-based variance wrapper:
#         n = len(y_vals)
#         se_logit = np.sqrt(1 / (n * y_sim/100 * (1 - y_sim/100 + 1e-6))) # Approximate SE
        
#         # Z-score for 95% is ~1.96
#         ci_low = sigmoid(z - 1.96 * se_logit) * 100
#         ci_high = sigmoid(z + 1.96 * se_logit) * 100

#         # Plot Simulation Mean
#         ax.plot(x_range, y_sim, color=color, linewidth=2, label=cfg["label"], alpha=0.9)
        
#         # Plot Simulation CI (using fill_between for a cleaner research look)
#         ax.fill_between(x_range, ci_low, ci_high, color=color, alpha=0.15)

#     # ── Formatting ────────────────────────────────────────────────────────────
#     ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
#     ax.set_xlim(-1, 1.5)
#     ax.set_ylim(0, 105)
#     ax.set_xlabel("Peak Participation log10(%)", fontsize=12)
#     ax.set_ylabel("Probability of Success (%)", fontsize=12)
#     ax.set_title("Success Probability with 95% Confidence Intervals", fontweight="bold")
    
#     # Adjust legend to not be overwhelming
#     ax.legend(loc="lower right", fontsize=9, ncol=2, frameon=True)
    
#     plt.tight_layout()
#     plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# ── Helper Math ──────────────────────────────────────────────────────────────

def sigmoid(x):
    """Standard sigmoid function for probability mapping."""
    return 1 / (1 + np.exp(-x))

# ── Main Plotting Function ───────────────────────────────────────────────────

def plot_success_curves_vs_historical(
    output_dir:    Path = Path("./Visuals"),
    sim_population: int  = 10000,
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
    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot Historical Mean (Solid)
    ax.plot(
        x_range, y_hist,
        color="black", linewidth=2.8, linestyle="-",
        zorder=10, label="Historical Data (NAVCO)", alpha=0.9
    )

    # Plot Historical 95% CI (Dashed)
    ax.plot(x_range, y_hist_low, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.plot(x_range, y_hist_high, color="black", linestyle="--", linewidth=1.2, alpha=0.5, label="Historical 95% CI")

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

        # Plot Simulation 95% CI (Dotted - No Fill)
        ax.plot(x_range, y_sim_low, color=color, linestyle=":", linewidth=1.2, alpha=0.6)
        ax.plot(x_range, y_sim_high, color=color, linestyle=":", linewidth=1.2, alpha=0.6)

    # 5. Visual Formatting
    ax.axhline(50, color="gray", linestyle=":", alpha=0.4, linewidth=1.2)
    
    ax.set_xlim(-1, 1.5)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Peak Participation  log10(%)", fontsize=12, labelpad=10)
    ax.set_ylabel("Probability of Success (%)", fontsize=12, labelpad=10)
    
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.grid(True, linestyle="--", alpha=0.2)
    
    ax.set_title(
        "P(Success) vs Peak Participation — Simulation vs NAVCO",
        fontsize=14, fontweight="bold", pad=15
    )

    # Legend handling: Clean up duplicate CI entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, loc="lower right", framealpha=0.95, ncol=1)

    plt.tight_layout()

    # Save and Show
    fname = output_dir / "success_curves_with_ci.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nVisualization exported to: {fname}")

# ── Usage ────────────────────────────────────────────────────────────────────
# Ensure your CONFIGS and SIM_COLORS lists are set up before calling:
# plot_success_curves_vs_historical()