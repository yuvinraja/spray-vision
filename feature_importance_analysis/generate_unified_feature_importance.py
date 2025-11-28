"""
Generate UNIFIED Feature Importance Plot (mean across all 4 targets)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Configuration - EXCLUDING TIME
FEATURES = ["chamb_pressure", "cham_temp", "injection_pres", "density", "viscosity"]
TARGETS = ["angle_mie", "angle_shadow", "length_mie", "length_shadow"]

FEATURE_LABELS = {
    "chamb_pressure": "Chamber Pressure (bar)",
    "cham_temp": "Chamber Temperature (K)",
    "injection_pres": "Injection Pressure (bar)",
    "density": "Density (kg/m³)",
    "viscosity": "Viscosity (Pa·s)",
}

# Styling constants
FONTSIZE = 14
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = FONTSIZE


def style_axes(ax):
    """Apply consistent styling to axes"""
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.8, color="gray")


def load_and_aggregate_importances():
    """Load feature importances and compute mean across all targets"""
    ALL_FEATURES = [
        "time",
        "chamb_pressure",
        "cham_temp",
        "injection_pres",
        "density",
        "viscosity",
    ]
    importance_data = []

    # Try loading separate models first
    separate_models_exist = True
    for target in TARGETS:
        model_path = f"models/GradientBoosting_{target}_regressor.joblib"
        if not Path(model_path).exists():
            separate_models_exist = False
            break

    if separate_models_exist:
        print("Using separate models...")
        for target in TARGETS:
            model_path = f"models/GradientBoosting_{target}_regressor.joblib"
            model = joblib.load(model_path)
            gb_estimator = model.named_steps["reg"]
            importances = gb_estimator.feature_importances_

            for feature, importance in zip(ALL_FEATURES, importances):
                if feature != "time":
                    importance_data.append(
                        {"Target": target, "Feature": feature, "Importance": importance}
                    )
    else:
        print("Using multi-output model...")
        model_path = "models/GradientBoosting_regressor.joblib"
        if not Path(model_path).exists():
            print(f"ERROR: No model file found at {model_path}")
            return None

        model = joblib.load(model_path)
        gb_multi = model.named_steps["reg"]

        for target_idx, target in enumerate(TARGETS):
            gb_estimator = gb_multi.estimators_[target_idx]
            importances = gb_estimator.feature_importances_

            for feature, importance in zip(ALL_FEATURES, importances):
                if feature != "time":
                    importance_data.append(
                        {"Target": target, "Feature": feature, "Importance": importance}
                    )

    df = pd.DataFrame(importance_data)

    # Normalize importances per target first
    for target in TARGETS:
        mask = df["Target"] == target
        total = df.loc[mask, "Importance"].sum()
        if total > 0:
            df.loc[mask, "Importance"] = df.loc[mask, "Importance"] / total

    # Compute mean and std across targets
    agg_df = df.groupby("Feature")["Importance"].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["Feature", "Mean_Importance", "Std_Importance"]

    return agg_df, df


def create_unified_plot(agg_df, output_dir="outputs"):
    """Create unified feature importance plot (mean across all targets)"""

    # Sort by mean importance
    agg_df = agg_df.sort_values("Mean_Importance", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors
    colors = sns.color_palette("tab10", len(agg_df))

    # Create horizontal bar plot with error bars
    bars = ax.barh(
        [FEATURE_LABELS[f] for f in agg_df["Feature"]],
        agg_df["Mean_Importance"],
        xerr=agg_df["Std_Importance"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
        capsize=5,
        error_kw={"linewidth": 2, "ecolor": "black"},
    )

    # Add value labels
    for bar, mean_val, std_val in zip(bars, agg_df["Mean_Importance"], agg_df["Std_Importance"]):
        ax.text(
            mean_val + std_val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.4f} ± {std_val:.4f}",
            va="center",
            ha="left",
            fontsize=FONTSIZE - 2,
            fontweight="normal",
        )

    # Styling
    ax.set_xlabel("Mean Importance Score (±Std)", fontsize=FONTSIZE, fontweight="bold")
    ax.set_ylabel("Input Feature", fontsize=FONTSIZE, fontweight="bold")
    ax.set_title(
        "Overall Feature Importance (Average Across All Targets)",
        fontsize=FONTSIZE + 1,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlim(0, max(agg_df["Mean_Importance"] + agg_df["Std_Importance"]) * 1.3)

    style_axes(ax)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / "feature_importance_unified_mean.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()

    return output_path


def create_comparison_plot(raw_df, agg_df, output_dir="outputs"):
    """Create comparison plot showing individual targets vs unified mean"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort features by unified mean
    feature_order = agg_df.sort_values("Mean_Importance", ascending=False)["Feature"].tolist()

    # Prepare data for grouped bar chart
    width = 0.15
    x = np.arange(len(feature_order))

    colors = {
        "angle_mie": "#1f77b4",
        "angle_shadow": "#ff7f0e",
        "length_mie": "#2ca02c",
        "length_shadow": "#d62728",
        "mean": "#9467bd",
    }

    # Plot bars for each target
    for i, target in enumerate(TARGETS):
        target_data = raw_df[raw_df["Target"] == target]
        target_data = target_data.set_index("Feature").reindex(feature_order)

        # Normalize
        target_data["Importance"] = target_data["Importance"] / target_data["Importance"].sum()

        ax.bar(
            x + i * width,
            target_data["Importance"],
            width,
            label=target.replace("_", " ").title(),
            color=colors[target],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

    # Plot mean as a line
    mean_data = agg_df.set_index("Feature").reindex(feature_order)
    ax.plot(
        x + 1.5 * width,
        mean_data["Mean_Importance"],
        color=colors["mean"],
        marker="D",
        markersize=10,
        linewidth=3,
        label="Overall Mean",
        zorder=5,
    )

    # Styling
    ax.set_xlabel("Input Feature", fontsize=FONTSIZE, fontweight="bold")
    ax.set_ylabel("Normalized Importance", fontsize=FONTSIZE, fontweight="bold")
    ax.set_title(
        "Feature Importance: Individual Targets vs. Unified Mean",
        fontsize=FONTSIZE + 1,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in feature_order], rotation=15, ha="right")
    ax.legend(fontsize=FONTSIZE - 2, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / "feature_importance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()

    return output_path


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("GENERATING UNIFIED FEATURE IMPORTANCE PLOT")
    print("=" * 80 + "\n")

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load and aggregate
    print("Loading feature importances and computing mean across targets...")
    result = load_and_aggregate_importances()

    if result is None:
        print("ERROR: Could not load feature importance data.")
        return

    agg_df, raw_df = result

    print(
        f"✓ Aggregated importances for {len(agg_df)} features across {raw_df['Target'].nunique()} targets\n"
    )

    # Print summary
    print("=" * 80)
    print("UNIFIED FEATURE IMPORTANCE RANKING (Mean ± Std)")
    print("=" * 80)
    agg_df_sorted = agg_df.sort_values("Mean_Importance", ascending=False)
    for idx, row in agg_df_sorted.iterrows():
        print(
            f"{FEATURE_LABELS[row['Feature']]:30s}  {row['Mean_Importance']:.4f} ± {row['Std_Importance']:.4f}"
        )
    print("=" * 80 + "\n")

    # Generate unified plot
    print("Generating unified feature importance plot...")
    create_unified_plot(agg_df, output_dir)

    print()

    # Generate comparison plot
    print("Generating comparison plot (individual targets vs. mean)...")
    create_comparison_plot(raw_df, agg_df, output_dir)

    # Save to CSV
    csv_path = Path(output_dir) / "feature_importance_unified.csv"
    agg_df_sorted.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"✓ Saved: {csv_path}")

    print("\n" + "=" * 80)
    print("COMPLETE! Unified feature importance plots saved to outputs/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
