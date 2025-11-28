"""
Generate Feature Importance Plots (EXCLUDING TIME) matching journal figure style
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

TARGET_TITLES = {
    "angle_mie": "Spray Angle (Mie)",
    "length_mie": "Spray Length (Mie)",
    "angle_shadow": "Spray Angle (Shadow)",
    "length_shadow": "Spray Length (Shadow)",
}

# Styling constants (matching reference image)
FONTSIZE = 14
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = FONTSIZE


def style_axes(ax):
    """Apply consistent styling to axes matching reference plot"""
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=6)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.8, color="gray")


def load_feature_importances():
    """Load feature importances from models (tries separate models first, then multi-output)"""
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
                if feature != "time":  # EXCLUDE TIME
                    importance_data.append(
                        {"Target": target, "Feature": feature, "Importance": importance}
                    )
    else:
        # Fall back to multi-output model
        print("Using multi-output model...")
        model_path = "models/GradientBoosting_regressor.joblib"
        if not Path(model_path).exists():
            print(f"ERROR: No model file found at {model_path}")
            return pd.DataFrame()

        model = joblib.load(model_path)
        gb_multi = model.named_steps["reg"]

        for target_idx, target in enumerate(TARGETS):
            gb_estimator = gb_multi.estimators_[target_idx]
            importances = gb_estimator.feature_importances_

            for feature, importance in zip(ALL_FEATURES, importances):
                if feature != "time":  # EXCLUDE TIME
                    importance_data.append(
                        {"Target": target, "Feature": feature, "Importance": importance}
                    )

    # Normalize importances to sum to 1.0 (since we removed time)
    df = pd.DataFrame(importance_data)
    for target in TARGETS:
        mask = df["Target"] == target
        total = df.loc[mask, "Importance"].sum()
        if total > 0:
            df.loc[mask, "Importance"] = df.loc[mask, "Importance"] / total

    return df


def create_single_target_plot(importance_df, target, output_dir="outputs"):
    """Create a single feature importance plot matching reference style"""

    # Filter and sort data for this target
    target_data = importance_df[importance_df["Target"] == target].copy()
    target_data = target_data.sort_values("Importance", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors matching reference (rainbow-like palette)
    colors = sns.color_palette("tab10", len(target_data))

    # Create horizontal bar plot
    bars = ax.barh(
        [FEATURE_LABELS[f] for f in target_data["Feature"]],
        target_data["Importance"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )

    # Add value labels on bars (matching reference style)
    for bar, val in zip(bars, target_data["Importance"]):
        ax.text(
            val + 0.01,  # Small offset
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=FONTSIZE - 2,
            fontweight="normal",
        )

    # Styling
    ax.set_xlabel("Importance Score", fontsize=FONTSIZE, fontweight="bold")
    ax.set_ylabel("Input Feature", fontsize=FONTSIZE, fontweight="bold")
    ax.set_xlim(0, max(target_data["Importance"]) * 1.25)

    style_axes(ax)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / f"feature_importance_{target}_no_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_all_targets_grid(importance_df, output_dir="outputs"):
    """Create 2x2 grid showing all targets"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]

        # Filter and sort data
        target_data = importance_df[importance_df["Target"] == target].copy()
        target_data = target_data.sort_values("Importance", ascending=True)

        # Colors
        colors = sns.color_palette("tab10", len(target_data))

        # Bar plot
        bars = ax.barh(
            [FEATURE_LABELS[f] for f in target_data["Feature"]],
            target_data["Importance"],
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.85,
        )

        # Value labels
        for bar, val in zip(bars, target_data["Importance"]):
            ax.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=FONTSIZE - 4,
                fontweight="normal",
            )

        ax.set_xlabel("Importance Score", fontsize=FONTSIZE, fontweight="bold")
        ax.set_ylabel("Input Feature", fontsize=FONTSIZE, fontweight="bold")
        ax.set_title(TARGET_TITLES[target], fontsize=FONTSIZE + 2, fontweight="bold", pad=10)
        ax.set_xlim(0, max(target_data["Importance"]) * 1.2)

        style_axes(ax)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / "feature_importance_all_targets_no_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(importance_df, output_dir="outputs"):
    """Create a summary table of feature importances"""

    # Pivot to wide format
    pivot_df = importance_df.pivot(index="Feature", columns="Target", values="Importance")
    pivot_df = pivot_df.reindex(FEATURES)

    # Rename features
    pivot_df.index = [FEATURE_LABELS[f] for f in pivot_df.index]
    pivot_df.columns = [TARGET_TITLES[t] for t in pivot_df.columns]

    # Add mean and std
    pivot_df["Mean"] = pivot_df.mean(axis=1)
    pivot_df["Std"] = pivot_df.std(axis=1)

    # Sort by mean importance
    pivot_df = pivot_df.sort_values("Mean", ascending=False)

    # Save to CSV
    output_path = Path(output_dir) / "feature_importance_summary_no_time.csv"
    pivot_df.to_csv(output_path, float_format="%.4f")
    print(f"✓ Saved: {output_path}")

    # Print to console
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE SUMMARY (EXCLUDING TIME)")
    print("=" * 80)
    print(pivot_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 80 + "\n")

    return pivot_df


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("GENERATING FEATURE IMPORTANCE PLOTS (EXCLUDING TIME)")
    print("=" * 80 + "\n")

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load feature importances
    print("Loading feature importances from models (excluding time)...")
    importance_df = load_feature_importances()

    if importance_df.empty:
        print("ERROR: No feature importance data loaded. Check model files.")
        return

    print(f"✓ Loaded importances for {importance_df['Target'].nunique()} targets")
    print(f"✓ Using {importance_df['Feature'].nunique()} features (time excluded)\n")

    # Generate plots for each target
    print("Generating individual plots...")
    for target in TARGETS:
        if target in importance_df["Target"].values:
            create_single_target_plot(importance_df, target, output_dir)

    print()

    # Generate combined grid
    print("Generating combined 2x2 grid plot...")
    create_all_targets_grid(importance_df, output_dir)

    print()

    # Generate summary table
    print("Generating summary table...")
    create_summary_table(importance_df, output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE! All feature importance visualizations saved to outputs/")
    print("Files are named with '_no_time' suffix")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
