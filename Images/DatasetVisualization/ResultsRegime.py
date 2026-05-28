import matplotlib.pyplot as plt

def make_regime_lift_chart(out_path="regime_lift_chart.png"):
    # Data (as proportions)
    pct_days_in_regime = 0.10
    pct_landslide_days_in_regime = 0.62
    lift = pct_landslide_days_in_regime / pct_days_in_regime

    labels = ["Extreme regime days", "Landslide days in extreme regime"]
    values = [pct_days_in_regime, pct_landslide_days_in_regime]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, values)

    # Y axis as percent
    ax.set_ylim(0, 1.0)
    yticks = [i / 10 for i in range(0, 11)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t*100)}%" for t in yticks])

    ax.set_title("Extreme Rainfall Regime Concentrates Landslide Days", pad=18)
    ax.text(0.5, 1.03, "Top 10% rainfall days capture a disproportionate share of landslide days.",
            transform=ax.transAxes, ha="center", va="bottom")

    # Bar labels
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{int(round(v * 100))}%",
                ha="center", va="bottom", fontsize=14, fontweight="bold")

    # Lift callout
    callout = f"{lift:.1f}× Lift\n({int(round(pct_landslide_days_in_regime*100))}% / {int(round(pct_days_in_regime*100))}%)"
    ax.text(0.72, 0.78, callout, transform=ax.transAxes,
            ha="center", va="center", fontsize=26, fontweight="bold")

    # Clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    make_regime_lift_chart()
