import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alerts.evaluate import skill_interval
from alerts.run import REPORTS

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LINES = {
    "persistence": ("persistence", MUTED, ":"),
    "logistic": ("logistic, 4 inputs", MUTED, "--"),
    "boosting": ("boosting, alert history", BLUE, "-"),
    "boosting_text": ("boosting + Telegram posts", ORANGE, "-"),
}
PANELS = {"all": "All forecasts", "quiet_now": "No alert at forecast time"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def clean(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def skill_chart(table, on):
    horizons = sorted(table.horizon.unique())
    x = np.arange(len(horizons))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (subset, title) in zip(axes, PANELS.items()):
        part = table[table.subset == subset]
        for method, (label, color, style) in LINES.items():
            rows = part[part.method == method].set_index("horizon").reindex(horizons)
            if rows.brier_skill.isna().all():
                continue
            ax.plot(x, rows.brier_skill, color=color, linestyle=style, linewidth=2, marker="o", markersize=5, label=label)
            if style == "-":
                ax.fill_between(x, rows.skill_low, rows.skill_high, color=color, alpha=0.12, linewidth=0)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xticks(x, [f"{h} h" for h in horizons])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("hours ahead")
        clean(ax)
    axes[0].set_ylabel("Brier skill vs 28-day climatology")
    axes[1].legend(frameon=False, loc="upper right", labelcolor=SECONDARY)
    fig.suptitle("Forecast skill by horizon (0 = no better than climatology)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS / f"skill_{on}.png", dpi=160)
    plt.close(fig)


def calibration_chart(predictions, method, on):
    horizons = [1, 3, 24]
    fig, axes = plt.subplots(1, len(horizons), figsize=(10, 3.6), sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, h in zip(axes, horizons):
        part = predictions[predictions.horizon == h]
        grouped = part.groupby(pd.cut(part[method], bins, include_lowest=True), observed=True)
        stats = grouped.agg(predicted=(method, "mean"), observed=("y", "mean"), n=("y", "size"))
        stats = stats[stats.n >= 30]
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
        ax.plot(stats.predicted, stats.observed, color=BLUE, linewidth=2, marker="o", markersize=5)
        ax.set_title(f"{h} h ahead", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("forecast probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        clean(ax)
    axes[0].set_ylabel("share of hours with an alert")
    fig.suptitle(f"Calibration of {LINES[method][0]} (on the diagonal = honest probabilities)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS / f"calibration_{on}.png", dpi=160)
    plt.close(fig)


def markdown(table, predictions):
    lines = []
    for subset, title in PANELS.items():
        part = table[table.subset == subset]
        skill = part.pivot(index="method", columns="horizon", values="brier_skill").reindex(LINES).dropna(how="all")
        lines += [f"**{title}**, Brier skill vs climatology", "", "| method | " + " | ".join(f"{h} h" for h in skill.columns) + " |"]
        lines.append("|---|" + "---|" * len(skill.columns))
        for method, row in skill.iterrows():
            lines.append(f"| {LINES[method][0]} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
        lines.append("")
    if "boosting_text" in predictions:
        lines += ["**Gain from Telegram posts**, Brier skill of boosting + Telegram vs boosting, 95% interval", ""]
        lines += ["| forecasts | " + " | ".join(f"{h} h" for h in sorted(predictions.horizon.unique())) + " |", "|---|" + "---|" * predictions.horizon.nunique()]
        for subset, title in PANELS.items():
            cells = []
            for h, part in predictions.groupby("horizon"):
                part = part if subset == "all" else part[part.on_now == 0]
                point = 1 - ((part.boosting_text - part.y) ** 2).sum() / ((part.boosting - part.y) ** 2).sum()
                low, high = skill_interval(part, "boosting_text", "boosting")
                cells.append(f"{point:.3f} ({low:.3f} to {high:.3f})")
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(on="val"):
    table = pd.read_csv(REPORTS / f"metrics_{on}.csv")
    predictions = pd.read_csv(REPORTS / f"predictions_{on}.csv.gz", parse_dates=["origin"])
    skill_chart(table, on)
    calibration_chart(predictions, "boosting_text" if "boosting_text" in predictions else "boosting", on)
    text = markdown(table, predictions)
    (REPORTS / f"results_{on}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
