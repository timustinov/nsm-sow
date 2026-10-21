import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from criteria import chi2_independence
from data_prep import OUT, block_label, month_blocks, today

CHARTS = OUT / "charts"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#d6a21a"]
GREY = "#898781"
STATISTICS = {"Среднее": np.mean, "Медиана": np.median}


def test_note(groups):
    return f"KS p={stats.ks_2samp(*groups).pvalue:.3f}" if len(groups) == 2 else f"χ² p={chi2_independence(groups):.3f}"


def draw(ax, data, column, classes, metric, stat, labels):
    values, block, n = data[metric].values, data.block.astype(float).values, len(labels)
    series = [(f"{c} (n={(data[column] == c).sum()})", (data[column] == c).values, color) for c, color in zip(classes, COLORS)]
    series.append(("все посты", np.ones(len(data), dtype=bool), GREY))
    for name, mask, color in series:
        monthly = [stat(values[mask & (block == i)]) if (mask & (block == i)).any() else np.nan for i in range(n)]
        ax.plot(range(n), monthly, marker="o", ms=4, color=color, label=name, **({"ls": "--", "lw": 1.2} if color == GREY else {"lw": 2}))
        ax.scatter([n + 0.7], [stat(values[mask])], marker="D", s=60, color=color)
    ax.axvline(n - 0.35, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(list(range(n)) + [n + 0.7], labels + ["всего"], rotation=45, fontsize=8)
    ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False)


def hypothesis_chart(name, data, column, classes, metric="ERV"):
    labels = [block_label(lo, hi) for lo, hi in month_blocks(today())]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (title, stat) in zip(axes, STATISTICS.items()):
        draw(ax, data, column, classes, metric, stat, labels)
        ax.set_title(f"{title} {metric}, %", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    note = test_note([data.loc[data[column] == c, metric].values for c in classes])
    fig.suptitle(f"{name.replace('_', ' ')} {metric}  ({note})", x=0.01, ha="left", fontsize=10, color=GREY)
    fig.tight_layout()
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}{'' if metric == 'ERV' else '_' + metric}.png", dpi=100)
    plt.close(fig)
