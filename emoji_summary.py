import pandas as pd
from scipy import stats

METRICS = {"views": "Просмотры", "reactions": "Реакции", "ERP": "ERP, %", "ERV": "ERV, %"}
FACTORS = {"emoji_start": "эмодзи в начале", "emoji_any": "эмодзи в тексте"}


def metric_table(d):
    rows = []
    for column, cls in FACTORS.items():
        hit = d[column] == cls
        for metric, title in METRICS.items():
            x, y = d.loc[hit, metric], d.loc[~hit, metric]
            rows.append({"фактор": cls, "метрика": title, "mean_с": x.mean(), "mean_без": y.mean(), "mean_отношение": x.mean() / y.mean(),
                         "median_с": x.median(), "median_без": y.median(), "median_отношение": x.median() / y.median(),
                         "p_KS": stats.ks_2samp(x, y).pvalue, "p_MWU": stats.mannwhitneyu(x, y).pvalue})
    return pd.DataFrame(rows)


def at_equal_reach(d, bins=4):
    d = d.assign(reach=pd.qcut(d.views, bins, labels=[f"Q{i + 1}" for i in range(bins)]))
    return d.groupby(["reach", "emoji_start"], observed=True).agg(n=("id", "size"), views=("views", "mean"), ERV=("ERV", "mean"), ERP=("ERP", "mean")).unstack("emoji_start")
