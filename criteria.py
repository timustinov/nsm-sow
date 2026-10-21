import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def chi2_independence(groups, bins=4):
    edges = np.unique(np.quantile(np.concatenate(groups), np.linspace(0, 1, bins + 1)))
    table = np.array([np.histogram(g, edges)[0] for g in groups])
    table = table[:, table.sum(axis=0) > 0]
    return stats.chi2_contingency(table)[1] if table.shape[1] > 1 else np.nan


TWO_SAMPLE = {
    "KS": lambda x, y: stats.ks_2samp(x, y).pvalue,
    "MWU": lambda x, y: stats.mannwhitneyu(x, y).pvalue,
    "Chi2": lambda x, y: chi2_independence([x, y]),
}


def class_vs_complement(data, hypothesis, factor, column, classes, metric):
    rows = []
    for cls in classes:
        hit = data[column] == cls
        x, y = data.loc[hit, metric].values, data.loc[~hit, metric].values
        rows.append({"гипотеза": hypothesis, "фактор": factor, "класс": cls, "метрика": metric, "n_класс": len(x), "n_доп": len(y),
                     "mean_класс": x.mean(), "mean_доп": y.mean(), "median_класс": np.median(x), "median_доп": np.median(y),
                     **{name: test(x, y) for name, test in TWO_SAMPLE.items()}})
    return rows


def overall_chi2(data, hypothesis, factor, column, classes, metric):
    groups = [g for g in (data.loc[data[column] == c, metric].values for c in classes) if len(g) > 1]
    return {"гипотеза": hypothesis, "фактор": factor, "метрика": metric, "классов": len(groups), "Chi2": chi2_independence(groups)}


def holm(table):
    for name in TWO_SAMPLE:
        table[name + "_holm"] = table.groupby("метрика")[name].transform(lambda p: multipletests(p.fillna(1), method="holm")[1])
    return table


def run_all(specs):
    pairs, overall = [], []
    for hypothesis, spec in specs.items():
        for factor, (column, classes) in spec["factors"].items():
            for metric in spec["metrics"]:
                pairs += class_vs_complement(spec["data"], hypothesis, factor, column, classes, metric)
                if len(classes) > 1:
                    overall.append(overall_chi2(spec["data"], hypothesis, factor, column, classes, metric))
    return holm(pd.DataFrame(pairs)), pd.DataFrame(overall)
