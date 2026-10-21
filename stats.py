import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HOUR_BINS = [0, 10, 12, 15, 18, 24]
FEATS = ["video", "photos", "text_len", "emoji_start", "edited", "weekend", "hour", "day_posts", "gap_h"]
RHS = "ltext + video + photos + emoji_start + edited + weekend + C(hour_bin) + lday + C(month)"


def load(name):
    df = pd.read_csv(f"data/posts_{name}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    text, a = df.text.fillna(""), df.attachments.fillna("")
    df["reactions"] = df.likes + df.comments + df.reposts
    df["ERV"] = df.reactions / df.views * 100
    df["month"], df["day"] = df.date.dt.strftime("%Y-%m"), df.date.dt.date
    df["hour"], df["weekend"] = df.date.dt.hour, (df.date.dt.dayofweek >= 5).astype(int)
    df["text_len"], df["photos"], df["video"] = text.str.len(), a.str.count("photo"), a.str.contains("video").astype(int)
    df["emoji_start"] = text.str.strip().str.match(r"^[^\w«“\"'(\[\s]").astype(int)
    df["gap_h"] = df.date.diff().dt.total_seconds() / 3600
    d = df[(df.is_repost == 0) & df.gap_h.notna()].copy()
    d["day_posts"] = d.groupby("day").id.transform("size")
    return d


def share_hi(s):
    return (s >= 10_000).mean()


def streams(d):
    g = d.groupby("emoji_start").agg(posts=("id", "size"), views=("views", "sum"), reactions=("reactions", "sum"), views_med=("views", "median"),
                                     react_med=("reactions", "median"), edited=("edited", "mean"), text_len_med=("text_len", "median"), video=("video", "mean"))
    g["доля постов"], g["доля просмотров"], g["доля реакций"] = (g.posts / g.posts.sum(), g.views / g.views.sum(), g.reactions / g.reactions.sum())
    g["ERV, %"] = g.reactions / g.views * 100
    return g


def streams_by_month(d):
    m = d.groupby(["month", "emoji_start"]).views.sum().unstack().div(1000).round(0)
    m.columns = ["просмотры без эмодзи, тыс.", "просмотры с эмодзи, тыс."]
    plain = d[d.emoji_start == 0].groupby("month").views
    m["без эмодзи: медиана views"] = plain.median()
    m["без эмодзи: доля постов >=10k"] = plain.apply(share_hi).round(2)
    return m


def rank_correlations(d):
    res = pd.DataFrame([(f, o, *stats.spearmanr(d[f], d[o])) for f in FEATS for o in ("views", "reactions", "ERV")], columns=["f", "o", "rho", "p"])
    res["cell"] = [f"{r:+.2f}{'*' if q < 0.05 else ' '}" for r, q in zip(res.rho, multipletests(res.p, method="fdr_bh")[1])]
    return res.pivot(index="f", columns="o", values="cell").loc[FEATS, ["views", "reactions", "ERV"]]


def by_hour(d):
    return d.groupby(pd.cut(d.hour, HOUR_BINS, right=False), observed=True).agg(n=("id", "size"), views=("views", "median"), ERV=("ERV", "median")).round(2)


def add_log_features(d):
    d["lviews"], d["lreact"], d["ltext"], d["lday"] = np.log(d.views), np.log(d.reactions), np.log1p(d.text_len), np.log(d.day_posts)
    d["hour_bin"] = pd.cut(d.hour, HOUR_BINS, right=False, labels=["<10", "10-11", "12-14", "15-17", "18+"])


def effects(d, formula):
    cl = {"cov_type": "cluster", "cov_kwds": {"groups": d.day.astype(str).factorize()[0]}}
    mod = smf.ols(formula, d).fit(**cl)
    ci = mod.conf_int()
    t = pd.DataFrame({"эффект, %": np.exp(mod.params) - 1, "от": np.exp(ci[0]) - 1, "до": np.exp(ci[1]) - 1, "p": mod.pvalues})
    t[["эффект, %", "от", "до"]] *= 100
    return mod, t[~t.index.str.startswith(("Intercept", "C(month)"))].round(3)


def daily(d):
    day = d.groupby("day").agg(posts=("id", "size"), views=("views", "sum"), reactions=("reactions", "sum"), weekend=("weekend", "first"))
    by_posts = day.groupby(pd.cut(day.posts, [0, 2, 4, 6, 8, 20]), observed=True).agg(дней=("posts", "size"), views=("views", "median"), reactions=("reactions", "median")).round(0)
    return day, by_posts


def plain_by_day_count(d):
    long = d[d.emoji_start == 0].copy()
    long["n_day"] = long.groupby("day").id.transform("size")
    return long.groupby(pd.cut(long.n_day, [0, 1, 2, 3, 10]), observed=True).views.agg(["size", "median"]).round(0)


def main(name):
    d = load(name)
    add_log_features(d)
    for title, formula in [("[3a] просмотры", "lviews ~ " + RHS), ("[3b] реакции при том же охвате", "lreact ~ lviews + " + RHS)]:
        mod, t = effects(d, formula)
    day, by_posts = daily(d)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-04_2026-09")
