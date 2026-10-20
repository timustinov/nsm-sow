"""Проверка гипотез по data/posts_<имя>.csv (репосты исключены).  Запуск: python stats.py 2026-04_2026-09"""
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
name = sys.argv[1] if len(sys.argv) > 1 else "2026-04_2026-09"
df = pd.read_csv(f"data/posts_{name}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
text, a = df.text.fillna(""), df.attachments.fillna("")
df["reactions"] = df.likes + df.comments + df.reposts
df["ERV"] = df.reactions / df.views * 100
df["month"], df["day"] = df.date.dt.strftime("%Y-%m"), df.date.dt.date
df["hour"], df["weekend"] = df.date.dt.hour, (df.date.dt.dayofweek >= 5).astype(int)
df["text_len"], df["photos"], df["video"] = text.str.len(), a.str.count("photo"), a.str.contains("video").astype(int)
df["emoji_start"] = text.str.strip().str.match(r"^[^\w«“\"'(\[\s]").astype(int)  # текст начинается с эмодзи или символа
df["gap_h"] = df.date.diff().dt.total_seconds() / 3600  # часов с предыдущего поста
d = df[(df.is_repost == 0) & df.gap_h.notna()].copy()
d["day_posts"] = d.groupby("day").id.transform("size")

# 1. два потока: посты с эмодзи в начале и без него
g = d.groupby("emoji_start").agg(posts=("id", "size"), views=("views", "sum"), reactions=("reactions", "sum"), views_med=("views", "median"),
                                 react_med=("reactions", "median"), edited=("edited", "mean"), text_len_med=("text_len", "median"), video=("video", "mean"))
g["доля постов"], g["доля просмотров"], g["доля реакций"] = (g.posts / g.posts.sum(), g.views / g.views.sum(), g.reactions / g.reactions.sum())
g["ERV, %"] = g.reactions / g.views * 100
print("[1] потоки (emoji_start=1: текст начинается с эмодзи)\n", g.round(3).T.to_string())
m = d.groupby(["month", "emoji_start"]).views.sum().unstack().div(1000).round(0)
m.columns = ["просмотры без эмодзи, тыс.", "просмотры с эмодзи, тыс."]
m["без эмодзи: медиана views"] = d[d.emoji_start == 0].groupby("month").views.median()
m["без эмодзи: доля постов >=10k"] = d[d.emoji_start == 0].groupby("month").views.apply(lambda s: (s >= 10_000).mean()).round(2)
print("\n   по месяцам:\n", m.to_string())

# 2. ранговые корреляции с поправкой на множественные сравнения (FDR 5%)
feats = ["video", "photos", "text_len", "emoji_start", "edited", "weekend", "hour", "day_posts", "gap_h"]
res = pd.DataFrame([(f, o, *stats.spearmanr(d[f], d[o])) for f in feats for o in ("views", "reactions", "ERV")], columns=["f", "o", "rho", "p"])
res["cell"] = [f"{r:+.2f}{'*' if q < 0.05 else ' '}" for r, q in zip(res.rho, multipletests(res.p, method="fdr_bh")[1])]
print(f"\n[2] Спирмен, n={len(d)} (* — значимо после FDR 5%)\n", res.pivot(index="f", columns="o", values="cell").loc[feats, ["views", "reactions", "ERV"]].to_string())
print("\n   по часам:\n", d.groupby(pd.cut(d.hour, [0, 10, 12, 15, 18, 24], right=False), observed=True).agg(n=("id", "size"), views=("views", "median"), ERV=("ERV", "median")).round(2).to_string())

# 3. регрессии с кластерами по дню: охват отдельно, реакции при том же охвате отдельно
d["lviews"], d["lreact"], d["ltext"], d["lday"] = np.log(d.views), np.log(d.reactions), np.log1p(d.text_len), np.log(d.day_posts)
d["hour_bin"] = pd.cut(d.hour, [0, 10, 12, 15, 18, 24], right=False, labels=["<10", "10-11", "12-14", "15-17", "18+"])
rhs = "ltext + video + photos + emoji_start + edited + weekend + C(hour_bin) + lday + C(month)"
cl = {"cov_type": "cluster", "cov_kwds": {"groups": d.day.astype(str).factorize()[0]}}
for title, formula in [("[3a] просмотры", "lviews ~ " + rhs), ("[3b] реакции при том же охвате", "lreact ~ lviews + " + rhs)]:
    mod = smf.ols(formula, d).fit(**cl)
    ci = mod.conf_int()
    t = pd.DataFrame({"эффект, %": np.exp(mod.params) - 1, "от": np.exp(ci[0]) - 1, "до": np.exp(ci[1]) - 1, "p": mod.pvalues})
    t[["эффект, %", "от", "до"]] *= 100
    print(f"\n{title}: изменение, % (ltext, lday, lviews — на каждое увеличение в e раз), n={int(mod.nobs)}, R2={mod.rsquared:.2f}")
    print(t[~t.index.str.startswith(("Intercept", "C(month)"))].round(3).to_string())

# 4. частота публикаций: суммы по дням
day = d.groupby("day").agg(posts=("id", "size"), views=("views", "sum"), reactions=("reactions", "sum"), weekend=("weekend", "first"))
print("\n[4] медианы за день по числу постов в день\n", day.groupby(pd.cut(day.posts, [0, 2, 4, 6, 8, 20]), observed=True).agg(дней=("posts", "size"), views=("views", "median"), reactions=("reactions", "median")).round(0).to_string())
print(f"   Спирмен(постов за день, просмотры дня) = {stats.spearmanr(day.posts, day.views)[0]:+.2f} | постов в день: будни {day[day.weekend == 0].posts.mean():.1f}, выходные {day[day.weekend == 1].posts.mean():.1f}")
long = d[d.emoji_start == 0].copy()
long["n_day"] = long.groupby("day").id.transform("size")
print("   без эмодзи: охват поста по числу таких постов за день\n", long.groupby(pd.cut(long.n_day, [0, 1, 2, 3, 10]), observed=True).views.agg(["size", "median"]).round(0).to_string())
