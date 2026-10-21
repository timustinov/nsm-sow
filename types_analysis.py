import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
SHORT = {1: "Меры поддержки", 2: "СВО и ветераны", 3: "Память и патриотика", 4: "Инфраструктура и ЖКХ", 5: "Образование и молодёжь",
         6: "Спорт", 7: "Культура и события", 8: "Власть и мероприятия", 9: "Экономика и бизнес", 10: "Вовлекающий контент"}
BASE = "ltext + video + photos + edited + weekend + lday + C(month)"
STREAMS = ["без эмодзи", "с эмодзи"]


def load(name):
    d = pd.read_csv(f"data/posts_{name}.csv", parse_dates=["date"]).merge(pd.read_csv(f"data/labels_{name}.csv"), on="id").sort_values("date").reset_index(drop=True)
    text, a = d.text.fillna(""), d.attachments.fillna("")
    d["type_name"] = d.type.map(SHORT)
    d["type_alt"] = np.where(d.confidence_pass1 == "low", d.alt_type.fillna(d.type), d.type).astype(int)
    d["reactions"] = d.likes + d.comments + d.reposts
    d["month"], d["half"] = d.date.dt.strftime("%Y-%m"), np.where(d.date.dt.month <= 6, "Q2", "Q3")
    d["emoji_start"] = text.str.strip().str.match(r"^[^\w«“\"'(\[\s]").astype(int)
    d["photos"], d["video"], d["weekend"] = a.str.count("photo"), a.str.contains("video").astype(int), (d.date.dt.dayofweek >= 5).astype(int)
    d["ltext"], d["lviews"] = np.log1p(text.str.len()), np.log(d.views)
    d["lday"] = np.log(d.groupby(d.date.dt.date).id.transform("size"))
    return d


def type_table(d):
    g = d.groupby("type_name")
    return pd.DataFrame({"постов": g.size(), "доля постов": g.size() / len(d), "доля просмотров": g.views.sum() / d.views.sum(), "доля реакций": g.reactions.sum() / d.reactions.sum(),
                         "медиана views": g.views.median(), "доля >=10k": g.views.apply(lambda s: (s >= 10_000).mean()), "медиана реакций": g.reactions.median(),
                         "ERV, %": g.reactions.sum() / g.views.sum() * 100, "доля с эмодзи": g.emoji_start.mean()}).sort_values("доля просмотров", ascending=False)


def emoji_models(d):
    cl = {"cov_type": "cluster", "cov_kwds": {"groups": d.date.dt.date.astype(str).factorize()[0]}}
    fit = lambda rhs: smf.ols(f"lviews ~ {BASE} + {rhs}", d).fit(**cl)
    pct = lambda m: "{:+.0f}% (ИД {:+.0f}…{:+.0f}%)".format(*[(np.exp(x) - 1) * 100 for x in (m.params["emoji_start"], *m.conf_int().loc["emoji_start"])])
    r = fit("emoji_start")
    print(f"\n[3] log(просмотры): база R2 = {smf.ols(f'lviews ~ {BASE}', d).fit().rsquared:.3f}; + эмодзи R2 = {r.rsquared:.3f}, эффект эмодзи {pct(r)}")
    for col in ["type", "type_pass1", "type_alt"]:
        print(f"   {col:10} + тип R2 = {fit(f'C({col})').rsquared:.3f}; + эмодзи + тип R2 = {fit(f'emoji_start + C({col})').rsquared:.3f}, эффект эмодзи {pct(fit(f'emoji_start + C({col})'))}")
    f0, f1 = smf.ols(f"lviews ~ {BASE} + emoji_start + C(type)", d).fit(), smf.ols(f"lviews ~ {BASE} + emoji_start * C(type)", d).fit()
    print(f"   эффект эмодзи одинаков по типам? F-тест взаимодействия p = {f1.compare_f_test(f0)[1]:.2f}")


def rate_ratio(d, y):
    m = smf.glm(f"{y} ~ 0 + C(type_name) + lviews + emoji_start", d, family=sm.families.Poisson()).fit(cov_type="HC1")
    k, n = d.type_name.nunique(), len(m.params)
    R = np.eye(n)[:k] - np.eye(n)[:k].mean(0)
    ci = np.exp(m.t_test(R).conf_int())
    res = pd.DataFrame({"RR": np.exp(m.t_test(R).effect), "lo": ci[:, 0], "hi": ci[:, 1]}, index=m.params.index[:k].str.extract(r"\[(.*)\]")[0].values)
    return res, float(m.wald_test(R[:k - 1], scalar=True).pvalue), m.params["lviews"], np.exp(m.params["emoji_start"])


def raw_rates(g):
    return pd.DataFrame({"репосты на 1000 просмотров": g.reposts.sum() / g.views.sum() * 1000, "комментарии на 1000": g.comments.sum() / g.views.sum() * 1000,
                         "доля постов с >=5 репостами": g.reposts.apply(lambda s: (s >= 5).mean())})


def shift(d, keys):
    c = d.groupby(keys + ["half"]).views.agg(["size", "mean"]).unstack("half").fillna(0)
    n2, n3, m2, m3 = c[("size", "Q2")], c[("size", "Q3")], c[("mean", "Q2")], c[("mean", "Q3")]
    N2, N3, s2, s3 = n2.sum(), n3.sum(), n2 / n2.sum(), n3 / n3.sum()
    v2 = N2 * (s2 * m2).sum()
    return pd.Series({"объём": (N3 - N2) * (s2 * m2).sum() / v2, "состав": N3 * ((s3 - s2) * m2).sum() / v2, "внутри": N3 * (s3 * (m3 - m2)).sum() / v2, "итого": N3 * (s3 * m3).sum() / v2 - 1}) * 100


def decomposition(d):
    return pd.DataFrame({"группы: типы": shift(d, ["type_name"]), "группы: потоки": shift(d, ["emoji_start"]), "группы: тип × поток": shift(d, ["type_name", "emoji_start"])}).T


def quarter_change(d, index):
    q = d.groupby(["type_name", "half"]).views.agg(["size", "mean"]).unstack()
    chg = pd.DataFrame({"n Q2": q[("size", "Q2")], "n Q3": q[("size", "Q3")], "среднее Q2": q[("mean", "Q2")], "среднее Q3": q[("mean", "Q3")]}).loc[index]
    chg["изм., %"] = (chg["среднее Q3"] / chg["среднее Q2"] - 1) * 100
    chg["p (Манн–Уитни)"] = [stats.mannwhitneyu(*[s.views for _, s in d[d.type_name == k].groupby("half")]).pvalue for k in chg.index]
    return chg


def plot(d, t, rr, chg, dec, name):
    ink, sub, mut, grid, surf, blue, orange, aqua = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"figure.facecolor": surf, "savefig.facecolor": surf, "axes.facecolor": surf, "text.color": ink, "axes.edgecolor": grid, "xtick.color": mut, "ytick.color": ink, "axes.labelcolor": sub,
                         "grid.color": grid, "grid.linewidth": 0.7, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left", "axes.titleweight": "bold",
                         "axes.titlesize": 13, "font.size": 11, "legend.frameon": False})
    fig, ax = plt.subplots(2, 3, figsize=(24, 12.5))
    order = t.sort_values("доля просмотров").index
    bars = dict(color=[blue, orange], width=0.72)
    (t.loc[order, ["доля постов", "доля просмотров"]] * 100).plot.barh(ax=ax[0, 0], title="Доля постов и доля просмотров, %", **bars)
    piv = (d.groupby(["type_name", "emoji_start"]).views.median().unstack().loc[order] / 1000).set_axis(STREAMS, axis=1)
    gap = piv.iloc[:, 0] / piv.iloc[:, 1]
    piv.plot.barh(ax=ax[0, 1], xlim=(0, 16), title=f"Медиана просмотров, тыс. (без эмодзи выше в {gap.min():.0f}–{gap.max():.0f} раз)", **bars)
    ax[0, 1].legend(loc="lower right")
    x = rr["reactions"][0].loc[order]
    ax[0, 2].errorbar(x.RR, range(len(x)), xerr=[x.RR - x.lo, x.hi - x.RR], fmt="o", ms=8, lw=2, color=blue)
    ax[0, 2].axvline(1, color=mut, lw=1)
    ax[0, 2].set(yticks=range(len(x)), yticklabels=x.index, title="Реакции при том же охвате, × к среднему типу")
    chg.loc[order, ["среднее Q2", "среднее Q3"]].div(1000).set_axis(["Q2 (апр–июн)", "Q3 (июл–сен)"], axis=1).plot.barh(ax=ax[1, 0], title="Среднее число просмотров на пост, тыс.", **bars)
    dec[["объём", "состав", "внутри"]].set_axis(["число постов", "состав групп", "просмотры внутри групп"], axis=1).plot.bar(
        ax=ax[1, 1], color=[blue, orange, aqua], width=0.7, rot=0, title=f"Из чего складывается падение просмотров Q2→Q3 ({dec['итого'].iloc[0]:.0f}%), п.п.")
    for c in ax[1, 1].containers:
        ax[1, 1].bar_label(c, fmt="%.0f", color=sub)
    (d.pivot_table(index="month", columns="emoji_start", values="views", aggfunc="sum") / 1000).set_axis(STREAMS, axis=1).plot(ax=ax[1, 2], color=[blue, orange], lw=2, marker="o", ms=8, ylim=(0, None), title="Сумма просмотров по месяцам, тыс.")
    for a_ in ax.flat:
        a_.set(xlabel="", ylabel="")
        a_.grid(axis="y" if a_ is ax[1, 2] else "x")
    ax[0, 2].set_xlabel("точка — оценка, линия — 95% ИД")
    fig.tight_layout()
    fig.savefig(f"data/types_{name}.png", dpi=110)


def main(name):
    d = load(name)
    t = type_table(d)
    print(f"[1] типы (n={len(d)}), по доле просмотров\n", t.round(2).to_string())
    cell = d.groupby(["type_name", "emoji_start"]).views.agg(["median", "size"]).unstack().loc[t.index]
    print("\n[2] медиана просмотров и n по типу и потоку (0 — без эмодзи в начале, 1 — с эмодзи)\n", cell.round(0).to_string())
    print(f"    корреляция Спирмена по типам: доля эмодзи vs медиана просмотров = {stats.spearmanr(t['доля с эмодзи'], t['медиана views'])[0]:.2f}")
    emoji_models(d)
    rr = {y: rate_ratio(d, y) for y in ("reactions", "reposts", "comments")}
    print("\n[4] при том же охвате и потоке, во сколько раз больше/меньше, чем у среднего типа (95% ИД)\n",
          pd.DataFrame({y: v[0].apply(lambda s: f"{s.RR:.2f} ({s.lo:.2f}–{s.hi:.2f})", axis=1) for y, v in rr.items()}).loc[t.index].to_string())
    for y, v in rr.items():
        print(f"    {y:9}: типы различаются p = {v[1]:.3f}; эластичность по просмотрам {v[2]:.2f}; эмодзи-поток ×{v[3]:.2f}")
    print("\n[5] сырые доли (не поправлены на охват)\n", raw_rates(d.groupby("type_name")).loc[t.index].round(2).to_string())
    dec, chg = decomposition(d), quarter_change(d, t.index)
    print("\n[6] изменение суммы просмотров Q2 -> Q3 по разложению, п.п. от Q2\n", dec.round(1).to_string())
    print("\n[7] просмотры на пост по типам, Q2 (апр–июн) и Q3 (июл–сен)\n", chg.round(2).to_string())
    print("\n[8] по потокам\n", d.groupby(["emoji_start", "half"]).views.agg(["size", "mean", "median", lambda s: (s >= 10_000).mean()]).rename(columns={"<lambda_0>": "доля >=10k"}).round(2).to_string())
    print("\n[9] доли типов по месяцам\n", pd.crosstab(d.month, d.type_name, normalize="index").loc[:, t.index].round(2).to_string())
    plot(d, t, rr, chg, dec, name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-04_2026-09")
