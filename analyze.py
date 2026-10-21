import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["views", "likes", "comments", "reposts", "ERV", "ERP"]


def load(month):
    df = pd.read_csv(f"data/posts_{month}.csv", parse_dates=["date"])
    df["reactions"] = df.likes + df.comments + df.reposts
    df["ERP"] = df.reactions / df.members * 100
    df["ERV"] = df.reactions / df.views.where(df.views > 0) * 100
    df["month"] = df.date.dt.strftime("%Y-%m")
    return df


def add_meta(df):
    text, a = df.text.fillna(""), df.attachments.fillna("")
    df["hour"], df["weekday"] = df.date.dt.hour, df.date.dt.dayofweek
    df["text_len"], df["hashtags"] = text.str.len(), text.str.count("#")
    df["photos"], df["video"] = a.str.count("photo"), a.str.contains("video").astype(int)
    df["format"] = a.str.split(",").str[0].replace("", "text")
    df.loc[df.is_repost == 1, "format"] = "repost"
    return [c for c in ["text_len", "hashtags", "photos", "video", "attachments_length", "edited"] if df[c].nunique() > 1]


def by_month(df):
    m = df.groupby("month").agg(posts=("id", "size"), views=("views", "sum"), reactions=("reactions", "sum"), members=("members", "first"))
    m["ERP"] = m.reactions / m.posts / m.members * 100
    m["ERV"] = m.reactions / m.views * 100
    return m.round({"ERP": 3, "ERV": 3})


def plot(df, meta, month):
    corr = df[meta + METRICS].corr(method="spearman").loc[meta, METRICS]
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ax[0, 0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0, 0].set(xticks=range(len(METRICS)), xticklabels=METRICS, yticks=range(len(meta)), yticklabels=meta, title="Корреляция Спирмена: метаданные × метрики")
    for (i, j), v in np.ndenumerate(corr.values):
        ax[0, 0].text(j, i, f"{v:.2f}", ha="center", va="center")
    df.groupby(pd.cut(df.hour, [0, 10, 12, 15, 18, 24], right=False)).ERV.median().plot.bar(ax=ax[0, 1], title="Медиана ERV по часу публикации (бины), %")
    df.groupby("weekday").ERV.median().plot.bar(ax=ax[0, 2], title="Медиана ERV по дню недели (0 = пн), %")
    df.boxplot("ERV", by="format", ax=ax[1, 0])
    ax[1, 0].set(title="ERV по типу первого вложения, % (лог. шкала)", yscale="log")
    df.groupby("month").ERV.median().plot.bar(ax=ax[1, 1], title="Медиана ERV по месяцам, %")
    df.groupby("month").views.median().plot.bar(ax=ax[1, 2], title="Медиана просмотров поста по месяцам")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(f"data/charts_{month}.png", dpi=130)


def main(month):
    df = load(month)
    print(f"{len(df)} постов | ERP {df.reactions.sum() / len(df) / df.members[0] * 100:.3f}% | ERV {df.reactions.sum() / df.views.sum() * 100:.3f}%")
    print(by_month(df))
    plot(df, add_meta(df), month)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-09")
