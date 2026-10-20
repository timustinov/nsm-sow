"""ERP/ERV и графики по data/posts_<месяц>.csv -> data/charts_<месяц>.png.  Запуск: python analyze.py 2026-09"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

month = sys.argv[1] if len(sys.argv) > 1 else "2026-09"
df = pd.read_csv(f"data/posts_{month}.csv", parse_dates=["date"])

df["reactions"] = df.likes + df.comments + df.reposts  # допущение: реакции = лайки + комментарии + репосты
df["ERP"] = df.reactions / df.members * 100  # реакции на 1 пост и 1 подписчика, %
df["ERV"] = df.reactions / df.views.where(df.views > 0) * 100  # реакции на 1 просмотр, %
# ERR (на 1 уникального пользователя) не посчитать: уникальный охват поста видит только администратор сообщества
print(f"{len(df)} постов | ERP {df.reactions.sum() / len(df) / df.members[0] * 100:.3f}% | ERV {df.reactions.sum() / df.views.sum() * 100:.3f}%")
df["month"] = df.date.dt.strftime("%Y-%m")
by_month = df.groupby("month").agg(posts=("id", "size"), views=("views", "sum"), reactions=("reactions", "sum"), members=("members", "first"))
by_month["ERP"] = by_month.reactions / by_month.posts / by_month.members * 100
by_month["ERV"] = by_month.reactions / by_month.views * 100
print(by_month.round({"ERP": 3, "ERV": 3}))

text, a = df.text.fillna(""), df.attachments.fillna("")
df["hour"], df["weekday"] = df.date.dt.hour, df.date.dt.dayofweek  # время уже в UTC+9
df["text_len"], df["hashtags"] = text.str.len(), text.str.count("#")
df["photos"], df["video"] = a.str.count("photo"), a.str.contains("video").astype(int)
df["format"] = a.str.split(",").str[0].replace("", "text")  # тип первого вложения
df.loc[df.is_repost == 1, "format"] = "repost"  # у репоста вложения лежат внутри copy_history, поэтому они не «text»

meta = [c for c in ["text_len", "hashtags", "photos", "video", "attachments_length", "edited"] if df[c].nunique() > 1]
metrics = ["views", "likes", "comments", "reposts", "ERV", "ERP"]
corr = df[meta + metrics].corr(method="spearman").loc[meta, metrics]

fig, ax = plt.subplots(2, 3, figsize=(20, 10))
ax[0, 0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[0, 0].set(xticks=range(len(metrics)), xticklabels=metrics, yticks=range(len(meta)), yticklabels=meta, title="Корреляция Спирмена: метаданные × метрики")
for (i, j), v in np.ndenumerate(corr.values):
    ax[0, 0].text(j, i, f"{v:.2f}", ha="center", va="center")
df.groupby(pd.cut(df.hour, [0, 10, 12, 15, 18, 24], right=False)).ERV.median().plot.bar(ax=ax[0, 1], title="Медиана ERV по часу публикации (бины), %")  # по отдельным часам в группах 2–8 постов
df.groupby("weekday").ERV.median().plot.bar(ax=ax[0, 2], title="Медиана ERV по дню недели (0 = пн), %")
df.boxplot("ERV", by="format", ax=ax[1, 0])
ax[1, 0].set(title="ERV по типу первого вложения, % (лог. шкала)", yscale="log")  # ERV с тяжёлым хвостом: выбросы сжимают ящики
df.groupby("month").ERV.median().plot.bar(ax=ax[1, 1], title="Медиана ERV по месяцам, %")
df.groupby("month").views.median().plot.bar(ax=ax[1, 2], title="Медиана просмотров поста по месяцам")
fig.suptitle("")
fig.tight_layout()
fig.savefig(f"data/charts_{month}.png", dpi=130)
