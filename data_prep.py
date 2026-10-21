import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = next(p for p in (HERE / "data", HERE.parent / "data") if p.exists())
OUT = HERE / "output" if DATA.parent == HERE else HERE
TZ = timezone(timedelta(hours=9))
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿⬀-⯿⌀-⏿‼⁉™ℹⓂ▪-◾⤴⤵〰〽㊗㊙]")
LEADING = re.compile(r"^[\s«“\"'(\[​️]+")
KEYCAP = re.compile(r"^[0-9#*]️?⃣")
HOUR_EDGES = [0, 10, 12, 15, 18, 24]
HOUR_LABELS = ["до 10", "10–11", "12–14", "15–17", "18+"]
WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
DAY_POSTS_EDGES = [0, 2, 4, 6, np.inf]
DAY_POSTS_LABELS = ["1–2", "3–4", "5–6", "7+"]
REPOST_EDGES = [-1, 0, 1, 3, np.inf]
REPOST_LABELS = ["0", "1", "2–3", "4+"]
MIN_AGE_DAYS = 2
MONTHS_BACK = 6


def today():
    return pd.Timestamp(datetime.now(TZ).date())


def month_blocks(end):
    return [(end - pd.DateOffset(months=k + 1), end - pd.DateOffset(months=k)) for k in range(MONTHS_BACK)][::-1]


def block_label(start, end):
    return f"{start:%d.%m}–{(end - pd.Timedelta(days=1)):%d.%m}"


def leading_emoji(text):
    lead = LEADING.sub("", text)
    return bool(lead) and (bool(EMOJI.match(lead[0])) or bool(KEYCAP.match(lead)))


def media_type(attachments):
    if "video" in attachments:
        return "видео"
    photos = attachments.count("photo")
    return "нет" if photos == 0 else "1 фото" if photos == 1 else "2+ фото"


def quantile_classes(values):
    edges = np.unique(np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).round().astype(int))
    labels = [f"{lo}–{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.cut(values, edges, labels=labels, include_lowest=True).astype(str)


def load(csv_name="posts_2026-04_2026-10.csv"):
    end = today()
    start = end - pd.DateOffset(months=MONTHS_BACK)
    raw = pd.read_csv(DATA / csv_name)
    raw["date"] = pd.to_datetime(raw.date.str[:19])
    raw = raw[(raw.is_repost == 0) & (raw.is_ad == 0) & (raw.date >= start)].copy()
    raw["day"] = raw.date.dt.normalize()
    day_posts, day_views = raw.groupby("day").id.transform("size"), raw.groupby("day").views.transform("sum")
    raw["day_posts"], raw["day_views_others"] = day_posts, day_views - raw.views
    d = raw[(raw.views > 0) & (raw.date < end - pd.Timedelta(days=MIN_AGE_DAYS))].copy()
    text, attachments = d.text.fillna(""), d.attachments.fillna("")
    d["reactions"] = d.likes + d.comments + d.reposts
    d["ERV"] = d.reactions / d.views * 100
    d["ERP"] = d.reactions / d.members * 100
    d["hour_class"] = pd.cut(d.date.dt.hour, HOUR_EDGES, right=False, labels=HOUR_LABELS).astype(str)
    d["weekday_class"] = d.date.dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    d["media_presence"] = np.where(attachments == "", "без медиа", "с медиа")
    d["media_class"] = attachments.map(media_type)
    d["text_len"] = text.str.len()
    d["len_class"] = quantile_classes(d.text_len.values)
    d["emoji_start"] = np.where(text.map(leading_emoji), "эмодзи в начале", "без эмодзи в начале")
    d["emoji_any"] = np.where(text.map(lambda t: bool(EMOJI.search(t))), "эмодзи в тексте", "без эмодзи в тексте")
    d["emoji_class"] = np.where(d.emoji_start == "эмодзи в начале", "эмодзи в начале", np.where(d.emoji_any == "эмодзи в тексте", "эмодзи только внутри", "без эмодзи"))
    d["day_posts_class"] = pd.cut(d.day_posts, DAY_POSTS_EDGES, labels=DAY_POSTS_LABELS).astype(str)
    d["day_views_class"] = quantile_classes(d.day_views_others.values)
    d["repost_class"] = pd.cut(d.reposts, REPOST_EDGES, labels=REPOST_LABELS).astype(str)
    d["block"] = pd.NA
    for i, (lo, hi) in enumerate(month_blocks(end)):
        d.loc[(d.date >= lo) & (d.date < hi), "block"] = i
    d["block"] = d.block.astype("Int64")
    return d.reset_index(drop=True)


def ordered(d, class_column, value_column):
    return list(dict.fromkeys(d.sort_values(value_column)[class_column]))


def chart_specs(d):
    return {
        "H1_время_постинга": (d, "hour_class", HOUR_LABELS),
        "H3_тип_медиа": (d[d.media_class != "нет"], "media_class", ["1 фото", "2+ фото", "видео"]),
        "H4_размер_поста": (d, "len_class", ordered(d, "len_class", "text_len")),
        "H5_H6_эмодзи": (d, "emoji_class", ["без эмодзи", "эмодзи только внутри", "эмодзи в начале"]),
        "H7_дневной_трафик": (d, "day_views_class", ordered(d, "day_views_class", "day_views_others")),
        "H8_репосты": (d, "repost_class", REPOST_LABELS),
    }


def hypotheses(d):
    media = d[d.media_class != "нет"]
    return {
        "H1_время_постинга": {
            "data": d, "metrics": ["ERV"],
            "factors": {"Час публикации (UTC+9)": ("hour_class", HOUR_LABELS), "День недели": ("weekday_class", WEEKDAYS)},
        },
        "H2_наличие_медиа": {"data": d, "metrics": ["ERV"], "factors": {"Наличие медиа": ("media_presence", ["без медиа"])}},
        "H3_тип_медиа": {"data": media, "metrics": ["ERV"], "factors": {"Тип медиа": ("media_class", ["1 фото", "2+ фото", "видео"])}},
        "H4_размер_поста": {"data": d, "metrics": ["ERV"], "factors": {"Длина текста, символов (квартили)": ("len_class", ordered(d, "len_class", "text_len"))}},
        "H5_эмодзи_в_начале": {"data": d, "metrics": ["ERV", "ERP"], "factors": {"Эмодзи в начале поста": ("emoji_start", ["эмодзи в начале"])}},
        "H6_эмодзи_в_посте": {"data": d, "metrics": ["ERV", "ERP"], "factors": {"Эмодзи в тексте поста": ("emoji_any", ["эмодзи в тексте"])}},
        "H7_дневной_трафик": {
            "data": d, "metrics": ["ERV"],
            "factors": {"Постов за день": ("day_posts_class", DAY_POSTS_LABELS), "Просмотры остальных постов за день (квартили)": ("day_views_class", ordered(d, "day_views_class", "day_views_others"))},
        },
        "H8_репосты": {"data": d, "metrics": ["ERV"], "factors": {"Репостов у поста": ("repost_class", REPOST_LABELS)}},
    }
