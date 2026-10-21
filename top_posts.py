import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from collect import TZ
from vk_client import VKClient

METRICS = {"erp": "ERP, %", "erv": "ERV, %", "views": "views", "reactions": "reactions"}


def collect(client, domain, start, end):
    group = client.get_group(domain, "members_count")
    posts = client.get_wall_posts_by_date_range(domain, end_date=end, start_date=start)
    return pd.DataFrame(
        [
            {
                "domain": domain,
                "group": group["name"],
                "members": group["members_count"],
                "id": p.id,
                "url": f"https://vk.com/wall-{group['id']}_{p.id}",
                "date": p.get_normalized_datetime(TZ).date(),
                "views": p.views.count,
                "likes": p.likes.count,
                "comments": p.comments.count,
                "reposts": p.reposts.count,
                "format": p.attachments[0]["type"] if p.attachments else "text",
                "is_repost": int(p.copy_history is not None),
                "is_ad": p.marked_as_ads,
                "text": p.text,
            }
            for p in posts
        ]
    )


def add_metrics(df):
    df["reactions"] = df.likes + df.comments + df.reposts
    df["ERP, %"] = df.reactions / df.members * 100
    df["ERV, %"] = df.reactions / df.views.where(df.views > 0) * 100
    return df


def top_per_public(df, metric, n):
    return (
        df.sort_values(METRICS[metric], ascending=False)
        .groupby("domain")
        .head(n)
        .sort_values(["domain", METRICS[metric]], ascending=[True, False])
    )


def show(df, metric):
    cols = ["domain", "date", "views", "reactions", "ERP, %", "ERV, %", "format", "url"]
    t = df[cols].copy()
    t["text"] = df.text.str.replace(r"\s+", " ", regex=True).str[:70]
    print(t.round(3).to_string(index=False))


def main(start, end, metric, n, domains_path, min_views):
    client = VKClient()
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=TZ)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=TZ) + timedelta(days=1)
    domains = [
        line.strip() for line in open(domains_path, encoding="utf-8") if line.strip()
    ]
    frames = [collect(client, d, s, e) for d in domains]
    df = add_metrics(pd.concat(frames, ignore_index=True))
    df = df[(df.is_repost == 0) & (df.is_ad == 0) & (df.views >= min_views)]
    print("=" * 80)
    print(f"{len(df)} posts count for {start} — {end}, with {METRICS[metric]}\n")
    
    if len(domains) == 1:
        top = df.sort_values(METRICS[metric], ascending=False)
        show(top.head(10), metric)
    else:
        top = top_per_public(df, metric, n)
        show(top, metric)
        print("=" * 80)
        print(f"\ntop-{n * 3}\n")
        overall = df.sort_values(METRICS[metric], ascending=False).head(n * 3)
        show(overall, metric)
    
    out = f"data/{domains[0] + "_" if len(domains) == 1 else ""}top_posts_{start}_{end}_{metric}.csv"
    top.to_csv(out, index=False, encoding="utf-8-sig")
    metric_col = np.array(top[METRICS[metric]])
    if len(domains) == 1:
        print("=" * 80)
        print(
            f"max = {max(metric_col)}, mean = {np.mean(metric_col)}, median = {np.median(metric_col)}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str)
    ap.add_argument("--end", type=str)
    ap.add_argument("--metric", choices=METRICS, default="erp")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--domains", default="data/publics.txt")
    ap.add_argument("--min-views", type=int, default=300)
    a = ap.parse_args()
    main(a.start, a.end, a.metric, a.top, a.domains, a.min_views)
