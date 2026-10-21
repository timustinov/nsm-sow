import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vk_client import VKClient

DOMAIN = "amuroblgov"
TZ = timezone(timedelta(hours=9))


def to_row(p, members):
    return {
        "id": p.id, "from_id": p.from_id, "date": p.get_normalized_datetime(TZ).isoformat(), "text": p.text,
        "post_type": p.post_type.value, "attachments": ",".join(a["type"] for a in p.attachments),
        "attachments_length": p.attachments_length, "likes": p.likes.count, "comments": p.comments.count,
        "reposts": p.reposts.count, "views": p.views.count, "edited": int(p.edited is not None),
        "is_ad": p.marked_as_ads, "is_repost": int(p.copy_history is not None), "members": members,
    }


def main(months):
    name = months[0] if months[0] == months[-1] else f"{months[0]}_{months[-1]}"
    start = datetime.strptime(months[0], "%Y-%m").replace(tzinfo=TZ)
    end = (datetime.strptime(months[-1], "%Y-%m").replace(tzinfo=TZ) + timedelta(days=32)).replace(day=1)
    client = VKClient()
    posts = client.get_wall_posts_by_date_range(DOMAIN, end_date=end, start_date=start)
    members = client.get_group_members_count(DOMAIN)
    rows = [to_row(p, members) for p in posts]
    Path("data").mkdir(exist_ok=True)
    with open(f"data/posts_{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:] or ["2026-09"])
