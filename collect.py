"""Посты amuroblgov за месяц или диапазон месяцев -> data/posts_<имя>.csv.  Запуск: python collect.py 2026-04 2026-09"""
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vk_client import VKClient

DOMAIN = "amuroblgov"
TZ = timezone(timedelta(hours=9))  # Амурская область: UTC+9, без перехода на летнее время
months = sys.argv[1:] or ["2026-09"]  # один месяц или первый и последний месяцы диапазона
name = months[0] if months[0] == months[-1] else f"{months[0]}_{months[-1]}"
start = datetime.strptime(months[0], "%Y-%m").replace(tzinfo=TZ)
end = (datetime.strptime(months[-1], "%Y-%m").replace(tzinfo=TZ) + timedelta(days=32)).replace(day=1)

client = VKClient()
posts = client.get_wall_posts_by_date_range(DOMAIN, end_date=end, start_date=start)
members = client.get_group_members_count(DOMAIN)

rows = [{
    "id": p.id, "from_id": p.from_id, "date": p.get_normalized_datetime(TZ).isoformat(), "text": p.text,
    "post_type": p.post_type.value, "attachments": ",".join(a["type"] for a in p.attachments),
    "attachments_length": p.attachments_length, "likes": p.likes.count, "comments": p.comments.count,
    "reposts": p.reposts.count, "views": p.views.count, "edited": int(p.edited is not None),
    "is_ad": p.marked_as_ads, "is_repost": int(p.copy_history is not None), "members": members,
} for p in posts]

Path("data").mkdir(exist_ok=True)
with open(f"data/posts_{name}.csv", "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel открывает без мастера импорта
    writer = csv.DictWriter(f, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print(f"{name}: {len(rows)} постов, подписчиков {members} -> data/posts_{name}.csv")
