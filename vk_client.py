import requests
import urllib.parse
import os
from dotenv import load_dotenv
import datetime
import schemas

load_dotenv()


class VKClient:
    BASE_URL: str = "https://api.vk.ru/method/"

    def __init__(self):
        self.token = os.getenv("VK_TOKEN")
        assert self.token is not None, "'VK_TOKEN' not defined"
        self.version = os.getenv("VK_VERSION")
        assert self.token is not None, "'VK_VERSION' not defined"

    def get(self, method: str, **kwargs):
        kwargs["v"] = self.version
        request = requests.get(
            url=urllib.parse.urljoin(self.BASE_URL, method),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            params=kwargs,
        )
        request.raise_for_status()
        response = request.json()
        if "error" in response:
            raise RuntimeError(f"{method}: {response['error']['error_msg']}")
        return response

    def get_wall_posts(self, **kwargs) -> list[schemas.VKPost]:
        return [
            schemas.VKPost.model_validate(item, extra="ignore")
            for item in self.get(method="wall.get", **kwargs)["response"]["items"]
        ]

    def get_wall_count(self, domain: str) -> int:
        return self.get(method="wall.get", domain=domain, count=1)["response"]["count"]

    def get_group(self, domain: str, fields: str = "members_count") -> dict:
        response = self.get(
            method="groups.getById", group_id=domain, fields=fields
        )["response"]
        group = response["groups"] if isinstance(response, dict) else response
        return group[0]

    def get_group_members_count(self, domain: str) -> int:
        return self.get_group(domain)["members_count"]

    def _find_offset(self, domain: str, timestamp: float, low: int, high: int) -> int:
        while low < high:
            middle = (low + high) // 2
            post = self.get_wall_posts(domain=domain, count=1, offset=middle)[0]
            if post.raw_date < timestamp:
                high = middle
            else:
                low = middle + 1
        return low

    def get_wall_posts_by_date_range(
        self, domain: str, end_date: datetime.datetime, start_date: datetime.datetime
    ) -> list[schemas.VKPost]:
        BATCH_SIZE: int = 100
        total = self.get_wall_count(domain)
        first = self._find_offset(domain, end_date.timestamp(), 1, total)
        last = self._find_offset(domain, start_date.timestamp(), first, total)

        result = self.get_wall_posts(domain=domain, count=1, offset=0)
        for offset in range(first, last, BATCH_SIZE):
            count = min(BATCH_SIZE, last - offset)
            result.extend(
                self.get_wall_posts(domain=domain, count=count, offset=offset)
            )
        return [
            post
            for post in result
            if start_date.timestamp() <= post.raw_date < end_date.timestamp()
        ]


if __name__ == "__main__":
    client = VKClient()
    for item in client.get_wall_posts_by_date_range(
        domain="amuroblgov",
        end_date=datetime.datetime.fromisoformat("2026-10-20"),
        start_date=datetime.datetime.fromisoformat("2026-10-01"),
    ):
        print("=" * 70)
        print(item.text)
