import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

CHANNEL = "kpszsu"
POSTS = Path("data/raw/kpszsu_posts.csv")
POSTS_SOURCE = Path("data/raw/kpszsu_source.json")
PARTIAL = Path("data/raw/kpszsu_partial.csv")
SINCE = "2026-04-01"
PAUSE = 0.3


def fetch(before=None):
    url = f"https://t.me/s/{CHANNEL}" + (f"?before={before}" if before else "")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def parse(page):
    posts = []
    for block in re.split(r'(?=<div class="tgme_widget_message_wrap)', page)[1:]:
        post_id = re.search(rf'data-post="{CHANNEL}/(\d+)"', block)
        posted = re.search(r'class="tgme_widget_message_date"[^>]*><time datetime="([^"]+)"', block)
        if not post_id or not posted:
            continue
        text = re.search(r'<div class="tgme_widget_message_text js-message_text"[^>]*>(.*?)</div>', block, flags=re.S)
        meta = re.search(r'<span class="tgme_widget_message_meta">(.*?)</span>', block, flags=re.S)
        posts.append({
            "id": int(post_id.group(1)),
            "posted_at": posted.group(1),
            "edited": bool(meta and "edited" in meta.group(1)),
            "text": " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text.group(1))).split()) if text else "",
        })
    return posts


def download(since=SINCE):
    since = pd.Timestamp(since, tz="UTC")
    POSTS.parent.mkdir(parents=True, exist_ok=True)
    posts = pd.read_csv(PARTIAL, keep_default_na=False).to_dict("records") if PARTIAL.exists() else []
    before = min(p["id"] for p in posts) if posts else None
    pages = 0
    while True:
        page = parse(fetch(before))
        if not page:
            break
        posts.extend(page)
        pages += 1
        before = min(p["id"] for p in page)
        oldest = pd.Timestamp(min(p["posted_at"] for p in page))
        if pages % 50 == 0:
            pd.DataFrame(posts).to_csv(PARTIAL, index=False)
            print(pages, "pages, back to", oldest, flush=True)
        if oldest < since or before <= 1:
            break
        time.sleep(PAUSE)
    df = pd.DataFrame(posts).drop_duplicates("id").sort_values("id")
    df["posted_at"] = pd.to_datetime(df.posted_at, utc=True)
    df = df[df.posted_at >= since]
    df.to_csv(POSTS, index=False)
    PARTIAL.unlink(missing_ok=True)
    source = {
        "channel": f"https://t.me/s/{CHANNEL}",
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "posts": len(df),
        "first_post": df.posted_at.min().isoformat(),
        "last_post": df.posted_at.max().isoformat(),
        "edited_share": round(float(df.edited.mean()), 4),
    }
    POSTS_SOURCE.write_text(json.dumps(source, indent=2))
    return source


def load_posts(path=POSTS):
    df = pd.read_csv(path, keep_default_na=False)
    df["posted_at"] = pd.to_datetime(df.posted_at, utc=True)
    return df.sort_values("posted_at").reset_index(drop=True)


if __name__ == "__main__":
    print(json.dumps(download(*sys.argv[1:]), indent=2))
