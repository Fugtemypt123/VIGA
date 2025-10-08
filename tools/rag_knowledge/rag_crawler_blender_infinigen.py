#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFE RAG Crawler & JSONL Generator (Blender Python 4.5 + Infinigen docs only; avoids binaries)

Key additions:
- URL extension denylist (e.g., .zip/.tar.gz/.pdf/.png/...)
- HEAD request for Content-Type + Content-Length check (skip non-docs or large files)
- Content-Type allowlist (text/html, text/plain, text/markdown, application/xhtml+xml)
- Streaming GET with byte cap (--max-bytes) to abort oversized responses
- Stricter link enqueue rules (same host + extension checks)

Usage:
  python rag_crawler_blender_infinigen.py \
    --out knowledge.jsonl --blender --infinigen \
    --max_pages 120 --delay 1.0 --max-bytes 2000000

Notes:
- We do NOT fetch papers; for Infinigen only official site + GitHub docs/*.md (converted to raw) are parsed.
- Respect your target sites; tune --delay and --max_pages as needed.
"""

import argparse, os, sys, time, json, re, uuid, datetime
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup

BLENDER_VERSION = "4.5"
TODAY = datetime.date.today().isoformat()

DEFAULT_ALLOWLIST = [
    # Blender
    "https://docs.blender.org/api/current/index.html",
    "https://docs.blender.org/api/current/info_quickstart.html",
    "https://docs.blender.org/api/current/info_overview.html",
    "https://docs.blender.org/api/current/info_api_reference.html",
    "https://docs.blender.org/api/current/bpy.types.html",
    # Infinigen website
    "https://infinigen.org/",
    "https://infinigen.org/docs/installation/intro",
    "https://infinigen.org/docs/category/installation-instructions",
    # Infinigen GitHub docs (HTML pages; we will map to raw for content where possible)
    "https://github.com/princeton-vl/infinigen",
    "https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md",
    "https://github.com/princeton-vl/infinigen/blob/main/docs/HelloWorld.md",
    "https://github.com/princeton-vl/infinigen/blob/main/docs/HelloRoom.md",
    "https://github.com/princeton-vl/infinigen/blob/main/docs/ConfiguringInfinigen.md",
    "https://github.com/princeton-vl/infinigen/blob/main/docs/ConfiguringCameras.md",
    "https://github.com/princeton-vl/infinigen/blob/main/docs/Exporting.md",
]

# --- Safety controls ---
EXT_DENY = {
    ".zip",".tar",".gz",".tgz",".bz2",".xz",".7z",
    ".pdf",
    ".png",".jpg",".jpeg",".gif",".svg",".webp",".ico",
    ".mp4",".mov",".avi",".mkv",".webm",
    ".mp3",".wav",".ogg",".flac",
    ".woff",".woff2",".ttf",".otf",
    ".exe",".msi",".dmg",".apk",".whl",
}
ALLOWED_CT_PREFIX = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
)
# Some servers send text/x-markdown
ALLOWED_CT_EXTRA = ("text/x-markdown",)

def has_denied_extension(url: str) -> bool:
    path = urlparse(url).path.lower()
    for ext in EXT_DENY:
        if path.endswith(ext):
            return True
    return False

def is_allowed(url: str) -> bool:
    if has_denied_extension(url):
        return False
    return (
        url.startswith("https://docs.blender.org/api/current/") or
        url.startswith("https://infinigen.org/") or
        url.startswith("https://github.com/princeton-vl/infinigen")
    )

def to_raw_github(url: str) -> str:
    # Convert GitHub blob URL to raw URL if possible
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url

class SkipFetch(Exception):
    pass

def safe_fetch(url: str, max_bytes: int = 2_000_000, timeout: int = 20) -> requests.Response:
    """
    HEAD check for Content-Type/Length -> skip non-docs/large files.
    Then GET with stream + byte cap. Returns a Response with ._text cached.
    """
    headers = {
        "User-Agent": "RAG-DOC-Crawler/1.1 (+research; contact: you@example.com)",
        "Accept": "text/html,application/xhtml+xml,text/markdown,text/plain;q=0.9,*/*;q=0.1",
    }

    # Quick deny by extension
    if has_denied_extension(url):
        raise SkipFetch("denied by extension")

    # HEAD
    try:
        h = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        ct = h.headers.get("Content-Type","").split(";")[0].strip().lower()
        cl = h.headers.get("Content-Length")
        if ct and not (ct.startswith(ALLOWED_CT_PREFIX) or ct in ALLOWED_CT_EXTRA):
            raise SkipFetch(f"content-type not allowed: {ct}")
        if cl is not None:
            try:
                size = int(cl)
                if size > max_bytes:
                    raise SkipFetch(f"content-length too large: {size} > {max_bytes}")
            except ValueError:
                pass
    except requests.RequestException:
        # Some servers do not support HEAD; continue to GET with streaming
        pass

    # GET streaming with byte cap
    r = requests.get(url, timeout=timeout, headers=headers, stream=True)
    r.raise_for_status()
    ct = r.headers.get("Content-Type","").split(";")[0].strip().lower()
    if ct and not (ct.startswith(ALLOWED_CT_PREFIX) or ct in ALLOWED_CT_EXTRA):
        r.close()
        raise SkipFetch(f"content-type not allowed on GET: {ct}")

    buf = bytearray()
    chunk_size = 16384
    for chunk in r.iter_content(chunk_size=chunk_size):
        if chunk:
            buf.extend(chunk)
            if len(buf) > max_bytes:
                r.close()
                raise SkipFetch(f"response too large > {max_bytes} bytes")
    enc = r.encoding or "utf-8"
    try:
        text = buf.decode(enc, errors="ignore")
    except Exception:
        text = buf.decode("utf-8", errors="ignore")
    r._text = text
    return r

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1200]

def parse_blender_html(url: str, html: str):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    page_title = title.get_text(strip=True) if title else "Blender Python API"
    records = []
    for head in soup.select("h2, h3"):
        section_title = head.get_text(" ", strip=True)
        section_path = [page_title, section_title]
        content_parts = []
        for sib in head.find_all_next():
            if sib.name in ["h1", "h2"] and sib is not head:
                break
            if sib.name in ["p", "ul", "ol", "pre", "table", "code"]:
                content_parts.append(sib.get_text(" ", strip=True))
        content = clean_text(" ".join(content_parts))
        code_refs = sorted(set([tok for tok in re.findall(r"\b[bB]py\.[\w\.]+", content)]))[:10]
        tags = ["blender","api"]
        if "bpy.types" in content or "bpy.types" in section_title:
            tags.append("bpy.types")
        if len(content) >= 40:
            records.append({
                "id": str(uuid.uuid4()),
                "domain": "blender",
                "title": section_title or page_title,
                "url": url,
                "version": BLENDER_VERSION,
                "section_path": section_path,
                "tags": tags,
                "updated": TODAY,
                "content_summary": content,
                "code_refs": code_refs,
                "source_type": "official-docs"
            })
    if not records:
        content = clean_text(soup.get_text(" ", strip=True))
        if content:
            records.append({
                "id": str(uuid.uuid4()),
                "domain": "blender",
                "title": page_title,
                "url": url,
                "version": BLENDER_VERSION,
                "section_path": [page_title],
                "tags": ["blender","api"],
                "updated": TODAY,
                "content_summary": content[:1200],
                "code_refs": [],
                "source_type": "official-docs"
            })
    return records

def parse_infinigen_html(url: str, html: str):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find(["h1","title"])
    page_title = title.get_text(strip=True) if title else "Infinigen Docs"
    records = []
    for head in soup.select("h2, h3"):
        section_title = head.get_text(" ", strip=True)
        section_path = [page_title, section_title]
        content_parts = []
        for sib in head.find_all_next():
            if sib.name in ["h1", "h2"] and sib is not head:
                break
            if sib.name in ["p", "ul", "ol", "pre", "code", "table"]:
                content_parts.append(sib.get_text(" ", strip=True))
        content = clean_text(" ".join(content_parts))
        if len(content) < 40:
            continue
        records.append({
            "id": str(uuid.uuid4()),
            "domain": "infinigen",
            "title": section_title or page_title,
            "url": url,
            "version": "2024-site",
            "section_path": section_path,
            "tags": ["infinigen","docs"],
            "updated": TODAY,
            "content_summary": content,
            "code_refs": [],
            "source_type": "official-docs"
        })
    if not records:
        content = clean_text(soup.get_text(" ", strip=True))
        if content:
            records.append({
                "id": str(uuid.uuid4()),
                "domain": "infinigen",
                "title": page_title,
                "url": url,
                "version": "2024-site",
                "section_path": [page_title],
                "tags": ["infinigen","docs"],
                "updated": TODAY,
                "content_summary": content[:1200],
                "code_refs": [],
                "source_type": "official-docs"
            })
    return records

def parse_github_markdown(url: str, text: str):
    lines = text.splitlines()
    page_title = None
    records = []
    current_title = ""
    current_buf = []
    def flush():
        if not current_title and not current_buf:
            return
        content = clean_text(" ".join(current_buf))
        if len(content) < 40:
            return
        records.append({
            "id": str(uuid.uuid4()),
            "domain": "infinigen",
            "title": current_title or (page_title or "Infinigen Docs"),
            "url": url,
            "version": "repo-main",
            "section_path": [page_title or "Infinigen Docs", current_title] if current_title else [page_title or "Infinigen Docs"],
            "tags": ["infinigen","docs"],
            "updated": TODAY,
            "content_summary": content,
            "code_refs": [],
            "source_type": "official-docs"
        })
    for ln in lines:
        if ln.startswith("# "):
            page_title = ln[2:].strip()
            continue
        if ln.startswith("## "):
            flush()
            current_title = ln[3:].strip()
            current_buf = []
        elif ln.startswith("### "):
            flush()
            current_title = ln[4:].strip()
            current_buf = []
        else:
            if ln.strip().startswith("```"):
                continue
            current_buf.append(ln.strip())
    flush()
    return records

def crawl(seeds, out_path, delay=1.0, max_pages=50, max_bytes=2_000_000, enable_blender=True, enable_infinigen=True):
    seen = set()
    q = deque(seeds)
    total = 0
    out = []
    while q and total < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)
        if not is_allowed(url):
            continue
        try:
            resp = safe_fetch(url, max_bytes=max_bytes)
        except SkipFetch as e:
            print(f"[skip] {url} -> {e}")
            continue
        except Exception as e:
            print(f"[error] {url} -> {e}")
            continue

        text = getattr(resp, "_text", None)
        if text is None:
            text = ""   # 或者再次按需获取，但不要直接触发 resp.text
        total += 1
        print(f"[{total}/{max_pages}] fetched: {url} ({len(text)} bytes)")

        # Parse and enqueue links (same host only, and pass extension filter)
        try:
            if resp.headers.get("Content-Type","").lower().startswith(("text/html","application/xhtml+xml")):
                soup = BeautifulSoup(text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"])
                    href = href.split("#")[0]
                    if is_allowed(href) and href not in seen and not has_denied_extension(href):
                        if "github.com" in urlparse(href).netloc:
                            if "/princeton-vl/infinigen" not in href:
                                continue
                            if not ("/docs/" in href or href.rstrip("/").endswith("infinigen")):
                                continue
                        q.append(href)
        except Exception:
            pass

        # Route to parser
        domain = urlparse(url).netloc
        try:
            if enable_blender and domain == "docs.blender.org":
                out.extend(parse_blender_html(url, text))
            elif enable_infinigen and domain == "infinigen.org":
                out.extend(parse_infinigen_html(url, text))
            elif enable_infinigen and domain == "github.com" and "/princeton-vl/infinigen" in url:
                raw_url = to_raw_github(url)
                try:
                    raw_resp = safe_fetch(raw_url, max_bytes=max_bytes)
                    raw_text = getattr(raw_resp, "_text", raw_resp.text)
                except Exception:
                    raw_text = text
                out.extend(parse_github_markdown(url, raw_text))
        except Exception as e:
            print(f"[parse-error] {url} -> {e}")

        time.sleep(delay)

    # Deduplicate
    dedup = {}
    for r in out:
        key = (r["url"], r["title"], tuple(r["section_path"]))
        if key not in dedup:
            dedup[key] = r
    with open(out_path, "w", encoding="utf-8") as f:
        for r in dedup.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(dedup)} records to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="knowledge.jsonl")
    ap.add_argument("--max_pages", type=int, default=80)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-bytes", type=int, default=2_000_000, help="Max bytes per response (skip larger)")
    ap.add_argument("--blender", action="store_true", help="crawl Blender docs")
    ap.add_argument("--infinigen", action="store_true", help="crawl Infinigen docs")
    args = ap.parse_args()

    seeds = DEFAULT_ALLOWLIST
    if not (args.blender or args.infinigen):
        args.blender = True
        args.infinigen = True

    crawl(
        seeds=seeds,
        out_path=args.out,
        delay=args.delay,
        max_pages=args.max_pages,
        max_bytes=args.max_bytes,
        enable_blender=args.blender,
        enable_infinigen=args.infinigen,
    )

if __name__ == "__main__":
    main()
