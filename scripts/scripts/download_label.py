import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://data.taeo-dev.com/dataset/traffic/subset/label/"
SAVE_DIR = r"D:\traffic-fault-platform\data\subset_3k\label"

os.makedirs(SAVE_DIR, exist_ok=True)


def get_links(url: str):
    res = requests.get(url, timeout=30)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    links = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href in ("../", "./"):
            continue
        links.append(href)

    return links


def is_directory_link(href: str) -> bool:
    return href.endswith("/")


def is_json_link(href: str) -> bool:
    return href.endswith(".json")


def crawl_json_links(start_url: str):
    visited = set()
    json_urls = []

    def _crawl(url: str):
        if url in visited:
            return
        visited.add(url)

        print(f"[CRAWL] {url}")

        try:
            links = get_links(url)
        except Exception as e:
            print(f"[ERROR] 링크 조회 실패: {url} -> {e}")
            return

        for href in links:
            full_url = urljoin(url, href)

            if is_json_link(href):
                json_urls.append(full_url)
            elif is_directory_link(href):
                _crawl(full_url)

    _crawl(start_url)
    return json_urls


def make_local_filename(file_url: str, index: int) -> str:
    """
    URL path 기반으로 파일명 충돌을 피하기 위해 하위 경로를 파일명에 반영
    """
    parsed = urlparse(file_url)
    path = parsed.path  # /dataset/traffic/subset/label/xxx/yyy.json
    rel = path.split("/dataset/traffic/subset/label/")[-1]
    rel = rel.replace("/", "__")
    if not rel.endswith(".json"):
        rel = f"{index:06d}.json"
    return rel


def main():
    json_urls = crawl_json_links(BASE_URL)
    print(f"\n[INFO] 찾은 json 파일 수: {len(json_urls)}")

    if not json_urls:
        print("[WARN] json 링크를 찾지 못했습니다.")
        return

    for i, file_url in enumerate(json_urls, start=1):
        save_name = make_local_filename(file_url, i)
        save_path = os.path.join(SAVE_DIR, save_name)

        try:
            r = requests.get(file_url, timeout=60)
            r.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(r.content)

            if i % 100 == 0 or i == len(json_urls):
                print(f"[INFO] 다운로드 진행: {i}/{len(json_urls)}")

        except Exception as e:
            print(f"[ERROR] 다운로드 실패: {file_url} -> {e}")

    print("\n[DONE] 다운로드 완료")
    print("[SAVE_DIR]", SAVE_DIR)


if __name__ == "__main__":
    main()