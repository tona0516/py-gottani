"""
X（旧Twitter）のブックマークから画像を一括ダウンロードするスクリプト
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote

import requests
from playwright.async_api import async_playwright

# ===== 設定（必要に応じて変更してください） =====
AUTH_FILE = "auth.json"  # ログイン情報の保存先
SAVE_DIR = "images"  # 画像の保存先フォルダ
X_URL = "https://x.com"
BOOKMARKS_URL = f"{X_URL}/i/bookmarks"
MAX_SCROLLS = 1000  # 最大スクロール回数
SCROLL_WAIT_MS = 1500  # スクロール後の待機時間（ミリ秒）
NO_NEW_CONTENT_LIMIT = 6  # 新しい画像が見つからない状態がこの回数続いたら終了

# ブックマークページからツイート情報（ユーザー名と画像URLリスト）を抽出する JavaScript ロジック
EXTRACT_TWEETS_JS = """articles => {
    let results = [];
    articles.forEach(article => {
        let imgs = article.querySelectorAll("img[src*='pbs.twimg.com/media']");
        let imageUrls = Array.from(imgs).map(img => img.src);
        
        if (imageUrls.length === 0) return;
        
        let username = 'unknown';
        let userNameEl = article.querySelector('[data-testid="User-Name"]');
        if (userNameEl) {
            let spans = userNameEl.querySelectorAll('span');
            for (let span of spans) {
                let text = span.textContent.trim();
                if (!text.startsWith('@') || text.length <= 1) continue;
                username = text.substring(1);
                break;
            }
        }
        
        results.push({
            username: username,
            urls: imageUrls
        });
    });
    return results;
}"""


def to_original_quality(url: str) -> str:
    """画像URLのクエリパラメータを変更し、最高画質（orig）で取得できるURLにする"""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["name"] = ["orig"]
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def filename_from_url(url: str, username: str) -> str:
    """画像情報から保存用のファイル名を作る (@アカウント名_URLエンコードされた画像URL.拡張子)"""
    # クエリパラメータから拡張子を取得
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    fmt = query.get("format", ["jpg"])[0]
    ext = f".{fmt}"

    # 念のためファイル名として使えない文字を除外
    safe_username = "".join(c for c in username if c.isalnum() or c in ("_", "-"))
    if not safe_username:
        safe_username = "unknown"

    # URLエンコードされた画像URL
    encoded_url = quote(url, safe="")

    return f"@{safe_username}_{encoded_url}{ext}"


async def get_logged_in_context(playwright):
    """ログイン済みのブラウザコンテキストを用意する（保存済みセッションがあれば再利用）"""
    browser = await playwright.chromium.launch(headless=False)

    if os.path.exists(AUTH_FILE):
        context = await browser.new_context(storage_state=AUTH_FILE)
        print(f"保存済みのログイン情報（{AUTH_FILE}）を読み込みました。")
    else:
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(X_URL)
        print("\nブラウザが開きました。手動でXにログインしてください。")
        input("ログインが完了したら、ここでEnterキーを押してください...")
        await context.storage_state(path=AUTH_FILE)
        print(
            f"ログイン情報を {AUTH_FILE} に保存しました。次回以降は自動ログインされます。"
        )

    return browser, context


async def collect_image_urls(page) -> dict:
    """ブックマークページをスクロールしながら画像情報を収集する"""
    collected = {}  # url -> {username}
    no_new_count = 0

    await page.goto(BOOKMARKS_URL)
    await page.wait_for_timeout(3000)

    for i in range(MAX_SCROLLS):
        # ツイート本文内の画像、ユーザー名（@アカウント名）を取得する
        items = await page.eval_on_selector_all(
            "article",
            EXTRACT_TWEETS_JS,
        )

        before_count = len(collected)
        for item in items:
            username = item["username"]
            urls = item["urls"]

            for url in urls:
                orig_url = to_original_quality(url)
                if orig_url in collected:
                    continue
                collected[orig_url] = {
                    "username": username,
                }
        after_count = len(collected)

        print(f"[{i + 1}/{MAX_SCROLLS}] 収集済み画像数: {after_count}")

        no_new_count = 0 if after_count > before_count else no_new_count + 1
        if no_new_count >= NO_NEW_CONTENT_LIMIT:
            print("新しい画像が見つからなくなったため、収集を終了します。")
            break

        await page.mouse.wheel(0, 3000)
        await page.wait_for_timeout(SCROLL_WAIT_MS)

    return collected


def download_images(items: dict, save_dir: str, max_workers: int = 10):
    """収集した画像URLをすべてダウンロードして保存する"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    items_sorted = sorted(items.items(), key=lambda x: x[0])  # URLでソート

    # 保存先ディレクトリ内の既存のファイルリストを取得
    existing_files = os.listdir(save_dir)

    completed_count = 0
    counter_lock = threading.Lock()
    total = len(items_sorted)

    def _download_single(url, info):
        nonlocal completed_count
        username = info["username"]

        # 既に同じ画像URLを持つファイルが保存先に存在するかチェック
        encoded_url = quote(url, safe="")
        existing_filename = next((f for f in existing_files if encoded_url in f), None)

        if existing_filename:
            with counter_lock:
                completed_count += 1
                idx = completed_count
            print(f"[{idx}/{total}] スキップ（既存画像URL）: {existing_filename}")
            return

        filename = filename_from_url(url, username)
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            with counter_lock:
                completed_count += 1
                idx = completed_count
            print(f"[{idx}/{total}] スキップ（既存ファイル名）: {filename}")
            return

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            with counter_lock:
                completed_count += 1
                idx = completed_count
            print(f"[{idx}/{total}] 保存完了: {filename}")
        except Exception as e:
            with counter_lock:
                completed_count += 1
                idx = completed_count
            print(f"[{idx}/{total}] 失敗: {url} ({e})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for url, info in items_sorted:
            executor.submit(_download_single, url, info)


async def main():
    parser = argparse.ArgumentParser(
        description="Xのブックマークから画像を一括ダウンロードするスクリプト"
    )
    parser.add_argument(
        "--dir", default=SAVE_DIR, help=f"画像の保存先フォルダ (デフォルト: {SAVE_DIR})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="並列ダウンロードの接続数 (デフォルト: 10)",
    )
    args = parser.parse_args()

    async with async_playwright() as playwright:
        browser, context = await get_logged_in_context(playwright)
        page = await context.new_page()

        try:
            items = await collect_image_urls(page)
        finally:
            await browser.close()

        print(f"\n合計 {len(items)} 件の画像を収集しました。ダウンロードを開始します。")
        download_images(items, args.dir, args.workers)
        print(f"\n完了しました！ {args.dir} フォルダを確認してください。")


if __name__ == "__main__":
    asyncio.run(main())
