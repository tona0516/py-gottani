"""
X（旧Twitter）のブックマークから画像を一括ダウンロードするスクリプト
"""

import argparse
import asyncio
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

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


def format_datetime_jst(dt_str: str | None) -> str:
    """ISO 8601 (UTC) 文字列を JST に変換し、YYYY-MM-DD-SS 形式で返す"""
    if not dt_str:
        # 取得できなかった場合は現在日時を使用
        dt = datetime.now()
        return dt.strftime("%Y-%m-%d-%S")

    try:
        # 例: "2026-06-27T10:50:32.000Z" -> "2026-06-27T10:50:32.000+00:00"
        clean_str = dt_str.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(clean_str)
        # JST (UTC+9) に変換
        jst = timezone(timedelta(hours=9))
        dt_jst = dt_utc.astimezone(jst)
        return dt_jst.strftime("%Y-%m-%d-%S")
    except Exception:
        # 何らかの理由でパースに失敗した場合の簡易フォールバック
        try:
            # 簡易的に文字列から抽出
            date_part = dt_str[:10]
            sec_part = dt_str[17:19]
            return f"{date_part}-{sec_part}"
        except Exception:
            return "unknown-date"


def to_original_quality(url: str) -> str:
    """画像URLのクエリパラメータを変更し、最高画質（orig）で取得できるURLにする"""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["name"] = ["orig"]
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def filename_from_url(
    url: str, username: str, datetime_str: str | None, index: int
) -> str:
    """画像情報から保存用のファイル名を作る (@アカウント名_YYYY-MM-DD-SS_インデックス.拡張子)"""
    parsed = urlparse(url)
    base = os.path.basename(parsed.path)  # 例: AbCdEfGh.jpg
    _, ext = os.path.splitext(base)
    if not ext:
        ext = ".jpg"

    # 念のためファイル名として使えない文字を除外
    safe_username = "".join(c for c in username if c.isalnum() or c in ("_", "-"))
    if not safe_username:
        safe_username = "unknown"

    # 日付フォーマットの生成
    date_str = format_datetime_jst(datetime_str)

    return f"@{safe_username}_{date_str}_{index}{ext}"


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
    collected = {}  # url -> {username, datetime, index}
    no_new_count = 0

    await page.goto(BOOKMARKS_URL)
    await page.wait_for_timeout(3000)

    for i in range(MAX_SCROLLS):
        # ツイート本文内の画像、ユーザー名（@アカウント名）、投稿日時を取得する
        items = await page.eval_on_selector_all(
            "article",
            """articles => {
                let results = [];
                articles.forEach(article => {
                    let username = 'unknown';
                    let spans = article.querySelectorAll('span');
                    for (let span of spans) {
                        let text = span.textContent.trim();
                        if (text.startsWith('@') && text.length > 1) {
                            username = text.substring(1);
                            break;
                        }
                    }
                    
                    let timeEl = article.querySelector('time');
                    let datetime = timeEl ? timeEl.getAttribute('datetime') : null;
                    
                    let imgs = article.querySelectorAll("img[src*='pbs.twimg.com/media']");
                    let imageUrls = Array.from(imgs).map(img => img.src);
                    
                    if (imageUrls.length > 0) {
                        results.push({
                            username: username,
                            datetime: datetime,
                            urls: imageUrls
                        });
                    }
                });
                return results;
            }""",
        )

        before_count = len(collected)
        for item in items:
            username = item["username"]
            datetime_str = item["datetime"]
            urls = item["urls"]

            for idx, url in enumerate(urls, start=1):
                orig_url = to_original_quality(url)
                if orig_url not in collected:
                    collected[orig_url] = {
                        "username": username,
                        "datetime": datetime_str,
                        "index": idx,
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


def download_images(items: dict, save_dir: str):
    """収集した画像URLをすべてダウンロードして保存する"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    items_sorted = sorted(items.items(), key=lambda x: x[0])  # URLでソート

    for idx, (url, info) in enumerate(items_sorted, start=1):
        username = info["username"]
        datetime_str = info["datetime"]
        img_index = info["index"]

        filename = filename_from_url(url, username, datetime_str, img_index)
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            print(f"[{idx}/{len(items_sorted)}] スキップ（既存）: {filename}")
            continue

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"[{idx}/{len(items_sorted)}] 保存完了: {filename}")
        except Exception as e:
            print(f"[{idx}/{len(items_sorted)}] 失敗: {url} ({e})")


async def main():
    parser = argparse.ArgumentParser(
        description="Xのブックマークから画像を一括ダウンロードするスクリプト"
    )
    parser.add_argument(
        "--dir", default=SAVE_DIR, help=f"画像の保存先フォルダ (デフォルト: {SAVE_DIR})"
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
        download_images(items, args.dir)
        print(f"\n完了しました！ {args.dir} フォルダを確認してください。")


if __name__ == "__main__":
    asyncio.run(main())
