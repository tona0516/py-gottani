"""
X（旧Twitter）のブックマークから画像を一括ダウンロードするスクリプト

【必要なライブラリのインストール】
    uv sync
    uv run playwright install chromium

【使い方】
    1. uv run main.py
    2. このスクリプトを実行すると、最初にブラウザウィンドウが開きます。
    3. 初回のみ、表示されたブラウザでXに手動でログインしてください。
       ログイン後ターミナルに戻り Enter キーを押すと、ログイン情報が
       auth.json に保存されます（次回以降は自動でログインされます）。
    4. 自動でブックマークページに移動し、下にスクロールしながら
       画像を検出し、URLを収集します。
    5. 収集が終わると images/ フォルダに画像が保存されます。

【注意点】
    - 自分のアカウントの私的なバックアップ目的での利用を想定しています。
    - X の利用規約や仕様は変更される可能性があり、HTML構造が変わると
      動作しなくなることがあります。
    - 動画やGIFのサムネイル以外（動画本体）はこのスクリプトでは
      ダウンロードされません（静止画のみが対象です）。
    - ブックマーク数が多い場合は MAX_SCROLLS を増やしてください。
"""

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
from playwright.async_api import async_playwright

# ===== 設定（必要に応じて変更してください） =====
AUTH_FILE = "auth.json"  # ログイン情報の保存先
SAVE_DIR = "images"  # 画像の保存先フォルダ
BOOKMARKS_URL = "https://x.com/i/bookmarks"
MAX_SCROLLS = 1000  # 最大スクロール回数
SCROLL_WAIT_MS = 1500  # スクロール後の待機時間（ミリ秒）
NO_NEW_CONTENT_LIMIT = 6  # 新しい画像が見つからない状態がこの回数続いたら終了


def to_original_quality(url: str) -> str:
    """画像URLのクエリパラメータを変更し、最高画質（orig）で取得できるURLにする"""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["name"] = ["orig"]
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def filename_from_url(url: str, username: str) -> str:
    """画像URLから保存用のファイル名を作る（アカウント名をプレフィックスにする）"""
    parsed = urlparse(url)
    base = os.path.basename(parsed.path)  # 例: AbCdEfGh.jpg
    name, ext = os.path.splitext(base)
    if not ext:
        ext = ".jpg"

    # 念のためファイル名として使えない文字を除外
    safe_username = "".join(c for c in username if c.isalnum() or c in ("_", "-"))
    if not safe_username:
        safe_username = "unknown"

    return f"{safe_username}_{name}{ext}"


async def get_logged_in_context(playwright):
    """ログイン済みのブラウザコンテキストを用意する（保存済みセッションがあれば再利用）"""
    browser = await playwright.chromium.launch(headless=False)

    if os.path.exists(AUTH_FILE):
        context = await browser.new_context(storage_state=AUTH_FILE)
        print(f"保存済みのログイン情報（{AUTH_FILE}）を読み込みました。")
    else:
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://x.com/login")
        print("\nブラウザが開きました。手動でXにログインしてください。")
        input("ログインが完了したら、ここでEnterキーを押してください...")
        await context.storage_state(path=AUTH_FILE)
        print(
            f"ログイン情報を {AUTH_FILE} に保存しました。次回以降は自動ログインされます。"
        )

    return browser, context


async def collect_image_urls(page) -> set:
    """ブックマークページをスクロールしながら画像URLとアカウント名を収集する"""
    collected = set()
    no_new_count = 0

    await page.goto(BOOKMARKS_URL)
    await page.wait_for_timeout(3000)

    for i in range(MAX_SCROLLS):
        # ツイート本文内の画像とユーザー名（@アカウント名）を同時に取得する
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
                    
                    let imgs = article.querySelectorAll("img[src*='pbs.twimg.com/media']");
                    imgs.forEach(img => {
                        results.push({url: img.src, username: username});
                    });
                });
                return results;
            }""",
        )

        before_count = len(collected)
        for item in items:
            collected.add((to_original_quality(item["url"]), item["username"]))
        after_count = len(collected)

        print(f"[{i + 1}/{MAX_SCROLLS}] 収集済み画像数: {after_count}")

        no_new_count = 0 if after_count > before_count else no_new_count + 1
        if no_new_count >= NO_NEW_CONTENT_LIMIT:
            print("新しい画像が見つからなくなったため、収集を終了します。")
            break

        await page.mouse.wheel(0, 3000)
        await page.wait_for_timeout(SCROLL_WAIT_MS)

    return collected


def download_images(items: set, save_dir: str):
    """収集した画像URLをすべてダウンロードして保存する"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    items_sorted = sorted(list(items), key=lambda x: x[0])  # URLでソート

    for idx, (url, username) in enumerate(items_sorted, start=1):
        filename = filename_from_url(url, username)
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
    async with async_playwright() as playwright:
        browser, context = await get_logged_in_context(playwright)
        page = await context.new_page()

        try:
            items = await collect_image_urls(page)
        finally:
            await browser.close()

        print(f"\n合計 {len(items)} 件の画像を収集しました。ダウンロードを開始します。")
        download_images(items, SAVE_DIR)
        print("\n完了しました！ images フォルダを確認してください。")


if __name__ == "__main__":
    asyncio.run(main())
