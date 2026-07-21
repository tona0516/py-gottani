"""
X（旧Twitter）のブックマークから画像や動画を一括ダウンロードするスクリプト
"""

from pathlib import Path
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import threading
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote

import requests
from playwright.async_api import async_playwright


# ===== 設定（必要に応じて変更してください） =====
AUTH_FILE = "auth.json"  # ログイン情報の保存先
COOKIES_TXT = "cookies.txt"  # yt-dlp用一時クッキーファイル
SAVE_DIR = "images"  # 保存先フォルダ（画像・動画共用）
X_URL = "https://x.com"
BOOKMARKS_URL = f"{X_URL}/i/bookmarks"
MAX_SCROLLS = 1000  # 最大スクロール回数
SCROLL_WAIT_MS = 1500  # スクロール後の待機時間（ミリ秒）
NO_NEW_CONTENT_LIMIT = 6  # 新しいメディアが見つからない状態がこの回数続いたら終了


# ブックマークページからツイート情報（ユーザー名、画像URL、動画情報、ツイートURL）を抽出する JavaScript ロジック
EXTRACT_TWEETS_JS = """articles => {
    let results = [];
    articles.forEach(article => {
        let imgs = article.querySelectorAll("img[src*='pbs.twimg.com/media']");
        let imageUrls = Array.from(imgs).map(img => img.src);
        
        let videoUrls = [];
        let videos = article.querySelectorAll("video");
        videos.forEach(v => {
            if (v.src && !v.src.startsWith("blob:")) {
                videoUrls.push(v.src);
            }
            let sources = v.querySelectorAll("source");
            sources.forEach(s => {
                if (s.src && !s.src.startsWith("blob:")) {
                    videoUrls.push(s.src);
                }
            });
        });

        // 動画要素がツイート内に存在するかチェック
        let hasVideoElement = videos.length > 0 ||
            article.querySelector('[data-testid="videoPlayer"]') !== null ||
            article.querySelector('[data-testid="videoComponent"]') !== null ||
            article.querySelector('div[aria-label*="動画"]') !== null ||
            article.querySelector('div[aria-label*="Video"]') !== null;

        let tweetUrl = null;
        let timeEl = article.querySelector("time");
        if (timeEl) {
            let aEl = timeEl.closest("a");
            if (aEl && aEl.href) {
                tweetUrl = aEl.href;
            }
        }

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
            imageUrls: imageUrls,
            videoUrls: videoUrls,
            hasVideoElement: hasVideoElement,
            tweetUrl: tweetUrl
        });
    });
    return results;
}"""


def export_cookies_txt(auth_file: str, output_path: str) -> str | None:
    """auth.json から Netscape Cookie File 形式の cookies.txt を生成する"""
    if not os.path.exists(auth_file):
        return None
    try:
        with open(auth_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        cookies = data.get("cookies", [])
        if not cookies:
            return None

        lines = [
            "# Netscape HTTP Cookie File",
            "# http://curl.haxx.se/rfc/cookie_spec.html",
            "# This is a generated file!  Do not edit.",
            "",
        ]
        for c in cookies:
            domain = c.get("domain", "")
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            path = c.get("path", "/")
            secure = "TRUE" if c.get("secure", False) else "FALSE"
            expires = str(int(c.get("expires", 0)))
            name = c.get("name", "")
            value = c.get("value", "")
            lines.append(
                f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return output_path
    except Exception as e:
        print(f"cookies.txt の書き出し失敗: {e}")
        return None


def to_original_quality(url: str) -> str:
    """画像URLのクエリパラメータを変更し、最高画質（orig）で取得できるURLにする"""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["name"] = ["orig"]
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def filename_from_url(url: str, username: str) -> str:
    """画像情報から保存用のファイル名を作る (@アカウント名_ハッシュ.拡張子).

    Windows のパス長制限（260 文字）を考慮し、URLのMD5ハッシュを使って
    ファイル名が長くなりすぎないようにする。
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    fmt = query.get("format", ["jpg"])[0]
    ext = f".{fmt}"

    safe_username = "".join(c for c in username if c.isalnum() or c in ("_", "-"))
    if not safe_username:
        safe_username = "unknown"

    url_hash = hashlib.md5(url.encode()).hexdigest()

    return f"@{safe_username}_{url_hash}{ext}"


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


async def collect_media_urls(page, include_video: bool = False) -> tuple[dict, dict]:
    """ブックマークページをスクロールしながら画像および動画情報を収集する"""
    collected_images = {}  # url -> {username}
    collected_videos = {}  # key -> {username, tweet_url, direct_urls}
    no_new_count = 0

    await page.goto(BOOKMARKS_URL)
    await page.wait_for_timeout(3000)

    for i in range(MAX_SCROLLS):
        items = await page.eval_on_selector_all(
            "article",
            EXTRACT_TWEETS_JS,
        )

        before_img_count = len(collected_images)
        before_vid_count = len(collected_videos)

        for item in items:
            username = item["username"]
            image_urls = item.get("imageUrls", [])
            video_urls = item.get("videoUrls", [])
            has_video_element = item.get("hasVideoElement", False)
            tweet_url = item.get("tweetUrl")

            # 画像収集
            for url in image_urls:
                orig_url = to_original_quality(url)
                if orig_url not in collected_images:
                    collected_images[orig_url] = {"username": username}

            # 動画収集（動画要素が存在するか、直接動画URLがある場合のみ対象）
            if include_video:
                has_video = has_video_element or len(video_urls) > 0
                if has_video:
                    key = (
                        tweet_url
                        if tweet_url
                        else (video_urls[0] if video_urls else None)
                    )
                    if key and key not in collected_videos:
                        collected_videos[key] = {
                            "username": username,
                            "tweet_url": tweet_url,
                            "direct_urls": video_urls,
                        }

        after_img_count = len(collected_images)
        after_vid_count = len(collected_videos)

        status_msg = f"[{i + 1}/{MAX_SCROLLS}] 収集済み画像数: {after_img_count}"
        if include_video:
            status_msg += f", 動画数: {after_vid_count}"
        print(status_msg)

        has_new = (after_img_count > before_img_count) or (
            include_video and after_vid_count > before_vid_count
        )
        no_new_count = 0 if has_new else no_new_count + 1

        if no_new_count >= NO_NEW_CONTENT_LIMIT:
            print("新しいコンテンツが見つからなくなったため、収集を終了します。")
            break

        await page.mouse.wheel(0, 3000)
        await page.wait_for_timeout(SCROLL_WAIT_MS)

    return collected_images, collected_videos


def download_images(items: dict, save_dir: str, max_workers: int = 10):
    """収集した画像URLをすべてダウンロードして保存する"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    items_sorted = sorted(items.items(), key=lambda x: x[0])

    # スレッドセーフなファイル名セット（レースコンディション防止）
    existing_files_set = set(os.listdir(save_dir))
    existing_files_lock = threading.Lock()

    completed_count = 0
    counter_lock = threading.Lock()
    total = len(items_sorted)

    def _download_single(url, info):
        nonlocal completed_count
        username = info["username"]

        filename = filename_from_url(url, username)

        with existing_files_lock:
            if filename in existing_files_set:
                with counter_lock:
                    completed_count += 1
                    idx = completed_count
                print(f"[{idx}/{total}] スキップ（既存ファイル）: {filename}")
                return
            # ダウンロード予定として先に登録（他スレッドの重複を防ぐ）
            existing_files_set.add(filename)

        filepath = os.path.join(save_dir, filename)

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            with counter_lock:
                completed_count += 1
                idx = completed_count
            print(f"[{idx}/{total}] 画像保存完了: {filename}")
        except Exception as e:
            # ダウンロード失敗時はセットから除去して再試行可能にする
            with existing_files_lock:
                existing_files_set.discard(filename)
            with counter_lock:
                completed_count += 1
                idx = completed_count
            print(f"[{idx}/{total}] 画像失敗: {url} ({e})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for url, info in items_sorted:
            executor.submit(_download_single, url, info)


def download_videos(videos: dict, save_dir: str, max_workers: int = 5):
    """収集した動画情報をダウンロードして保存する（yt-dlpまたは直リンクを使用）"""
    if not videos:
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    yt_dlp_cmd = shutil.which("yt-dlp")

    # スレッドセーフなファイル名セット（レースコンディション防止）
    existing_files_set = set(os.listdir(save_dir))
    existing_files_lock = threading.Lock()

    # cookies.txt を書き出し
    cookies_file = export_cookies_txt(AUTH_FILE, COOKIES_TXT)

    completed_count = 0
    counter_lock = threading.Lock()
    total = len(videos)

    def _download_single_video(key, info):
        nonlocal completed_count
        username = info["username"]
        tweet_url = info["tweet_url"]
        direct_urls = info["direct_urls"]

        safe_username = (
            "".join(c for c in username if c.isalnum() or c in ("_", "-")) or "unknown"
        )

        # 既存ファイルのチェック (ツイートIDを含む動画ファイル)
        tweet_id = tweet_url.split("/")[-1] if tweet_url else quote(key, safe="")
        with existing_files_lock:
            existing_match = next(
                (f for f in existing_files_set if tweet_id in f), None
            )
        if existing_match:
            with counter_lock:
                completed_count += 1
                idx = completed_count
            print(f"[{idx}/{total}] スキップ（既存動画）: {existing_match}")
            return

        # 1. yt-dlp コマンドが使用可能な場合
        if yt_dlp_cmd and tweet_url:
            output_template = os.path.join(
                save_dir, f"@{safe_username}_{tweet_id}.%(ext)s"
            )
            cmd = [
                yt_dlp_cmd,
                "-o",
                output_template,
                "--no-mtime",
                "--quiet",
                "--no-warnings",
            ]
            if cookies_file and os.path.exists(cookies_file):
                cmd.extend(["--cookies", cookies_file])
            cmd.append(tweet_url)

            try:
                subprocess.run(cmd, check=True, timeout=120)
                with counter_lock:
                    completed_count += 1
                    idx = completed_count
                print(
                    f"[{idx}/{total}] 動画保存完了 (yt-dlp): @{safe_username}_{tweet_id}"
                )
                # 保存したファイルをセットに追加
                with existing_files_lock:
                    existing_files_set.add(f"@{safe_username}_{tweet_id}")
                return
            except Exception as e:
                print(f"yt-dlpでのダウンロード失敗 (フォールバック試行): {e}")

        # 2. 直リンクからのフォールバック（または yt-dlp がない場合）
        if direct_urls:
            for url in direct_urls:
                filename = f"@{safe_username}_{quote(url, safe='')}.mp4"
                filepath = os.path.join(save_dir, filename)

                if os.path.exists(filepath):
                    with counter_lock:
                        completed_count += 1
                        idx = completed_count
                    print(f"[{idx}/{total}] スキップ（既存動画）: {filename}")
                    return

                try:
                    resp = requests.get(url, headers=headers, timeout=60)
                    resp.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    with counter_lock:
                        completed_count += 1
                        idx = completed_count
                    print(f"[{idx}/{total}] 動画保存完了: {filename}")
                    with existing_files_lock:
                        existing_files_set.add(filename)
                    return
                except Exception as e:
                    print(f"直リンクからの動画保存失敗: {url} ({e})")

        # ダウンロード方法がない、あるいはすべて失敗した場合
        with counter_lock:
            completed_count += 1
            idx = completed_count
        if not yt_dlp_cmd:
            print(
                f"[{idx}/{total}] 動画スキップ: @{safe_username} (yt-dlp未インストールのため。必要に応じて `pip install yt-dlp` を行ってください)"
            )
        else:
            print(
                f"[{idx}/{total}] 動画保存失敗: @{safe_username} ({tweet_url or key})"
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, info in videos.items():
            executor.submit(_download_single_video, key, info)


async def main():
    parser = argparse.ArgumentParser(
        description="Xのブックマークから画像・動画を一括ダウンロードするスクリプト"
    )
    parser.add_argument(
        "--dir", default=SAVE_DIR, help=f"保存先フォルダ (デフォルト: {SAVE_DIR})"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="動画もダウンロード対象に含める",
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
            images, videos = await collect_media_urls(page, include_video=args.video)
        finally:
            await browser.close()

        print(
            f"\n合計 {len(images)} 件の画像を収集しました。ダウンロードを開始します。"
        )
        download_images(images, args.dir, args.workers)

        if args.video:
            print(
                f"\n合計 {len(videos)} 件の動画を収集しました。ダウンロードを開始します。"
            )
            download_videos(videos, args.dir, max(1, args.workers // 2))

        print(f"\n完了しました！ {args.dir} フォルダを確認してください。")


if __name__ == "__main__":
    asyncio.run(main())
