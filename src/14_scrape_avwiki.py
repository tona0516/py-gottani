import argparse
import html
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def create_ssl_context() -> ssl.SSLContext:
    """SSL証明書エラーを防ぐためのSSLコンテキストを作成します。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_page_html(
    url: str,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    timeout: float = 10.0,
    ssl_context: ssl.SSLContext | None = None,
) -> str | None:
    """指定されたURLのHTMLコンテンツを取得します。

    Args:
        url: 取得対象のURL。
        user_agent: HTTPリクエストヘッダーに設定するUser-Agent。
        timeout: タイムアウト秒数。
        ssl_context: SSL検証用コンテキスト。

    Returns:
        取得したHTML文字列。失敗時は None。
    """
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(
            req, timeout=timeout, context=ssl_context
        ) as response:
            return response.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        sys.stderr.write(
            f"[警告] HTTPエラーが発生しました ({url}): ステータスコード {e.code}\n"
        )
        return None
    except urllib.error.URLError as e:
        sys.stderr.write(f"[警告] URL接続エラーが発生しました ({url}): {e.reason}\n")
        return None
    except Exception as e:
        sys.stderr.write(
            f"[警告] リクエスト中に予期しないエラーが発生しました ({url}): {e}\n"
        )
        return None


def extract_actresses(content: str) -> list[str]:
    """HTMLから出演女優名を抽出します。

    Args:
        content: HTML文字列。

    Returns:
        抽出された女優名のリスト。
    """
    actresses = []

    # 1. <dl class="dltable"> から "AV女優名" を探索
    dl_match = re.search(
        r'<dl class="dltable">(.*?)</dl>', content, re.DOTALL | re.IGNORECASE
    )
    if dl_match:
        dl_content = dl_match.group(1)
        m = re.search(
            r"<dt[^>]*>\s*AV女優名\s*</dt>\s*<dd[^>]*>(.*?)</dd>",
            dl_content,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            dd_content = m.group(1)
            a_tags = re.findall(
                r"<a[^>]*>(.*?)</a>", dd_content, re.DOTALL | re.IGNORECASE
            )
            if a_tags:
                for a in a_tags:
                    name = html.unescape(re.sub(r"<[^>]+>", "", a)).strip()
                    if name:
                        actresses.append(name)
            else:
                raw = html.unescape(re.sub(r"<[^>]+>", "", dd_content)).strip()
                if raw:
                    actresses.append(raw)

    # 2. dlから取得できなかった場合のフォールバック: タグリンクから抽出
    if not actresses:
        meta_tags = re.findall(
            r'<a href="https?://av-wiki\.net/av-actress/[^"]*"[^>]*>(.*?)</a>',
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if meta_tags:
            seen = set()
            for tag in meta_tags:
                name = html.unescape(re.sub(r"<[^>]+>", "", tag)).strip()
                if name and name not in seen:
                    seen.add(name)
                    actresses.append(name)

    return actresses


def extract_title(content: str) -> str:
    """HTMLから作品タイトルを抽出します。

    Args:
        content: HTML文字列。

    Returns:
        抽出された作品タイトル。
    """
    title = ""

    # 1. <div class="blockquote-like"><p>【型番】タイトル</p></div> から抽出
    bq_match = re.search(
        r'<div class="blockquote-like">\s*<p>(.*?)</p>',
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if bq_match:
        raw_bq = html.unescape(re.sub(r"<[^>]+>", "", bq_match.group(1))).strip()
        # 先頭の 【型番】 を除去
        title = re.sub(r"^【[^】]+】\s*", "", raw_bq).strip()

    # 2. 取得できなかった場合のフォールバック: <h1 class="entry-title"> から抽出
    if not title:
        h1_match = re.search(
            r'<h1 class="entry-title">(.*?)</h1>',
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if h1_match:
            raw_h1 = h1_match.group(1)
            # <span class="entry-subtitle">...</span> を除去
            raw_h1 = re.sub(
                r'<span class="entry-subtitle">.*?</span>',
                "",
                raw_h1,
                flags=re.DOTALL | re.IGNORECASE,
            )
            raw_h1 = html.unescape(re.sub(r"<[^>]+>", "", raw_h1)).strip()
            # 末尾の "に出てるAV女優名まとめ" などを除去
            raw_h1 = re.sub(r"に出てるAV女優.*$", "", raw_h1).strip()
            title = raw_h1

    # 連続する空白や改行を1つのスペースに正規化
    title = " ".join(title.split())
    return title


def scrape_item(
    code: str,
    timeout: float = 10.0,
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[str, str, str] | None:
    """指定された型番の情報をav-wiki.netから取得します。

    Args:
        code: 作品型番 (例: "LULU-427")。
        timeout: タイムアウト秒数。
        user_agent: User-Agentヘッダー。
        ssl_context: SSL検証用コンテキスト。

    Returns:
        (女優名, 型番, タイトル名) のタプル。情報が取得できない場合は None。
    """
    clean_code = code.strip()
    if not clean_code:
        return None

    slug = clean_code.lower()
    url = f"https://av-wiki.net/{urllib.parse.quote(slug)}/"

    content = fetch_page_html(
        url,
        user_agent=user_agent,
        timeout=timeout,
        ssl_context=ssl_context,
    )
    if not content:
        return None

    actresses = extract_actresses(content)
    title = extract_title(content)

    actress_str = ", ".join(actresses) if actresses else "不明"
    title_str = title if title else "タイトル不明"

    return actress_str, clean_code, title_str


def main() -> None:
    # Windows環境での標準入出力の文字化けを防ぐ設定
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="av-wiki.netから型番リストに基づいて女優名・型番・タイトル名をスクレイピングします。"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="型番が1行ずつ記載されたテキストファイルのパス（省略時は標準入力から読み込みます）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="結果を保存するテキストファイルのパス（省略時は標準出力のみ）",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=1.0,
        help="各リクエスト間の待機秒数 (デフォルト: 1.0秒)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=10.0,
        help="HTTPリクエストのタイムアウト秒数 (デフォルト: 10.0秒)",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        help="HTTPリクエスト送信時のUser-Agent",
    )

    args = parser.parse_args()

    # 入力ソースの読み込み
    lines: list[str] = []
    if args.input and args.input != "-":
        input_path = Path(args.input)
        if not input_path.is_file():
            sys.stderr.write(f"[エラー] 入力ファイルが見つかりません: {input_path}\n")
            sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        if sys.stdin.isatty():
            sys.stderr.write("型番を入力してください（Ctrl+Z / Ctrl+D で終了）:\n")
        lines = sys.stdin.readlines()

    codes = [
        line.strip() for line in lines if line.strip() and not line.startswith("#")
    ]
    if not codes:
        sys.stderr.write("[情報] 処理対象の型番がありません。\n")
        return

    ssl_context = create_ssl_context()
    results: list[str] = []

    for idx, code in enumerate(codes):
        info = scrape_item(
            code=code,
            timeout=args.timeout,
            user_agent=args.user_agent,
            ssl_context=ssl_context,
        )

        if info:
            actress, item_code, title = info
            line_out = f"{actress} {item_code} {title}"
            print(line_out, flush=True)
            results.append(line_out)

        # 最後のアイテムでなければ待機時間を挟む
        if idx < len(codes) - 1 and args.delay > 0:
            time.sleep(args.delay)

    # 出力ファイルへの保存
    if args.output and results:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r + "\n")
        sys.stderr.write(
            f"[情報] {len(results)} 件の結果を {output_path} に保存しました。\n"
        )


if __name__ == "__main__":
    main()
