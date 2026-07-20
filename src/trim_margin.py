"""
フォルダ内にある画像の余白を自動でトリミングするスクリプト。

Pillowの機能を用いて、画像四辺の単色（またはほぼ単色）な余白を検出し、
コンテンツ部分だけを切り出して保存します。
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def detect_trim_box(img: Image.Image, tolerance: int) -> tuple | None:
    """
    画像の余白領域を検出し、トリミング後のバウンディングボックスを返す。

    四隅のピクセル色を背景色として推定し、tolerance の範囲内の色を
    余白とみなす。コンテンツが存在しない場合は None を返す。

    Args:
        img: 対象画像
        tolerance: 背景色との許容差（0〜255）。大きいほど余白として判定しやすくなる。

    Returns:
        (left, upper, right, lower) のバウンディングボックス、またはNone
    """
    # RGBAに統一して処理する
    if img.mode != "RGBA":
        img_rgba = img.convert("RGBA")
    else:
        img_rgba = img

    width, height = img_rgba.size
    pixels = img_rgba.load()

    # 四隅のピクセルから背景色を推定（最頻値を採用）
    corners = [
        pixels[0, 0],
        pixels[width - 1, 0],
        pixels[0, height - 1],
        pixels[width - 1, height - 1],
    ]
    bg_color = max(set(corners), key=corners.count)

    def is_background(pixel: tuple) -> bool:
        """ピクセルが背景色かどうかを判定する。"""
        return all(abs(int(pixel[c]) - int(bg_color[c])) <= tolerance for c in range(3))

    # 上辺のトリミング位置を探索
    top = 0
    for y in range(height):
        if not all(is_background(pixels[x, y]) for x in range(width)):
            top = y
            break
    else:
        return None  # 全ピクセルが背景色 → トリミング不要（空白画像）

    # 下辺のトリミング位置を探索
    bottom = height
    for y in range(height - 1, -1, -1):
        if not all(is_background(pixels[x, y]) for x in range(width)):
            bottom = y + 1
            break

    # 左辺のトリミング位置を探索
    left = 0
    for x in range(width):
        if not all(is_background(pixels[x, y]) for y in range(height)):
            left = x
            break

    # 右辺のトリミング位置を探索
    right = width
    for x in range(width - 1, -1, -1):
        if not all(is_background(pixels[x, y]) for y in range(height)):
            right = x + 1
            break

    if left >= right or top >= bottom:
        return None

    return (left, top, right, bottom)


def trim_image(
    input_path: str,
    output_path: str,
    tolerance: int,
    padding: int,
) -> bool:
    """
    1枚の画像を余白トリミングして保存する。

    Args:
        input_path: 入力画像のパス
        output_path: 出力画像のパス
        tolerance: 背景色との許容差
        padding: トリミング後に追加する余白ピクセル数

    Returns:
        成功した場合は True、スキップまたは失敗の場合は False
    """
    try:
        with Image.open(input_path) as img:
            original_mode = img.mode
            original_size = img.size

            bbox = detect_trim_box(img, tolerance)

            if bbox is None:
                print(f"  スキップ: {os.path.basename(input_path)} （コンテンツが検出できませんでした）")
                return False

            # padding を適用（画像の境界をはみ出さないようにクランプ）
            width, height = original_size
            left = max(0, bbox[0] - padding)
            top = max(0, bbox[1] - padding)
            right = min(width, bbox[2] + padding)
            bottom = min(height, bbox[3] + padding)

            trimmed = img.crop((left, top, right, bottom))

            # 元のモードに戻して保存（JPEG はアルファチャンネルを持てないため RGB に変換）
            if original_mode in ("RGBA", "LA", "PA") and output_path.lower().endswith(
                (".jpg", ".jpeg")
            ):
                trimmed = trimmed.convert("RGB")
            elif trimmed.mode != original_mode:
                trimmed = trimmed.convert(original_mode)

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            trimmed.save(output_path)

        reduction_w = original_size[0] - (right - left)
        reduction_h = original_size[1] - (bottom - top)
        print(
            f"  完了: {os.path.basename(input_path)} "
            f"({original_size[0]}x{original_size[1]} → {right - left}x{bottom - top}, "
            f"削減: {reduction_w}x{reduction_h}px)"
        )
        return True

    except Exception as e:
        print(f"  エラー: {os.path.basename(input_path)} の処理に失敗しました: {e}", file=sys.stderr)
        return False


def collect_image_files(directory: str) -> list:
    """
    指定ディレクトリ直下からサポートされている拡張子の画像ファイルをスキャンし、
    (ファイルパス, ファイル名) のリストを返す。

    Args:
        directory: スキャン対象のディレクトリパス

    Returns:
        (ファイルパス, ファイル名) のリスト
    """
    files = []
    try:
        for filename in sorted(os.listdir(directory)):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files.append((path, filename))
    except Exception as e:
        print(f"ディレクトリのスキャン中にエラーが発生しました: {e}", file=sys.stderr)
    return files


def resolve_output_path(input_path: str, input_dir: str, output_dir: str) -> str:
    """
    入力パスに対応する出力パスを計算する。

    Args:
        input_path: 入力ファイルの絶対パス
        input_dir: 入力ディレクトリの絶対パス
        output_dir: 出力ディレクトリの絶対パス

    Returns:
        出力ファイルの絶対パス
    """
    rel_path = os.path.relpath(input_path, input_dir)
    return os.path.join(output_dir, rel_path)


def process_images(
    input_dir: str,
    output_dir: str,
    tolerance: int,
    padding: int,
    overwrite: bool,
) -> None:
    """
    ディレクトリ内の全画像をトリミングして保存する。

    Args:
        input_dir: 入力画像フォルダ
        output_dir: 出力画像フォルダ
        tolerance: 背景色との許容差
        padding: トリミング後に追加する余白ピクセル数
        overwrite: 上書き保存フラグ（True の場合は input_dir に上書き）
    """
    files = collect_image_files(input_dir)
    if not files:
        print("処理対象の画像が見つかりませんでした。")
        return

    print(f"{len(files)} 枚の画像を検出しました。トリミングを開始します...\n")

    def process_one(args: tuple) -> bool:
        input_path, filename = args
        if overwrite:
            output_path = input_path
        else:
            output_path = resolve_output_path(input_path, input_dir, output_dir)
        return trim_image(input_path, output_path, tolerance, padding)

    workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_one, files))

    success_count = sum(1 for r in results if r)
    skip_count = len(results) - success_count

    print(f"\n{'=' * 50}")
    print(f"処理完了: {success_count} 枚成功、{skip_count} 枚スキップ")
    if not overwrite:
        print(f"出力先: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="フォルダ内にある画像の余白を自動でトリミングします。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 入力フォルダを指定して出力フォルダへ保存（デフォルト）
  python trim_margin.py -i images/

  # 許容差を指定（背景が完全な単色でない場合に有効）
  python trim_margin.py -i images/ -t 20

  # トリミング後に余白を追加
  python trim_margin.py -i images/ -p 10

  # 元のファイルに上書き保存
  python trim_margin.py -i images/ --overwrite
        """,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="トリミング対象の画像フォルダ",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="trimmed",
        help="トリミング済み画像の出力フォルダ (デフォルト: trimmed)",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=int,
        default=10,
        help="背景色との許容差 (0〜255)。大きいほど余白として判定しやすくなります (デフォルト: 10)",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=0,
        help="トリミング後にコンテンツ周囲へ追加する余白のピクセル数 (デフォルト: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="元のファイルに上書き保存します（--output-dir は無視されます）",
    )


    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)

    if not os.path.isdir(input_dir):
        print(f"エラー: 入力フォルダ '{input_dir}' が存在しません。", file=sys.stderr)
        sys.exit(1)

    if args.tolerance < 0 or args.tolerance > 255:
        print("エラー: --tolerance は 0〜255 の範囲で指定してください。", file=sys.stderr)
        sys.exit(1)

    if args.padding < 0:
        print("エラー: --padding は 0 以上の値を指定してください。", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)

    if not args.overwrite and input_dir == output_dir:
        print(
            "エラー: 入力フォルダと出力フォルダが同じです。上書きする場合は --overwrite を指定してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"入力フォルダ  : {input_dir}")
    if args.overwrite:
        print("出力先        : 元ファイルに上書き")
    else:
        print(f"出力フォルダ  : {output_dir}")
    print(f"許容差        : {args.tolerance}")
    print(f"パディング    : {args.padding}px")
    print(f"スレッド数    : {os.cpu_count() or 1} (CPUコア数)")
    print()

    process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        tolerance=args.tolerance,
        padding=args.padding,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
