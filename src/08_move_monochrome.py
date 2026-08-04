"""
フォルダ内の画像がモノクロ（グレースケール）かどうかを判定し、
該当する画像を指定フォルダへ移動するスクリプト。

HSV色空間の彩度（Saturation）チャンネルの平均値を用いて判定します。
彩度の平均が閾値以下であればモノクロと見なします。
"""

import os
import sys
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_monochrome(image_path: str, threshold: float) -> bool:
    """
    画像がモノクロかどうかを判定する。

    HSV色空間に変換し、彩度チャンネルの平均値が閾値以下であれば
    モノクロと判定する。

    Args:
        image_path: 画像ファイルのパス
        threshold: モノクロ判定の彩度閾値 (0.0〜255.0)

    Returns:
        モノクロと判定された場合は True
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        img_hsv = img_rgb.convert("HSV")

        # 彩度チャンネル (S) を取り出す
        saturation = img_hsv.split()[1]
        pixels = list(saturation.getdata())

        if not pixels:
            return True

        avg_saturation = sum(pixels) / len(pixels)
        return avg_saturation <= threshold


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


def process_image(
    image_path: str,
    filename: str,
    output_dir: str,
    threshold: float,
) -> bool:
    """
    1枚の画像をモノクロ判定し、該当すれば移動する。

    Args:
        image_path: 画像ファイルのパス
        filename: ファイル名
        output_dir: 移動先ディレクトリ
        threshold: モノクロ判定の彩度閾値

    Returns:
        モノクロと判定して移動した場合は True
    """
    try:
        if is_monochrome(image_path, threshold):
            dest = os.path.join(output_dir, filename)
            shutil.move(image_path, dest)
            print(f"  移動: {filename}")
            return True
        else:
            print(f"  スキップ（カラー）: {filename}")
            return False
    except Exception as e:
        print(
            f"  エラー: {filename} の処理に失敗しました: {e}",
            file=sys.stderr,
        )
        return False


def process_images(
    input_dir: str,
    output_dir: str,
    threshold: float,
) -> None:
    """
    ディレクトリ内の全画像をモノクロ判定し、該当するものを移動する。

    Args:
        input_dir: 入力画像フォルダ
        output_dir: モノクロ画像の移動先フォルダ
        threshold: モノクロ判定の彩度閾値
    """
    files = collect_image_files(input_dir)
    if not files:
        print("処理対象の画像が見つかりませんでした。")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"{len(files)} 枚の画像を検出しました。モノクロ判定を開始します...\n")

    def process_one(args: tuple) -> bool:
        path, filename = args
        return process_image(path, filename, output_dir, threshold)

    workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_one, files))

    moved_count = sum(1 for r in results if r)
    skipped_count = len(results) - moved_count

    print(f"\n{'=' * 50}")
    print(f"処理完了: {moved_count} 枚移動、{skipped_count} 枚スキップ")
    print(f"移動先: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="フォルダ内のモノクロ画像を判定し、指定フォルダへ移動します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用例:
  # 入力フォルダと出力フォルダを指定
  python 08_move_monochrome.py -i images/ -o monochrome/

  # 彩度閾値を調整（値が大きいほどモノクロと判定されやすくなる）
  python 08_move_monochrome.py -i images/ -o monochrome/ -t 30
        """,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="スキャン対象の画像フォルダ",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="モノクロ画像の移動先フォルダ",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=10.0,
        help="モノクロ判定の彩度閾値 (0.0〜255.0)。"
        "彩度の平均がこの値以下ならモノクロと判定します (デフォルト: 10.0)",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"エラー: 入力フォルダ '{input_dir}' が存在しません。", file=sys.stderr)
        sys.exit(1)

    if args.threshold < 0 or args.threshold > 255:
        print(
            "エラー: --threshold は 0〜255 の範囲で指定してください。", file=sys.stderr
        )
        sys.exit(1)

    if input_dir == output_dir:
        print(
            "エラー: 入力フォルダと出力フォルダが同じです。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"入力フォルダ  : {input_dir}")
    print(f"出力フォルダ  : {output_dir}")
    print(f"彩度閾値      : {args.threshold}")
    print(f"スレッド数    : {os.cpu_count() or 1} (CPUコア数)")
    print()

    process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
