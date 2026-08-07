"""
指定したフォルダ内で類似画像を検出し、類似画像を持つ親フォルダ名を出力するスクリプト。

<指定するフォルダ>
  |-フォルダ1
  |-フォルダ2
  |-...

このような構造のルートフォルダを1つ指定し、その直下のサブフォルダ（親フォルダ）に
含まれる画像同士を dHash のハミング距離で比較します。
同一フォルダ内の画像同士は比較対象外とし、異なるフォルダ間でのみ類似判定を行います。
ハミング距離がしきい値以下の類似画像ペアが存在する場合、どのフォルダ間で見つかったかを
結果に反映し、コンソールおよびファイルへ出力します。

背景が支配的なロゴなどのシンプルな画像は dHash の判定が不安定になるため、
比較対象から除外します。
"""

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import imagehash
from PIL import Image
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# シンプル画像判定用のサンプル解像度
COMPLEXITY_SAMPLE_SIZE = 64
# 背景以外のピクセル割合がこの値未満ならロゴ的画像とみなす
SIMPLE_MIN_CONTENT_RATIO = 0.10


@dataclass
class ImageInfo:
    """1枚の画像の情報を保持するデータクラス。"""

    path: str
    filename: str
    parent_dir: str
    dhash: int


def convert_palette_to_rgba_if_needed(img: Image.Image) -> Image.Image:
    """
    Pモードで透過情報がある場合、RGBAに変換する（Pillowの警告回避）。
    """
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA")
    return img


def calculate_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """
    ImageオブジェクトからdHash（Difference Hash）値を計算する。
    """
    try:
        img = convert_palette_to_rgba_if_needed(img)
        h = imagehash.dhash(img, hash_size=hash_size)
        return int(str(h), 16)
    except Exception as e:
        print(f"dHash計算中にエラーが発生しました: {e}", file=sys.stderr)
        return None


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    2つのハッシュ値のハミング距離（異なるビット数）を計算する。
    """
    return (hash1 ^ hash2).bit_count()


def is_simple_image(img: Image.Image) -> bool:
    """
    ロゴや単色などのシンプルな画像かどうかを判定する。

    最頻色（背景色）と異なるピクセルの割合が
    SIMPLE_MIN_CONTENT_RATIO 未満であればシンプル画像とみなす。

    Args:
        img: 判定対象の画像

    Returns:
        シンプル画像と判定された場合は True
    """
    try:
        img_rgb = convert_palette_to_rgba_if_needed(img).convert("RGB")

        # 縮小して色情報を取り出す
        small = img_rgb.resize(
            (COMPLEXITY_SAMPLE_SIZE, COMPLEXITY_SAMPLE_SIZE),
            Image.Resampling.NEAREST,
        )
        colors = small.getcolors(
            maxcolors=COMPLEXITY_SAMPLE_SIZE * COMPLEXITY_SAMPLE_SIZE
        )
        if colors is None:
            return False

        total_pixels = sum(count for count, _ in colors)

        # 最頻色を背景色とみなし、背景以外のピクセル割合が小さければロゴ的とみなす
        _, bg_color = max(colors, key=lambda item: item[0])
        tolerance = 32
        bg_count = sum(
            count
            for count, (r, g, b) in colors
            if all(abs(c - b) <= tolerance for c, b in zip((r, g, b), bg_color))
        )
        content_ratio = 1.0 - bg_count / total_pixels
        return content_ratio < SIMPLE_MIN_CONTENT_RATIO
    except Exception as e:
        print(f"シンプル画像判定中にエラーが発生しました: {e}", file=sys.stderr)
        return False


def collect_parent_folders(root_dir: str) -> list:
    """
    ルートフォルダ直下のサブフォルダ（親フォルダ）名の一覧を返す。
    """
    parent_folders = []
    try:
        for name in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, name)
            if os.path.isdir(path):
                parent_folders.append(name)
    except Exception as e:
        print(f"フォルダのスキャン中にエラーが発生しました: {e}", file=sys.stderr)
    return parent_folders


def process_image_file(path: str, parent_dir: str) -> tuple:
    """
    1枚の画像を処理し、(ImageInfo, シンプル画像として除外されたか) を返す。

    ロゴや単色などのシンプル画像は比較対象から除外する。
    読み込み失敗などのエラー時は (None, False) を返す。
    """
    try:
        with Image.open(path) as img:
            if is_simple_image(img):
                return None, True
            h = calculate_dhash(img, hash_size=8)
        if h is None:
            return None, False
        return (
            ImageInfo(
                path=path,
                filename=os.path.basename(path),
                parent_dir=parent_dir,
                dhash=h,
            ),
            False,
        )
    except Exception as e:
        print(f"警告: {path} の処理に失敗しました: {e}", file=sys.stderr)
        return None, False


def collect_images_in_parent(root_dir: str, parent_dir: str) -> tuple:
    """
    親フォルダ内の画像ファイルを再帰的にスキャンし、
    (ImageInfo のリスト, シンプル画像として除外された数) を返す。
    """
    tasks = []
    parent_path = os.path.join(root_dir, parent_dir)
    for current_root, _, files in os.walk(parent_path):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                tasks.append((os.path.join(current_root, file), parent_dir))

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image_file, path, parent) for path, parent in tasks
        ]
        results = [f.result() for f in futures]

    image_infos = []
    skipped_count = 0
    for info, is_simple in results:
        if info is not None:
            image_infos.append(info)
        elif is_simple:
            skipped_count += 1

    return image_infos, skipped_count


def scan_root_folder(root_dir: str) -> tuple:
    """
    ルートフォルダ直下の全親フォルダから画像情報を収集し、
    (ImageInfo のリスト, シンプル画像として除外された数) を返す。
    """
    if not os.path.isdir(root_dir):
        print(
            f"エラー: フォルダ {root_dir} が存在しないか、フォルダではありません。",
            file=sys.stderr,
        )
        return [], 0

    parent_folders = collect_parent_folders(root_dir)
    if not parent_folders:
        print(
            f"警告: フォルダ {root_dir} にサブフォルダが見つかりません。",
            file=sys.stderr,
        )

    image_infos = []
    total = 0
    skipped_total = 0
    for parent in tqdm(parent_folders, desc=f"スキャン中: {root_dir}"):
        infos, skipped = collect_images_in_parent(root_dir, parent)
        total += len(infos)
        skipped_total += skipped
        image_infos.extend(infos)

    print(f"{root_dir}: 合計 {total} 枚の画像を読み込みました。")
    if skipped_total > 0:
        print(
            f"{root_dir}: シンプル画像（ロゴ・単色など）として {skipped_total} 枚を除外しました。"
        )
    return image_infos, skipped_total


def find_similar_pairs(image_infos: list, threshold: int) -> list:
    """
    異なる親フォルダ間の画像リストでハミング距離を総当たり比較し、
    しきい値以下の類似ペアを返す。同一フォルダ内の画像同士は比較しない。
    """
    similar_pairs = []
    n = len(image_infos)
    for i in tqdm(range(n), desc="類似ペア探索"):
        info_a = image_infos[i]
        for j in range(i + 1, n):
            info_b = image_infos[j]
            # 同じ親フォルダ内の画像同士は比較対象外
            if info_a.parent_dir == info_b.parent_dir:
                continue
            if hamming_distance(info_a.dhash, info_b.dhash) <= threshold:
                similar_pairs.append((info_a, info_b))
    return similar_pairs


def format_folder_pair(dir_a: str, dir_b: str) -> str:
    """
    2つの親フォルダ名をソートして正規化した「フォルダ間」キーを返す。
    """
    return " <-> ".join(sorted((dir_a, dir_b)))


def build_results_text(similar_pairs: list, root_dir: str) -> str:
    """
    類似ペアの詳細と、どのフォルダ間で類似画像が見つかったかの結果テキストを生成する。
    """
    if not similar_pairs:
        return "類似画像は見つかりませんでした。\n"

    lines = [
        "=" * 60,
        f"{len(similar_pairs)} 件の類似画像ペアが見つかりました。",
        "=" * 60,
        "",
    ]

    # フォルダ間ごとのペア数を集計
    folder_pair_counts: dict[str, int] = {}
    folder_names = set()

    for i, (info_a, info_b) in enumerate(similar_pairs, 1):
        dist = hamming_distance(info_a.dhash, info_b.dhash)
        folder_pair = format_folder_pair(info_a.parent_dir, info_b.parent_dir)
        folder_pair_counts[folder_pair] = folder_pair_counts.get(folder_pair, 0) + 1
        folder_names.add(info_a.parent_dir)
        folder_names.add(info_b.parent_dir)

        lines.append(f"ペア {i}: ハミング距離 {dist}  |  フォルダ間: {folder_pair}")
        lines.append(f"  [{info_a.parent_dir}] {info_a.filename}")
        lines.append(f"  [{info_b.parent_dir}] {info_b.filename}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("類似画像が見つかったフォルダ間:")
    for folder_pair, count in sorted(folder_pair_counts.items()):
        lines.append(f"  {folder_pair}: {count} 件")

    lines.append("")
    lines.append("-" * 60)
    lines.append("類似画像がある親フォルダ名:")
    lines.append(f"  {os.path.basename(root_dir)}: {', '.join(sorted(folder_names))}")
    lines.append("")

    return "\n".join(lines)


def print_results(similar_pairs: list, root_dir: str) -> str:
    """
    類似ペア中の親フォルダ間情報と画像情報を出力し、結果テキストを返す。
    """
    results_text = build_results_text(similar_pairs, root_dir)
    print("\n" + results_text, end="")
    return results_text


def write_results_file(results_text: str, output_path: str) -> None:
    """
    結果テキストを指定パスへ書き出す。
    """
    try:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(results_text)
        print(f"結果をファイルに出力しました: {output_path}")
    except Exception as e:
        print(f"エラー: 結果ファイルの書き込みに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定したフォルダ内で類似画像を検出し、類似画像を持つ親フォルダ名を出力します。"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="スキャン対象のルートフォルダ（直下にフォルダ1, フォルダ2, ... を持つ）",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=2,
        help="類似判定するハミング距離のしきい値。小さいほど厳密 (デフォルト: 2, 範囲: 0-64)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="結果を書き出すファイルパス (未指定時: 入力フォルダ直下の ls.txt)",
    )

    args = parser.parse_args()

    if args.threshold < 0 or args.threshold > 64:
        print(
            "エラー: --threshold は 0〜64 の範囲で指定してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    root_dir = os.path.abspath(args.input_dir)

    if not os.path.isdir(root_dir):
        print(
            f"エラー: 入力フォルダ {root_dir} が存在しないか、フォルダではありません。",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = (
        os.path.abspath(args.output)
        if args.output
        else os.path.join(root_dir, "ls.txt")
    )

    print(f"入力フォルダ : {root_dir}")
    print(f"ハミング距離しきい値: {args.threshold}")
    print(f"シンプル画像判定: コンテンツ割合 {SIMPLE_MIN_CONTENT_RATIO} 未満")
    print(f"結果出力ファイル: {output_path}")
    print()

    image_infos, _ = scan_root_folder(root_dir)

    if not image_infos:
        print(
            "エラー: フォルダ内に画像が見つかりませんでした。",
            file=sys.stderr,
        )
        sys.exit(1)

    similar_pairs = find_similar_pairs(image_infos, args.threshold)
    results_text = print_results(similar_pairs, root_dir)
    write_results_file(results_text, output_path)


if __name__ == "__main__":
    main()
