"""
画像を重複削除するスクリプト
"""

import os
import sys
import argparse
import shutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageInfo:
    path: str
    filename: str
    dhash: int
    width: int
    height: int
    size: int


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False


def calculate_dhash(img, hash_size=8):
    """
    ImageオブジェクトからdHash（Difference Hash）値を計算する。
    """
    try:
        # dHashは各行の隣接ピクセルを比較するため、横方向に1ピクセル多く縮小する
        img_resized = img.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.BILINEAR
        )
        pixels = list(img_resized.tobytes())

        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                difference.append(pixel_left > pixel_right)

        decimal_value = 0
        for bit in difference:
            decimal_value = (decimal_value << 1) | bit
        return decimal_value
    except Exception as e:
        print(f"Error calculating hash: {e}", file=sys.stderr)
        return None


def hamming_distance(hash1, hash2):
    """
    2つのハッシュ値のハミング距離（異なるビット数）を計算する。
    """
    return bin(hash1 ^ hash2).count("1")


def process_single_image(path, file, hash_size):
    """
    1枚の画像を処理し、ImageInfoを返す。
    """
    try:
        # ファイルサイズの取得
        size = os.path.getsize(path)

        # 解像度取得とハッシュ計算を1回のオープンで行う
        with Image.open(path) as img:
            width, height = img.size
            h = calculate_dhash(img, hash_size)

        if h is not None:
            return ImageInfo(
                path=path,
                filename=file,
                dhash=h,
                width=width,
                height=height,
                size=size,
            )
    except Exception as e:
        print(f"警告: {path} の処理に失敗しました: {e}", file=sys.stderr)
    return None


def scan_directory(directory_path, hash_size=8):
    """
    指定ディレクトリ内の画像をスキャンして情報を取得する。
    """
    print(f"ディレクトリをスキャン中: {directory_path} ...")

    if not os.path.exists(directory_path):
        print(
            f"エラー: ディレクトリ {directory_path} が存在しません。", file=sys.stderr
        )
        return []

    # スキャン対象ファイルのリストアップ
    tasks = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                path = os.path.join(root, file)
                tasks.append((path, file))

    image_infos = []

    # スレッドプールを使用して並列処理
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_image, path, file, hash_size)
            for path, file in tasks
        ]
        for future in futures:
            result = future.result()
            if result is not None:
                image_infos.append(result)

    print(f"{len(image_infos)} 枚の画像を正常に読み込みました。")
    return image_infos


def find_duplicate_groups(image_infos, threshold):
    """
    ハミング距離のしきい値に基づいて、重複画像をグループ化する。
    """
    n = len(image_infos)
    uf = UnionFind(n)

    print("画像を比較して重複グループを検出中...")
    # 総当たりで比較（$O(N^2)$）
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming_distance(image_infos[i].dhash, image_infos[j].dhash)
            if dist <= threshold:
                uf.union(i, j)

    # グループごとに整理
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(image_infos[i])

    # 画像が2枚以上のグループ（重複画像あり）のみを抽出
    duplicate_groups = [g for g in groups.values() if len(g) > 1]
    return duplicate_groups


def select_best_image(group):
    """
    グループの中から最も品質の高い画像（優先生存）を選択する。
    ルール:
    1. 解像度（幅 x 高さ）が最大のものを優先
    2. 解像度が同じなら、ファイルサイズが大きい方を優先
    3. それでも同じなら、ファイル名のアルファベット順が先に来るものを優先
    """

    # 比較用キー関数: (-解像度, -ファイルサイズ, ファイル名)
    def key_func(info):
        resolution = info.width * info.height
        return (-resolution, -info.size, info.filename)

    sorted_group = sorted(group, key=key_func)
    return sorted_group[0], sorted_group[1:]


def handle_duplicates(duplicate_groups, action, output_dir):
    """
    重複画像を移動または削除する。
    """
    dry_run = action == "dry-run"
    total_duplicates = sum(len(group) - 1 for group in duplicate_groups)

    if total_duplicates == 0:
        print("重複画像は見つかりませんでした。")
        return

    print("\n" + "=" * 50)
    print(
        f"{len(duplicate_groups)} 個の重複グループが見つかりました（合計 {total_duplicates} 枚の余分な画像）。"
    )
    print("=" * 50 + "\n")

    if dry_run:
        print("=== DRY RUN MODE: ファイル操作は行われません ===")

    if action == "move" and not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    moved_or_deleted_count = 0
    saved_space = 0

    for i, group in enumerate(duplicate_groups, 1):
        original, redundants = select_best_image(group)
        print(f"Group {i}:")
        print(
            f"  [保持] {original.filename} ({original.width}x{original.height}, {original.size / 1024:.1f} KB)"
        )

        for rep in redundants:
            print(
                f"  [重複] {rep.filename} ({rep.width}x{rep.height}, {rep.size / 1024:.1f} KB) - ハミング距離: {hamming_distance(original.dhash, rep.dhash)}"
            )

            saved_space += rep.size

            if not dry_run:
                if action == "delete":
                    try:
                        os.remove(rep.path)
                        moved_or_deleted_count += 1
                    except Exception as e:
                        print(
                            f"    {rep.filename} の削除に失敗しました: {e}",
                            file=sys.stderr,
                        )
                elif action == "move":
                    try:
                        # 移動先でファイル名が衝突しないようにリネーム処理
                        dest_path = os.path.join(output_dir, rep.filename)
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(rep.filename)
                            counter = 1
                            while os.path.exists(
                                os.path.join(output_dir, f"{base}_{counter}{ext}")
                            ):
                                counter += 1
                            dest_path = os.path.join(
                                output_dir, f"{base}_{counter}{ext}"
                            )

                        shutil.move(rep.path, dest_path)
                        moved_or_deleted_count += 1
                    except Exception as e:
                        print(
                            f"    {rep.filename} の移動に失敗しました: {e}",
                            file=sys.stderr,
                        )
        print()

    print("-" * 50)
    if dry_run:
        print(
            f"ドライランが完了しました。実際に実行すると、{total_duplicates} 個のファイルが処理され、{saved_space / 1024 / 1024:.2f} MB の容量が削減されます。"
        )
        print(
            "実際に処理を実行するには、'-a move' または '-a delete' を指定して実行してください。"
        )
    else:
        action_ja = "削除" if action == "delete" else "移動"
        print(
            f"処理が完了しました。{moved_or_deleted_count} / {total_duplicates} 枚の画像を正常に{action_ja}しました。"
        )
        print(f"削減された容量: {saved_space / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="dHashを用いてフォルダ内の重複画像を検出し、整理します。"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="images",
        help="スキャン対象の画像フォルダ (デフォルト: images)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="duplicates",
        help="重複画像の移動先フォルダ（移動アクション時） (デフォルト: duplicates)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=3,
        help="重複判定するハミング距離のしきい値。小さいほど厳密 (デフォルト: 2, 範囲: 0-64)",
    )
    parser.add_argument(
        "-a",
        "--action",
        choices=["dry-run", "move", "delete"],
        default="dry-run",
        help="重複画像に対する処理。実際にファイルを移動または削除する場合は 'move' または 'delete' を指定します (デフォルト: dry-run)",
    )

    args = parser.parse_args()

    # パスの正規化
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if input_dir == output_dir and args.action == "move":
        print(
            "エラー: 'move' アクションを行う際、入力フォルダと出力フォルダは異なる必要があります。",
            file=sys.stderr,
        )
        sys.exit(1)

    image_infos = scan_directory(input_dir)
    if not image_infos:
        print("画像が見つからないか、スキャンに失敗しました。")
        sys.exit(0)

    duplicate_groups = find_duplicate_groups(image_infos, args.threshold)
    handle_duplicates(duplicate_groups, args.action, output_dir)


if __name__ == "__main__":
    main()
