"""画像を含む各フォルダについて、最頻出の解像度を表示する。

使い方:
    uv run src/12_print_first_image_resolution.py -i <ディレクトリ>
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def configure_output_encoding() -> None:
    """Unicode を含むパスをリダイレクト先へ安全に出力できるようにする。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def find_images_by_directory(root: Path) -> dict[Path, list[Path]]:
    """root 配下の画像候補を親フォルダごとに、パス順でまとめる。"""
    images_by_directory: dict[Path, list[Path]] = defaultdict(list)

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_directory[path.parent].append(path)

    for images in images_by_directory.values():
        images.sort(key=lambda path: path.name.casefold())

    return dict(images_by_directory)


def print_most_common_resolutions(root: Path) -> int:
    """各フォルダの最頻出解像度を、総画素数の大きい順に表示する。"""
    images_by_directory = find_images_by_directory(root)
    if not images_by_directory:
        print("画像ファイルが見つかりませんでした。", file=sys.stderr)
        return 1

    resolutions: list[tuple[int, int, Path]] = []
    for images in images_by_directory.values():
        paths_by_resolution: dict[tuple[int, int], list[Path]] = defaultdict(list)
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    paths_by_resolution[image.size].append(image_path)
            except (OSError, UnidentifiedImageError) as error:
                print(
                    f"[スキップ] 読み込みに失敗しました: {image_path} ({error})",
                    file=sys.stderr,
                )

        if not paths_by_resolution:
            continue

        # 同数の場合は総画素数、幅、高さが大きい解像度を優先する。
        (width, height), paths = min(
            paths_by_resolution.items(),
            key=lambda item: (
                -len(item[1]),
                -item[0][0] * item[0][1],
                -item[0][0],
                -item[0][1],
            ),
        )
        # 同じ解像度の画像は名前順に並んでいるため、先頭を代表パスとして出力する。
        resolutions.append((width, height, paths[0]))

    if not resolutions:
        print("読み込める画像ファイルが見つかりませんでした。", file=sys.stderr)
        return 1

    # 総画素数、幅、高さの順で降順。同じ解像度ではパス順にする。
    resolutions.sort(
        key=lambda item: (
            -item[0] * item[1],
            -item[0],
            -item[1],
            str(item[2]).casefold(),
        )
    )
    for width, height, path in resolutions:
        print(f"{width}x{height}\t{path}")

    return 0


def main() -> None:
    configure_output_encoding()

    parser = argparse.ArgumentParser(
        description="指定ディレクトリ配下の各画像フォルダについて、最頻出の解像度と代表パスを表示します。"
    )
    parser.add_argument(
        "-i", "--input-dir", type=Path, help="再帰的に探索するディレクトリ"
    )
    args = parser.parse_args()

    root = args.input_dir
    if not root.is_dir():
        parser.error(f"ディレクトリが見つかりません: {root}")

    raise SystemExit(print_most_common_resolutions(root))


if __name__ == "__main__":
    main()
