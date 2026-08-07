"""指定ディレクトリ配下の画像を JPEG に正規化する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

JPEG_QUALITY = 85


def is_image(path: Path) -> bool:
    """path が Pillow で認識できる画像なら True を返す。"""
    try:
        with Image.open(path) as image:
            image.verify()
    except UnidentifiedImageError, OSError, ValueError:
        return False
    return True


def jpeg_destination(path: Path) -> Path:
    """拡張子だけを .jpg に置き換えた出力先を返す。"""
    return path.with_suffix(".jpg")


def convert_png(path: Path) -> None:
    """PNG を品質 85 の JPEG に変換し、成功時に元の PNG を削除する。"""
    destination = jpeg_destination(path)
    if destination.exists():
        raise FileExistsError(f"出力先が既に存在します: {destination}")

    # JPEG はアルファチャンネルを保存できないため、透過部分は白で合成する。
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            background = Image.new("RGB", image.size, "white")
            alpha = image.convert("RGBA").getchannel("A")
            background.paste(image, mask=alpha)
            image = background
        else:
            image = image.convert("RGB")
        image.save(destination, format="JPEG", quality=JPEG_QUALITY)

    path.unlink()


def rename_jpeg(path: Path) -> None:
    """JPEG ファイルの拡張子を .jpg に変更する。"""
    destination = jpeg_destination(path)
    if destination.exists():
        raise FileExistsError(f"変更先が既に存在します: {destination}")
    path.rename(destination)


def process_directory(input_dir: Path) -> None:
    """input_dir 配下の画像を再帰的に処理する。"""
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or not is_image(path):
            continue

        extension = path.suffix.lower()
        try:
            if extension == ".png":
                convert_png(path)
                print(f"変換: {path} -> {jpeg_destination(path)}")
            elif extension == ".jpeg":
                rename_jpeg(path)
                print(f"リネーム: {path} -> {jpeg_destination(path)}")
        except (OSError, ValueError) as error:
            print(f"エラー: {path}: {error}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "指定ディレクトリ配下の画像を再帰的に処理します。"
            "PNG は品質 85 の JPG に変換し、JPEG は拡張子を JPG に変更します。"
        )
    )
    parser.add_argument("-i", "--input-dir", type=Path, help="探索対象のディレクトリ")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        parser.error(f"ディレクトリではないか、存在しません: {input_dir}")

    process_directory(input_dir)


if __name__ == "__main__":
    main()
