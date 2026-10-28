"""指定ディレクトリ配下の画像を変換する。"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_QUALITY = 80

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_OUTPUT_FORMATS = ("png", "jpg", "webp")

TARGET_EXTENSIONS: dict[str, str] = {
    "png": ".png",
    "jpg": ".jpg",
    "webp": ".webp",
}


def is_image(path: Path) -> bool:
    """path が Pillow で認識できる画像なら True を返す。"""
    try:
        with Image.open(path) as image:
            image.verify()
    except UnidentifiedImageError, OSError, ValueError:
        return False
    return True


def format_size(num_bytes: int | float, signed: bool = False) -> str:
    """バイト数を適切な単位 (B, KB, MB, GB) にフォーマットする。"""
    prefix = "+" if signed and num_bytes > 0 else ""
    abs_bytes = abs(num_bytes)
    if abs_bytes < 1024:
        return f"{prefix}{num_bytes:.0f} B"
    for unit in ("KB", "MB", "GB", "TB"):
        num_bytes /= 1024
        abs_bytes /= 1024
        if abs_bytes < 1024 or unit == "TB":
            return f"{prefix}{num_bytes:.1f} {unit}"
    return f"{prefix}{num_bytes:.1f} B"


def save_image(
    image: Image.Image,
    destination: Path,
    target_format: str,
    quality: int = DEFAULT_QUALITY,
) -> None:
    """指定されたフォーマットに応じた設定で画像を保存する。"""
    normalized_format = target_format.lower()

    if normalized_format == "jpg":
        # JPEG はアルファチャンネルを保存できないため、透過部分は白で合成する。
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            background = Image.new("RGB", image.size, "white")
            alpha = image.convert("RGBA").getchannel("A")
            background.paste(image, mask=alpha)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.save(destination, format="JPEG", quality=quality)
    elif normalized_format == "webp":
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            image = image.convert("RGBA")
        elif image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(destination, format="WEBP", quality=quality)
    elif normalized_format == "png":
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            image = image.convert("RGBA")
        elif image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(destination, format="PNG")
    else:
        raise ValueError(f"未対応のフォーマットです: {target_format}")


def convert_image(
    path: Path,
    destination: Path,
    target_format: str,
    quality: int = DEFAULT_QUALITY,
) -> tuple[int, int]:
    """画像を target_format に変換して destination に保存し、成功時に元のファイルを削除する。

    Returns:
        (変換前ファイルサイズ, 変換後ファイルサイズ)
    """
    if destination.exists():
        raise FileExistsError(f"出力先が既に存在します: {destination}")

    before_size = path.stat().st_size

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        save_image(image, destination, target_format, quality=quality)

    after_size = destination.stat().st_size
    path.unlink()
    return before_size, after_size


def rename_image(path: Path, destination: Path) -> None:
    """画像の拡張子を変更する（同一フォーマットで再エンコード不要な場合）。"""
    if destination.exists():
        raise FileExistsError(f"変更先が既に存在します: {destination}")
    path.rename(destination)


def process_file(
    path: Path, output_format: str, quality: int = DEFAULT_QUALITY
) -> str | None:
    """単一の画像ファイルを処理する。"""
    dest_ext = TARGET_EXTENSIONS[output_format]
    destination = path.with_suffix(dest_ext)

    current_ext = path.suffix.lower()

    # 拡張子が大文字小文字を区別せず一致している場合は変換不要
    if current_ext == dest_ext:
        return None

    # JPEG 同士 (.jpeg -> .jpg) のリネーム
    if current_ext in {".jpg", ".jpeg"} and dest_ext == ".jpg":
        rename_image(path, destination)
        return f"リネーム: {path} -> {destination}"

    before_size, after_size = convert_image(
        path, destination, output_format, quality=quality
    )
    diff = after_size - before_size
    percent = (diff / before_size * 100) if before_size > 0 else 0.0
    diff_str = f"{format_size(diff, signed=True)} ({percent:+.1f}%)"
    return (
        f"変換: {path} -> {destination} "
        f"({format_size(before_size)} -> {format_size(after_size)}, {diff_str})"
    )


def process_single_file(
    path: Path,
    output_format: str,
    quality: int = DEFAULT_QUALITY,
) -> tuple[bool, str | None]:
    """1つの画像ファイルを検証・処理し、(成功フラグ, メッセージ) を返す。"""
    if not is_image(path):
        return True, None

    try:
        msg = process_file(path, output_format, quality=quality)
        return True, msg
    except (OSError, ValueError) as error:
        return False, f"エラー: {path}: {error}"


def process_directory(
    input_dir: Path,
    target_input_extensions: set[str],
    output_format: str,
    quality: int = DEFAULT_QUALITY,
    max_workers: int | None = None,
) -> None:
    """input_dir 配下の画像を並列で再帰的に処理する。"""
    candidate_paths = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in target_input_extensions
    ]

    if not candidate_paths:
        return

    workers = max_workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                path,
                output_format,
                quality,
            ): path
            for path in candidate_paths
        }

        for future in as_completed(futures):
            path = futures[future]
            try:
                success, msg = future.result()
                if msg:
                    if success:
                        print(msg)
                    else:
                        print(msg, file=sys.stderr)
            except Exception as error:
                print(f"エラー: {path}: {error}", file=sys.stderr)


def normalize_output_format(value: str) -> str:
    """出力フォーマット文字列を正規化・検証する。"""
    normalized = value.strip().lstrip(".").lower()
    if normalized not in SUPPORTED_OUTPUT_FORMATS:
        raise argparse.ArgumentTypeError(
            f"未対応の出力フォーマットです: {value} "
            f"(指定可能: {', '.join(SUPPORTED_OUTPUT_FORMATS)})"
        )
    return normalized


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """コマンドライン引数の妥当性を検証する。"""
    if not args.input_dir.is_dir():
        parser.error(f"ディレクトリではないか、存在しません: {args.input_dir}")

    if not (1 <= args.quality <= 100):
        parser.error(f"--quality は 1〜100 の範囲で指定してください: {args.quality}")

    if args.workers is not None and args.workers <= 0:
        parser.error(f"--workers は 1 以上の整数を指定してください: {args.workers}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定ディレクトリ配下の画像のフォーマットを再帰的に変換します。"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="探索対象のディレクトリ",
    )
    parser.add_argument(
        "-of",
        "--output-format",
        type=normalize_output_format,
        default="jpg",
        help=f"出力フォーマット ({', '.join(SUPPORTED_OUTPUT_FORMATS)}。デフォルト: jpg)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=DEFAULT_QUALITY,
        help=f"JPEG / WebP 変換時の品質 (1-100, デフォルト: {DEFAULT_QUALITY})",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="並列処理のワーカー数 (デフォルト: CPUコア数)",
    )
    args = parser.parse_args()

    validate_args(args, parser)

    output_format: str = args.output_format
    target_output_ext = TARGET_EXTENSIONS[output_format]
    target_input_extensions = SUPPORTED_EXTENSIONS - {target_output_ext}

    process_directory(
        input_dir=args.input_dir,
        target_input_extensions=target_input_extensions,
        output_format=output_format,
        quality=args.quality,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
