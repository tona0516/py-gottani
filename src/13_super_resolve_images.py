"""Swin2SR を使ってディレクトリ内の画像を 2 倍に超解像する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms.functional import to_pil_image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

MODEL_ID = "caidas/swin2SR-classical-sr-x2-64"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def select_device() -> torch.device:
    """利用可能なら CUDA、次に MPS、なければ CPU を選ぶ。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_paths(input_dir: Path, output_dir: Path) -> list[Path]:
    """出力先を除いた input_dir 配下の対応画像を再帰的に返す。"""
    output_dir = output_dir.resolve()
    paths: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(output_dir)
        except ValueError:
            paths.append(path)
    return paths


def load_model(
    device: torch.device,
) -> tuple[AutoImageProcessor, Swin2SRForImageSuperResolution]:
    """2 倍超解像モデルと前処理器を読み込む。初回はモデルをダウンロードする。"""
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model


def super_resolve(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: Swin2SRForImageSuperResolution,
    device: torch.device,
) -> Image.Image:
    """画像をモデルで正確に縦横 2 倍にする。アルファチャンネルは保持する。"""
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB")

    inputs = processor(rgb, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        reconstruction = model(**inputs).reconstruction[0].clamp(0, 1).cpu()

    # プロセッサのパディング分を取り除き、入力画像のちょうど 2 倍にそろえる。
    width, height = rgb.size
    result = to_pil_image(reconstruction[:, : height * 2, : width * 2])
    result.putalpha(alpha.resize(result.size, Image.Resampling.LANCZOS))
    return result


def save_image(image: Image.Image, destination: Path) -> None:
    """拡張子に合う形式で画像を保存する。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        background.save(destination, quality=95)
    else:
        image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定ディレクトリ内の画像を Swin2SR で縦横 2 倍に超解像します。"
    )
    parser.add_argument(
        "-i", "--input-dir", type=Path, required=True, help="入力ディレクトリ"
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        parser.error(f"ディレクトリではないか、存在しません: {input_dir}")
    output_dir = input_dir.with_name(f"{input_dir.name}_upscaled")

    paths = image_paths(input_dir, output_dir)
    if not paths:
        print("処理対象の画像がありません。")
        return

    device = select_device()
    print(f"モデルを読み込みます: {MODEL_ID} ({device})")
    try:
        processor, model = load_model(device)
    except (OSError, ValueError) as error:
        print(f"モデルを読み込めません: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for path in paths:
        destination = output_dir / path.relative_to(input_dir)
        try:
            with Image.open(path) as image:
                result = super_resolve(image, processor, model, device)
            save_image(result, destination)
            print(f"超解像: {path} -> {destination}")
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as error:
            print(f"エラー: {path}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
