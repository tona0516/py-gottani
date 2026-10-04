"""Swin2SR を使ってディレクトリ内の画像を 2 倍または 4 倍に超解像する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms.functional import to_pil_image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

MODEL_IDS: dict[int, str] = {
    2: "caidas/swin2SR-classical-sr-x2-64",
    4: "caidas/swin2SR-classical-sr-x4-64",
}
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
    scale: int,
    device: torch.device,
) -> tuple[AutoImageProcessor, Swin2SRForImageSuperResolution]:
    """指定倍率の超解像モデルと前処理器を読み込む。初回はモデルをダウンロードする。"""
    if scale not in MODEL_IDS:
        msg = f"サポートされていない倍率です: {scale} (サポート: {list(MODEL_IDS.keys())})"
        raise ValueError(msg)
    model_id = MODEL_IDS[scale]
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def _tile_weight(height: int, width: int, overlap: int) -> torch.Tensor:
    """タイルの境界を滑らかに合成するための 2 次元重みマップを生成する。"""

    def _linear_ramp(length: int, pad: int) -> torch.Tensor:
        ramp = torch.ones(length, dtype=torch.float32)
        if pad > 0 and length > pad * 2:
            ramp[:pad] = torch.linspace(0.01, 1.0, pad)
            ramp[-pad:] = torch.linspace(1.0, 0.01, pad)
        return ramp

    h_weight = _linear_ramp(height, overlap)
    w_weight = _linear_ramp(width, overlap)
    return torch.outer(h_weight, w_weight).unsqueeze(0)


def _generate_tiles(
    length: int, tile_size: int, tile_overlap: int
) -> list[tuple[int, int]]:
    """1 次元のタイル開始・終了座標のリストを生成する。"""
    if length <= tile_size:
        return [(0, length)]

    stride = max(1, tile_size - tile_overlap)
    coords: list[tuple[int, int]] = []
    start = 0
    while start + tile_size < length:
        coords.append((start, start + tile_size))
        start += stride
    last_tile = (length - tile_size, length)
    if not coords or coords[-1] != last_tile:
        coords.append(last_tile)
    return coords


def _infer_tile(
    tile_rgb: Image.Image,
    processor: AutoImageProcessor,
    model: Swin2SRForImageSuperResolution,
    scale: int,
    device: torch.device,
) -> torch.Tensor:
    """単一タイルの超解像推論を行い、(3, H*scale, W*scale) のテンソルを返す。"""
    inputs = processor(tile_rgb, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        reconstruction = model(**inputs).reconstruction[0].clamp(0, 1).cpu()

    width, height = tile_rgb.size
    return reconstruction[:, : height * scale, : width * scale]


def super_resolve(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: Swin2SRForImageSuperResolution,
    scale: int,
    device: torch.device,
    tile_size: int = 256,
    tile_overlap: int = 16,
) -> Image.Image:
    """画像をモデルで正確に縦横指定倍率にする。VRAM 節約のためタイル分割推論を行う。"""
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB")
    width, height = rgb.size

    # タイル分割が無効または画像全体がタイルサイズ以下の場合は直接推論
    if tile_size <= 0 or (width <= tile_size and height <= tile_size):
        reconstruction = _infer_tile(rgb, processor, model, scale, device)
        result = to_pil_image(reconstruction)
        result.putalpha(alpha.resize(result.size, Image.Resampling.LANCZOS))
        return result

    # タイル分割推論
    out_width = width * scale
    out_height = height * scale
    output = torch.zeros((3, out_height, out_width), dtype=torch.float32)
    weight_sum = torch.zeros((1, out_height, out_width), dtype=torch.float32)

    x_tiles = _generate_tiles(width, tile_size, tile_overlap)
    y_tiles = _generate_tiles(height, tile_size, tile_overlap)
    scaled_overlap = tile_overlap * scale

    for y1, y2 in y_tiles:
        for x1, x2 in x_tiles:
            tile_image = rgb.crop((x1, y1, x2, y2))
            tile_recon = _infer_tile(tile_image, processor, model, scale, device)

            t_h, t_w = (y2 - y1) * scale, (x2 - x1) * scale
            w_map = _tile_weight(t_h, t_w, scaled_overlap)

            output[:, y1 * scale : y2 * scale, x1 * scale : x2 * scale] += (
                tile_recon * w_map
            )
            weight_sum[:, y1 * scale : y2 * scale, x1 * scale : x2 * scale] += w_map

    output = (output / weight_sum).clamp(0, 1)
    result = to_pil_image(output)
    result.putalpha(alpha.resize(result.size, Image.Resampling.LANCZOS))

    if device.type == "cuda":
        torch.cuda.empty_cache()

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
        description="指定ディレクトリ内の画像を Swin2SR で縦横 2 倍または 4 倍に超解像します。"
    )
    parser.add_argument(
        "-i", "--input-dir", type=Path, required=True, help="入力ディレクトリ"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="出力ディレクトリ（省略時は input_dir_upscaled）",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        choices=[2, 4],
        default=2,
        help="超解像の倍率 (2 または 4, デフォルト: 2)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="VRAM 節約用のタイルサイズ (0 でタイル分割無効化, デフォルト: 256)",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=16,
        help="タイルのオーバーラップ幅 (デフォルト: 16)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        parser.error(f"ディレクトリではないか、存在しません: {input_dir}")
    scale: int = args.scale
    output_dir: Path = args.output_dir or input_dir.with_name(
        f"{input_dir.name}_upscaled"
    )

    paths = image_paths(input_dir, output_dir)
    if not paths:
        print("処理対象の画像がありません。")
        return

    device = select_device()
    model_id = MODEL_IDS[scale]
    print(
        f"モデルを読み込みます: {model_id} ({device}) [倍率: {scale}x, タイルサイズ: {args.tile_size}]"
    )
    try:
        processor, model = load_model(scale, device)
    except (OSError, ValueError) as error:
        print(f"モデルを読み込めません: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for path in paths:
        destination = output_dir / path.relative_to(input_dir)
        try:
            with Image.open(path) as image:
                result = super_resolve(
                    image,
                    processor,
                    model,
                    scale,
                    device,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                )
            save_image(result, destination)
            print(f"超解像: {path} -> {destination}")
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as error:
            print(f"エラー: {path}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
