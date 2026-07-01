"""
指定された画像と同一ディレクトリ内の画像群との類似度を計算し、
類似度の近い順に採番して別ディレクトリにコピーするスクリプト。
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
        print(f"Error calculating dhash: {e}", file=sys.stderr)
        return None


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    2つのハッシュ値のハミング距離（異なるビット数）を計算する。
    """
    return bin(hash1 ^ hash2).count("1")


def compare_color_histograms(img1: Image.Image, img2: Image.Image) -> float:
    """
    Pillowのhistogram()を用いて、2つの画像の正規化ヒストグラム交差（Histogram Intersection）を計算する。
    値は 0.0 から 1.0 の間。
    """
    try:
        img1 = convert_palette_to_rgba_if_needed(img1)
        img2 = convert_palette_to_rgba_if_needed(img2)
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        hist1 = img1.histogram()
        hist2 = img2.histogram()

        total1 = img1.width * img1.height
        total2 = img2.width * img2.height

        intersection = 0.0
        for h1, h2 in zip(hist1, hist2):
            intersection += min(h1 / total1, h2 / total2)

        return intersection / 3.0
    except Exception as e:
        print(f"Error comparing color histograms: {e}", file=sys.stderr)
        return 0.0


def compare_pixel_difference(img1: Image.Image, img2: Image.Image, size: int = 128) -> float:
    """
    2つの画像を縮小し、ピクセル間の平均絶対誤差（MAE）を計算する。
    値は 0.0 から 255.0 の間。
    """
    try:
        img1 = convert_palette_to_rgba_if_needed(img1)
        img2 = convert_palette_to_rgba_if_needed(img2)
        img1_gray = img1.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        img2_gray = img2.convert("L").resize((size, size), Image.Resampling.BILINEAR)

        bytes1 = img1_gray.tobytes()
        bytes2 = img2_gray.tobytes()

        total_diff = 0
        for b1, b2 in zip(bytes1, bytes2):
            total_diff += abs(b1 - b2)

        mae = total_diff / (size * size)
        return mae
    except Exception as e:
        print(f"Error comparing pixel difference: {e}", file=sys.stderr)
        return 255.0


def compute_clip_similarity(
    target_path: Path,
    image_paths: List[Path],
    model_name: str,
    device_name: str,
    batch_size: int = 64
) -> List[Tuple[Path, float]]:
    """
    CLIPを用いて対象画像と画像群とのコサイン類似度を一括計算する。
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("Error: torch or transformers is not installed. Please install them or use another metric.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    print(f"Using device: {device}")

    print(f"Loading CLIP model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # 基準画像の特徴量抽出
    try:
        target_image = Image.open(target_path).convert("RGB")
    except Exception as e:
        print(f"Error opening target image {target_path}: {e}", file=sys.stderr)
        return []

    with torch.no_grad():
        target_inputs = processor(images=target_image, return_tensors="pt").to(device)
        target_outputs = model.get_image_features(**target_inputs)
        target_feature = target_outputs.pooler_output if hasattr(target_outputs, "pooler_output") else target_outputs
        target_feature = target_feature / target_feature.norm(dim=-1, keepdim=True)

    results = []
    total_images = len(image_paths)

    with torch.no_grad():
        for idx in range(0, total_images, batch_size):
            batch_paths = image_paths[idx : idx + batch_size]
            batch_images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error opening image {path}: {e}", file=sys.stderr)

            if not batch_images:
                continue

            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            features = features / features.norm(dim=-1, keepdim=True)

            similarities = torch.matmul(features, target_feature.T).squeeze(-1).cpu().tolist()

            if isinstance(similarities, float):
                similarities = [similarities]

            for path, sim in zip(valid_paths, similarities):
                results.append((path, sim))

            processed = min(idx + batch_size, total_images)
            print(f"Processed {processed}/{total_images} images...", end="\r")

    print("\nCLIP feature extraction and comparison complete.")
    return results


def compute_other_similarity(
    target_path: Path,
    image_paths: List[Path],
    metric: str
) -> List[Tuple[Path, float]]:
    """
    dHash, Histogram, Pixel Diffのいずれかを用いて対象画像と画像群との類似度・距離を計算する。
    """
    try:
        target_image = Image.open(target_path)
    except Exception as e:
        print(f"Error opening target image {target_path}: {e}", file=sys.stderr)
        return []

    target_hash = None
    if metric == "dhash":
        target_hash = calculate_dhash(target_image)
        if target_hash is None:
            return []

    results = []
    total_images = len(image_paths)

    for idx, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                if metric == "dhash":
                    img_hash = calculate_dhash(img)
                    if img_hash is not None:
                        dist = hamming_distance(target_hash, img_hash)
                        results.append((path, float(dist)))
                elif metric == "histogram":
                    sim = compare_color_histograms(target_image, img)
                    results.append((path, sim))
                elif metric == "pixel_diff":
                    dist = compare_pixel_difference(target_image, img)
                    results.append((path, dist))
        except Exception as e:
            print(f"Error processing image {path}: {e}", file=sys.stderr)

        processed = idx + 1
        if processed % 10 == 0 or processed == total_images:
            print(f"Processed {processed}/{total_images} images...", end="\r")

    print("\nComparison complete.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定された画像と同一ディレクトリ内の画像群との類似度を計算し、類似度の近い順に採番して別ディレクトリにコピーします。"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="基準となる入力画像のパス"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="コピー先の出力ディレクトリパス"
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="clip",
        choices=["clip", "dhash", "histogram", "pixel_diff"],
        help="類似度計算方法 (default: clip)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIPモデル名 (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="使用するデバイス (cpu, cuda, mps 等。指定がない場合は自動検出)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="CLIP特徴量抽出時のバッチサイズ (default: 64)",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"Error: Input file {input_path} does not exist or is not a file.", file=sys.stderr)
        sys.exit(1)

    input_dir = input_path.parent
    output_dir = Path(args.output_dir).resolve()

    # コピー先とコピー元が同じディレクトリになるのを防ぐ
    if input_dir == output_dir:
        print("Error: Output directory cannot be the same as the input image directory.", file=sys.stderr)
        sys.exit(1)

    # 同一ディレクトリ内の他画像一覧を取得
    image_paths = []
    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            if file.resolve() != input_path:
                image_paths.append(file)

    if not image_paths:
        print("No other images found in the same directory.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(image_paths)} other images in {input_dir}")
    print(f"Target image: {input_path.name}")
    print(f"Metric: {args.metric}")

    # 類似度計算の実行
    if args.metric == "clip":
        results = compute_clip_similarity(
            input_path,
            image_paths,
            args.model_name,
            args.device,
            args.batch_size
        )
    else:
        results = compute_other_similarity(
            input_path,
            image_paths,
            args.metric
        )

    if not results:
        print("No similarity results generated.", file=sys.stderr)
        sys.exit(1)

    # metricに応じたソート順の決定 (類似度なら降順、距離なら昇順)
    reverse_sort = True
    if args.metric in ["dhash", "pixel_diff"]:
        reverse_sort = False

    sorted_results = sorted(results, key=lambda x: x[1], reverse=reverse_sort)

    # コピー先ディレクトリの用意
    output_dir.mkdir(parents=True, exist_ok=True)

    # コピーと採番
    print(f"Copying files to {output_dir}...")

    # 基準画像を0000としてコピー
    target_ext = input_path.suffix
    target_dest = output_dir / f"0000{target_ext}"
    try:
        shutil.copy2(input_path, target_dest)
    except Exception as e:
        print(f"Error copying target image {input_path.name} to 0000{target_ext}: {e}", file=sys.stderr)

    for idx, (path, val) in enumerate(sorted_results):
        ext = path.suffix
        new_name = f"{idx + 1:04d}{ext}"
        dest_path = output_dir / new_name
        try:
            shutil.copy2(path, dest_path)
        except Exception as e:
            print(f"Error copying {path.name} to {new_name}: {e}", file=sys.stderr)

    print(f"Done. Successfully sorted and copied {len(sorted_results) + 1} images (including target as 0000) to {output_dir}")


if __name__ == "__main__":
    main()
