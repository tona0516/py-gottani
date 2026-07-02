"""
指定されたフォルダ直下の画像をHDBSCANでクラスタリングし、クラスタごとに採番して別のフォルダにコピーするスクリプト。
"""

import os
import sys
import argparse
import shutil
import urllib.request
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def download_yolo_model(dest_path: Path) -> Path:
    """
    アニメ顔検出用のYOLOv8モデルをHugging Faceからダウンロードする。
    """
    url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"

    if dest_path.exists():
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YOLO model from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Model downloaded successfully and saved to {dest_path}")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        raise e
    return dest_path


def detect_and_crop_character(image_path: Path, model_path: Path) -> Image.Image:
    """
    YOLOv8を用いて画像からキャラクター（顔など）を検出し、
    最大のバウンディングボックスをクロップした画像を返す。
    検出されなかった場合は元の画像を返す。
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        # ultralyticsがインストールされていない場合はログを出力して元の画像を返す
        return Image.open(image_path).convert("RGB")

    try:
        model = YOLO(str(model_path))
        results = model(str(image_path), verbose=False)

        img = Image.open(image_path).convert("RGB")

        if not results or len(results[0].boxes) == 0:
            return img

        max_area = 0.0
        best_box = None

        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            if area > max_area:
                max_area = area
                best_box = xyxy

        if best_box:
            xmin, ymin, xmax, ymax = best_box
            width, height = img.size

            w = xmax - xmin
            h = ymax - ymin

            # マージン（20%）を追加してクロップ範囲を広げる
            margin_x = w * 0.2
            margin_y = h * 0.2

            xmin = max(0, int(xmin - margin_x))
            ymin = max(0, int(ymin - margin_y))
            xmax = min(width, int(xmax + margin_x))
            ymax = min(height, int(ymax + margin_y))

            cropped_img = img.crop((xmin, ymin, xmax, ymax))
            return cropped_img

        return img
    except Exception as e:
        print(f"Error during character detection on {image_path.name}: {e}", file=sys.stderr)
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            raise e


def convert_palette_to_rgba_if_needed(img: Image.Image) -> Image.Image:
    """
    Pモードで透過情報がある場合、RGBAに変換する（Pillowの警告回避）。
    """
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA")
    return img


# PyTorch / Transformersの依存確認
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("Error: torch or transformers is not installed. Please install them to run this script.", file=sys.stderr)
    sys.exit(1)

# scikit-learnの依存確認 (HDBSCANに必要)
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    print("Error: scikit-learn is not installed. Please run 'uv sync' or install it to use HDBSCAN.", file=sys.stderr)
    sys.exit(1)


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], use_detection: bool = False, yolo_model_path: Path = None):
        self.image_paths = image_paths
        self.use_detection = use_detection
        self.yolo_model_path = yolo_model_path

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        path = self.image_paths[idx]
        try:
            if self.use_detection and self.yolo_model_path:
                image = detect_and_crop_character(path, self.yolo_model_path)
            else:
                image = Image.open(path).convert("RGB")
            return image, str(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}", file=sys.stderr)
            return None, str(path)


def collate_fn_with_processor(
    batch: List[Tuple[Any, str]], processor: CLIPProcessor
) -> Tuple[Any, List[str]]:
    """
    読み込みに失敗した（Noneの）アイテムを除外し、複数画像をまとめてテンソル化する。
    """
    valid_batch = [item for item in batch if item[0] is not None]
    if len(valid_batch) == 0:
        return None, []
    images = [item[0] for item in valid_batch]
    paths = [item[1] for item in valid_batch]

    inputs = processor(images=images, return_tensors="pt")
    return inputs, paths


def extract_features(
    image_paths: List[Path],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int = 64,
    use_detection: bool = False,
    yolo_model_path: Path = None
) -> Tuple[torch.Tensor, List[str]]:
    """
    CLIPを用いて画像群の特徴量を一括抽出する。
    """
    dataset = ImageDataset(image_paths, use_detection, yolo_model_path)

    def custom_collate(batch):
        return collate_fn_with_processor(batch, processor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate,
    )

    all_features = []
    valid_paths = []

    total_images = len(image_paths)
    processed_images = 0
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_paths in dataloader:
            if batch_inputs is None:
                continue

            pixel_values = batch_inputs["pixel_values"].to(device)
            outputs = model.get_image_features(pixel_values=pixel_values)
            image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            # L2正規化して類似度計算をドット積のみにする
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_features.append(image_features.cpu())
            valid_paths.extend(batch_paths)

            processed_images += len(batch_paths)
            elapsed = time.time() - start_time
            speed = processed_images / elapsed if elapsed > 0 else 0
            eta = (total_images - processed_images) / speed if speed > 0 else 0

            print(
                f"Processed {processed_images}/{total_images} images ({processed_images/total_images*100:.1f}%) | "
                f"Speed: {speed:.1f} img/s | ETA: {eta:.1f}s",
                end="\r",
            )

    print("\nExtraction complete.")

    if len(all_features) == 0:
        return torch.empty(0), []

    return torch.cat(all_features, dim=0), valid_paths


def copy_clustered_images(
    image_paths: List[str],
    labels: torch.Tensor,
    features: torch.Tensor,
    output_root: Path,
    rename_format: str,
) -> None:
    """
    クラスタリングされた画像をフォルダ毎に分け、類似度順に採番してコピーする。
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # ユニークなクラスタラベルの一覧を取得
    unique_labels = torch.unique(labels)
    image_paths_np = [Path(p) for p in image_paths]

    print(f"\nCopying clustered images to {output_root}...")

    for k in unique_labels.tolist():
        indices = (labels == k).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue

        # フォルダ名の決定 (-1 の場合は noise)
        if k == -1:
            cluster_dir = output_root / "noise"
        else:
            cluster_dir = output_root / f"cluster_{k:02d}"

        cluster_dir.mkdir(parents=True, exist_ok=True)

        cluster_features = features[indices]
        cluster_paths = [image_paths_np[idx] for idx in indices]

        if k == -1:
            # ノイズの場合は類似度ソートをせず、ファイル名順でソート
            path_sim_pairs = list(zip(cluster_paths, [0.0] * len(cluster_paths)))
            sorted_pairs = sorted(path_sim_pairs, key=lambda x: x[0].name)
            print(f"Noise (Unclassified): {len(sorted_pairs)} images")
        else:
            # クラスタの平均ベクトル（セントロイド）を計算し、L2正規化
            centroid = cluster_features.mean(dim=0)
            centroid = centroid / centroid.norm(dim=-1, keepdim=True)

            # セントロイドとの類似度（近さ）を計算して降順ソート
            similarities = torch.matmul(cluster_features, centroid).tolist()
            path_sim_pairs = list(zip(cluster_paths, similarities))
            sorted_pairs = sorted(path_sim_pairs, key=lambda x: x[1], reverse=True)
            print(f"Cluster {k:02d}: {len(sorted_pairs)} images")

        for idx, (src_path, sim) in enumerate(sorted_pairs):
            ext = src_path.suffix
            orig_name = src_path.name

            if rename_format == "prefix":
                new_name = f"{idx + 1:04d}_{orig_name}"
            elif rename_format == "number":
                new_name = f"{idx + 1:04d}{ext}"
            else:  # "original"
                new_name = orig_name

            dest_path = cluster_dir / new_name

            if dest_path.exists():
                base = Path(new_name).stem
                counter = 1
                while dest_path.exists():
                    dest_path = cluster_dir / f"{base}_{counter}{ext}"
                    counter += 1

            try:
                shutil.copy2(src_path, dest_path)
            except Exception as e:
                print(f"Error copying {src_path.name} to {dest_path.name}: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定フォルダ直下の画像をHDBSCANでクラスタリングし、クラスタごとに採番して別のフォルダにコピーします。"
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True, help="入力画像フォルダ"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="コピー出力先フォルダ"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCANのクラスタとみなす最小サンプル数 (default: 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCANの密度判定用近傍点数。None の場合は min-cluster-size と同一になります (default: None)",
    )
    parser.add_argument(
        "--use-detection",
        type=bool,
        default=True,
        help="YOLOを用いてキャラクター顔領域を検出し、その特徴量でクラスタリングを行う (default: True)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="models/face_yolov8n.pt",
        help="YOLOモデルのパスまたは保存先 (default: models/face_yolov8n.pt)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIPモデル名 (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="特徴量抽出時のバッチサイズ (default: 256)",
    )
    parser.add_argument(
        "--rename-format",
        type=str,
        default="number",
        choices=["prefix", "number", "original"],
        help="ファイルリネーム形式。prefix: 0001_name.jpg, number: 0001.jpg, original: 元のまま (default: number)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    if input_dir == output_dir:
        print("Error: Output directory cannot be the same as the input directory.", file=sys.stderr)
        sys.exit(1)

    # 入力フォルダ内の画像一覧を取得
    image_paths = []
    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(file)

    if not image_paths:
        print(f"No valid images found in {input_dir}.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(image_paths)} images in {input_dir}")

    # 最適なデバイスの自動検出
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # YOLOモデルの準備
    yolo_model_path = Path(args.yolo_model).resolve()
    if args.use_detection:
        try:
            download_yolo_model(yolo_model_path)
        except Exception as e:
            print(f"Failed to prepare YOLO model, falling back to non-detection mode: {e}", file=sys.stderr)
            args.use_detection = False

    # CLIPモデルのロード
    print(f"Loading CLIP model: {args.model_name}...")
    try:
        model = CLIPModel.from_pretrained(args.model_name).to(device)
        processor = CLIPProcessor.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading CLIP model: {e}", file=sys.stderr)
        sys.exit(1)

    # 特徴量抽出
    features, valid_paths = extract_features(
        image_paths,
        model,
        processor,
        device,
        batch_size=args.batch_size,
        use_detection=args.use_detection,
        yolo_model_path=yolo_model_path if args.use_detection else None
    )

    if len(valid_paths) == 0:
        print("No features extracted.", file=sys.stderr)
        sys.exit(1)

    # HDBSCANによるクラスタリングの実行
    features_numpy = features.numpy()
    print(f"Clustering {len(valid_paths)} images using HDBSCAN...")
    try:
        clusterer = HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="cosine"
        )
        labels_numpy = clusterer.fit_predict(features_numpy)
        labels = torch.tensor(labels_numpy, dtype=torch.long)
    except Exception as e:
        print(f"Error during HDBSCAN clustering: {e}", file=sys.stderr)
        sys.exit(1)

    # コピー処理
    copy_clustered_images(
        valid_paths,
        labels,
        features,
        output_dir,
        args.rename_format
    )

    print(f"Done. Successfully clustered and copied images to {output_dir}")


if __name__ == "__main__":
    main()
