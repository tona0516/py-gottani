"""
指定されたフォルダ直下の画像をHDBSCANでクラスタリングし、クラスタごとに採番して別のフォルダにコピーするスクリプト。
"""

import sys
import argparse
import shutil
from pathlib import Path
from typing import List
import torch
from sklearn.cluster import HDBSCAN

from image_utils import SUPPORTED_EXTENSIONS
from yolo_utils import download_yolo_model
from clip_utils import extract_features
from transformers import CLIPProcessor, CLIPModel


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
                print(
                    f"Error copying {src_path.name} to {dest_path.name}: {e}",
                    file=sys.stderr,
                )


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
        print(
            f"Error: Input directory {input_dir} does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    if input_dir == output_dir:
        print(
            "Error: Output directory cannot be the same as the input directory.",
            file=sys.stderr,
        )
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
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # YOLOモデルの準備
    yolo_model_path = Path(args.yolo_model).resolve()
    if args.use_detection:
        try:
            download_yolo_model(yolo_model_path)
        except Exception as e:
            print(
                f"Failed to prepare YOLO model, falling back to non-detection mode: {e}",
                file=sys.stderr,
            )
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
        yolo_model_path=yolo_model_path if args.use_detection else None,
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
            metric="cosine",
        )
        labels_numpy = clusterer.fit_predict(features_numpy)
        labels = torch.tensor(labels_numpy, dtype=torch.long)
    except Exception as e:
        print(f"Error during HDBSCAN clustering: {e}", file=sys.stderr)
        sys.exit(1)

    # コピー処理
    copy_clustered_images(valid_paths, labels, features, output_dir, args.rename_format)

    print(f"Done. Successfully clustered and copied images to {output_dir}")


if __name__ == "__main__":
    main()
