"""
指定されたフォルダ直下の画像をクラスタリングし、クラスタごとに採番して別のフォルダに移動するスクリプト。
"""

import sys
import argparse
import shutil
from pathlib import Path
from typing import List
import torch
from sklearn.cluster import AgglomerativeClustering

from utils.image import SUPPORTED_EXTENSIONS
from utils.yolo import download_yolo_model
from utils.clip import extract_features
from transformers import CLIPProcessor, CLIPModel


def save_clustered_images(
    image_paths: List[str],
    labels: torch.Tensor,
    features: torch.Tensor,
    output_root: Path,
    rename_format: str,
    action: str = "copy",
) -> None:
    """
    クラスタリングされた画像をフォルダ毎に分け、類似度順に採番してコピーまたは移動する。
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # ユニークなクラスタラベルの一覧を取得
    unique_labels = torch.unique(labels)
    image_paths_np = [Path(p) for p in image_paths]

    action_label = "Moving" if action == "move" else "Copying"
    print(f"\n{action_label} clustered images to {output_root}...")

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
                if action == "move":
                    shutil.move(src_path, dest_path)
                else:
                    shutil.copy2(src_path, dest_path)
            except Exception as e:
                err_action = "moving" if action == "move" else "copying"
                print(
                    f"Error {err_action} {src_path.name} to {dest_path.name}: {e}",
                    file=sys.stderr,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定フォルダ直下の画像を階層的クラスタリングでクラスタリングし、クラスタごとに採番して別のフォルダにコピーまたは移動します。"
    )

    # グループ定義
    io_group = parser.add_argument_group("基本設定 (Input/Output & General Settings)")
    feature_group = parser.add_argument_group("顔検出・特徴量抽出設定 (Feature Extraction & Face Detection)")
    cluster_group = parser.add_argument_group("クラスタリング設定 (Clustering Settings)")

    # 基本設定 (IO)
    io_group.add_argument(
        "-i", "--input-dir", type=str, required=True, help="入力元の画像ディレクトリ"
    )
    io_group.add_argument(
        "-o", "--output-dir", type=str, required=True, help="出力先のディレクトリ"
    )
    io_group.add_argument(
        "--action",
        type=str,
        default="copy",
        choices=["copy", "move"],
        help="クラスタリング後の処理方法。copy: コピー, move: 移動 (default: copy)",
    )
    io_group.add_argument(
        "--rename-format",
        type=str,
        default="original",
        choices=["prefix", "number", "original"],
        help="ファイル名のリネーム形式。prefix: 連番+元名前, number: 連番のみ, original: 変更なし (default: original)",
    )

    # 特徴量・顔検出
    feature_group.add_argument(
        "--feature-type",
        type=str,
        default="facenet",
        choices=["clip", "facenet"],
        help="特徴量抽出に使用するモデル。facenet: 顔認識, clip: 汎用画像認識 (default: facenet)",
    )
    feature_group.add_argument(
        "--use-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="YOLOを用いた顔検出を行うかどうか。無効にする場合は --no-use-detection を指定 (default: True)",
    )
    feature_group.add_argument(
        "--yolo-model",
        type=str,
        default="models/face_yolov8n.pt",
        help="YOLO顔検出モデルの保存先パス (default: models/face_yolov8n.pt)",
    )
    feature_group.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP使用時の事前学習済みモデル名 (default: openai/clip-vit-base-patch32)",
    )
    feature_group.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="特徴量抽出時のバッチサイズ (default: 256)",
    )
    feature_group.add_argument(
        "--crop-margin",
        type=float,
        default=0.0,
        help="顔検出時にクロップする領域のマージン比率。負の値で内側、正の値で外側へ拡張 (default: 0.0)",
    )
    feature_group.add_argument(
        "--fallback-to-full-image",
        action="store_true",
        help="顔検出に失敗した際、画像全体をフォールバックとして使用 (default: False)",
    )

    # クラスタリング
    cluster_group.add_argument(
        "--distance-threshold",
        type=float,
        default=0.60,
        help="Agglomerative Clustering のコサイン距離しきい値。小さいほど類似度の判定が厳しくなります (default: 0.60)",
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

    if args.feature_type == "facenet" and not args.use_detection:
        print("Warning: Facenet requires face detection. Forcing --use-detection to True.", file=sys.stderr)
        args.use_detection = True

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
            if args.feature_type == "facenet":
                print("Facenet requires face detection. Falling back to CLIP feature extraction.", file=sys.stderr)
                args.feature_type = "clip"

    # モデルのロードと特徴量抽出
    if args.feature_type == "facenet":
        from utils.facenet import load_facenet_model, extract_facenet_features
        print("Loading Facenet model...")
        try:
            model = load_facenet_model(device)
        except Exception as e:
            print(f"Error loading Facenet model: {e}", file=sys.stderr)
            sys.exit(1)

        features, valid_paths = extract_facenet_features(
            image_paths,
            model,
            device,
            batch_size=args.batch_size,
            use_detection=args.use_detection,
            yolo_model_path=yolo_model_path if args.use_detection else None,
            fallback_to_full_image=args.fallback_to_full_image,
            crop_margin=args.crop_margin,
        )
    else:
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
            fallback_to_full_image=args.fallback_to_full_image,
            crop_margin=args.crop_margin,
        )

    # 除外されたファイル数を表示
    skipped_count = len(image_paths) - len(valid_paths)
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images (face detection failed or invalid).")

    if len(valid_paths) == 0:
        print("No features extracted.", file=sys.stderr)
        sys.exit(1)

    # クラスタリングの実行
    features_numpy = features.numpy()
    print(f"Clustering {len(valid_paths)} images using agglomerative...")
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="complete",
            distance_threshold=args.distance_threshold,
        )
        labels_numpy = clusterer.fit_predict(features_numpy)
        labels = torch.tensor(labels_numpy, dtype=torch.long)
    except Exception as e:
        print(f"Error during clustering: {e}", file=sys.stderr)
        sys.exit(1)

    # コピーまたは移動処理
    save_clustered_images(
        valid_paths,
        labels,
        features,
        output_dir,
        args.rename_format,
        action=args.action,
    )

    action_past_verb = "copied" if args.action == "copy" else "moved"
    print(f"Done. Successfully clustered and {action_past_verb} images to {output_dir}")


if __name__ == "__main__":
    main()
