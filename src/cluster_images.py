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

import urllib.request
from typing import Tuple, Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

from tqdm import tqdm
import torch.nn.functional as F

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def download_yolo_model(dest_path: Path) -> Path:
    """
    顔検出用のYOLOv8モデルをHugging Faceからダウンロードする。
    """
    if dest_path.exists():
        return dest_path

    model_name = dest_path.name
    url = f"https://huggingface.co/Bingsu/adetailer/resolve/main/{model_name}"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"YOLOモデルをダウンロード中: {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"モデルのダウンロードが完了しました: {dest_path}")
    except Exception as e:
        print(f"モデルのダウンロード中にエラーが発生しました: {e}", file=sys.stderr)
        raise e
    return dest_path


def detect_and_crop_character(
    image_path: Path,
    model_path: Path,
    fallback_to_full_image: bool = True,
    margin: float = 0.2,
) -> Image.Image:
    """
    YOLOv8を用いて画像から顔領域を検出し、最大のバウンディングボックスをクロップした画像を返す。
    """
    try:
        model = YOLO(str(model_path))
        results = model(str(image_path), verbose=False)

        if not results or len(results[0].boxes) == 0:
            if fallback_to_full_image:
                return Image.open(image_path).convert("RGB")
            return None

        max_area = 0.0
        best_box = None

        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            if area > max_area:
                max_area = area
                best_box = xyxy

        if best_box:
            img = Image.open(image_path).convert("RGB")
            xmin, ymin, xmax, ymax = best_box
            width, height = img.size

            w = xmax - xmin
            h = ymax - ymin

            margin_x = w * margin
            margin_y = h * margin

            xmin = max(0, int(xmin - margin_x))
            ymin = max(0, int(ymin - margin_y))
            xmax = min(width, int(xmax + margin_x))
            ymax = min(height, int(ymax + margin_y))

            cropped_img = img.crop((xmin, ymin, xmax, ymax))
            return cropped_img

        if fallback_to_full_image:
            return Image.open(image_path).convert("RGB")
        return None
    except Exception as e:
        print(
            f"キャラクター検出中にエラーが発生しました ({image_path.name}): {e}",
            file=sys.stderr,
        )
        if fallback_to_full_image:
            try:
                return Image.open(image_path).convert("RGB")
            except Exception:
                raise e
        return None


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        use_detection: bool = False,
        yolo_model_path: Path = None,
        fallback_to_full_image: bool = True,
        crop_margin: float = 0.2,
    ):
        self.image_paths = image_paths
        self.use_detection = use_detection
        self.yolo_model_path = yolo_model_path
        self.fallback_to_full_image = fallback_to_full_image
        self.crop_margin = crop_margin

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        path = self.image_paths[idx]
        try:
            if self.use_detection and self.yolo_model_path:
                image = detect_and_crop_character(
                    path,
                    self.yolo_model_path,
                    fallback_to_full_image=self.fallback_to_full_image,
                    margin=self.crop_margin,
                )
            else:
                image = Image.open(path).convert("RGB")
            return image, str(path)
        except Exception as e:
            print(f"画像の読み込みに失敗しました ({path}): {e}", file=sys.stderr)
            return None, str(path)


def collate_fn_with_processor(
    batch: List[Tuple[Any, str]], processor: CLIPProcessor
) -> Tuple[Any, List[str]]:
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
    yolo_model_path: Path = None,
    fallback_to_full_image: bool = True,
    crop_margin: float = 0.2,
) -> Tuple[torch.Tensor, List[str]]:
    dataset = ImageDataset(
        image_paths,
        use_detection,
        yolo_model_path,
        fallback_to_full_image=fallback_to_full_image,
        crop_margin=crop_margin,
    )

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

    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_paths in tqdm(dataloader, desc="CLIP特徴量抽出"):
            if batch_inputs is None:
                continue

            pixel_values = batch_inputs["pixel_values"].to(device)
            outputs = model.get_image_features(pixel_values=pixel_values)
            image_features = (
                outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            )
            image_features = F.normalize(image_features, dim=-1)

            all_features.append(image_features.cpu())
            valid_paths.extend(batch_paths)

    if len(all_features) == 0:
        return torch.empty(0), []

    return torch.cat(all_features, dim=0), valid_paths


FACENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def load_facenet_model(device: torch.device) -> InceptionResnetV1:
    try:
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        return model
    except Exception as e:
        print(f"Facenetモデルのロード中にエラーが発生しました: {e}", file=sys.stderr)
        raise e


def collate_fn_facenet(batch: List[Tuple[Any, str]]) -> Tuple[torch.Tensor, List[str]]:
    valid_batch = [item for item in batch if item[0] is not None]
    if len(valid_batch) == 0:
        return None, []

    tensors = []
    paths = []
    for img, path in valid_batch:
        try:
            tensor = FACENET_TRANSFORM(img)
            tensors.append(tensor)
            paths.append(path)
        except Exception as e:
            print(f"画像の前処理に失敗しました ({path}): {e}", file=sys.stderr)

    if len(tensors) == 0:
        return None, []

    inputs = torch.stack(tensors)
    return inputs, paths


def extract_facenet_features(
    image_paths: List[Path],
    model: InceptionResnetV1,
    device: torch.device,
    batch_size: int = 64,
    use_detection: bool = False,
    yolo_model_path: Path = None,
    fallback_to_full_image: bool = True,
    crop_margin: float = 0.2,
) -> Tuple[torch.Tensor, List[str]]:
    dataset = ImageDataset(
        image_paths,
        use_detection=use_detection,
        yolo_model_path=yolo_model_path,
        fallback_to_full_image=fallback_to_full_image,
        crop_margin=crop_margin,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_facenet,
    )

    all_features = []
    valid_paths = []

    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_paths in tqdm(dataloader, desc="FaceNet特徴量抽出"):
            if batch_inputs is None:
                continue

            batch_inputs = batch_inputs.to(device)
            outputs = model(batch_inputs)
            image_features = F.normalize(outputs, dim=-1)

            all_features.append(image_features.cpu())
            valid_paths.extend(batch_paths)

    if len(all_features) == 0:
        return torch.empty(0), []

    return torch.cat(all_features, dim=0), valid_paths


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

    action_label = "移動中" if action == "move" else "コピー中"
    print(f"\nクラスタリングされた画像を {output_root} に{action_label}...")

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
            print(f"ノイズ (未分類): {len(sorted_pairs)} 枚")
        else:
            # クラスタの平均ベクトル（セントロイド）を計算し、L2正規化
            centroid = cluster_features.mean(dim=0)
            centroid = centroid / centroid.norm(dim=-1, keepdim=True)

            # セントロイドとの類似度（近さ）を計算して降順ソート
            similarities = torch.matmul(cluster_features, centroid).tolist()
            path_sim_pairs = list(zip(cluster_paths, similarities))
            sorted_pairs = sorted(path_sim_pairs, key=lambda x: x[1], reverse=True)
            print(f"クラスタ {k:02d}: {len(sorted_pairs)} 枚")

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
                err_action = "移動" if action == "move" else "コピー"
                print(
                    f"エラー: {src_path.name} から {dest_path.name} への{err_action}に失敗しました: {e}",
                    file=sys.stderr,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定フォルダ直下の画像を階層的クラスタリングでクラスタリングし、クラスタごとに採番して別のフォルダにコピーまたは移動します。"
    )

    # グループ定義
    io_group = parser.add_argument_group("基本設定 (Input/Output & General Settings)")
    feature_group = parser.add_argument_group(
        "顔検出・特徴量抽出設定 (Feature Extraction & Face Detection)"
    )
    cluster_group = parser.add_argument_group(
        "クラスタリング設定 (Clustering Settings)"
    )

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
            f"エラー: 入力ディレクトリ {input_dir} が存在しないか、ディレクトリではありません。",
            file=sys.stderr,
        )
        sys.exit(1)

    if input_dir == output_dir:
        print(
            "エラー: 出力ディレクトリは入力ディレクトリと同じにすることはできません。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 入力フォルダ内の画像一覧を取得
    image_paths = []
    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(file)

    if not image_paths:
        print(f"エラー: {input_dir} に有効な画像が見つかりません。", file=sys.stderr)
        sys.exit(0)

    print(f"{input_dir} に {len(image_paths)} 枚の画像が見つかりました")

    # 最適なデバイスの自動検出
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"使用デバイス: {device}")

    if args.feature_type == "facenet" and not args.use_detection:
        print(
            "警告: Facenet には顔検出が必要です。--use-detection を True に設定します。",
            file=sys.stderr,
        )
        args.use_detection = True

    # YOLOモデルの準備（use_detection が有効な場合のみ）
    yolo_model_path = Path(args.yolo_model).resolve()
    if args.use_detection:
        try:
            download_yolo_model(yolo_model_path)
        except Exception as e:
            print(
                f"YOLOモデルの準備に失敗しました。検出なしモードにフォールバックします: {e}",
                file=sys.stderr,
            )
            args.use_detection = False
            if args.feature_type == "facenet":
                print(
                    "Facenet には顔検出が必要です。CLIP特徴量抽出にフォールバックします。",
                    file=sys.stderr,
                )
                args.feature_type = "clip"

    # feature_type に基づいてモデルをロードし特徴量を抽出
    if args.feature_type == "facenet":
        print("Facenetモデルを読み込み中...")
        try:
            model = load_facenet_model(device)
        except Exception as e:
            print(f"Facenetモデルの読み込みエラー: {e}", file=sys.stderr)
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
        print(f"CLIPモデルを読み込み中: {args.model_name}...")
        try:
            model = CLIPModel.from_pretrained(args.model_name).to(device)
            processor = CLIPProcessor.from_pretrained(args.model_name)
        except Exception as e:
            print(f"CLIPモデルの読み込みエラー: {e}", file=sys.stderr)
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
        print(
            f"{skipped_count} 枚の画像をスキップしました（顔検出失敗または無効な画像）。"
        )

    if len(valid_paths) == 0:
        print("特徴量が抽出されませんでした。", file=sys.stderr)
        sys.exit(1)

    # クラスタリングの実行
    features_numpy = features.numpy()
    print(f"階層的クラスタリングで {len(valid_paths)} 枚の画像をクラスタリング中...")
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
        print(f"クラスタリング中にエラーが発生しました: {e}", file=sys.stderr)
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

    action_past_verb = "移動" if args.action == "move" else "コピー"
    print(
        f"完了。画像を正常にクラスタリングし、{output_dir} に{action_past_verb}しました。"
    )


if __name__ == "__main__":
    main()
