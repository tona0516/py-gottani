from datetime import datetime
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import random
import shutil
import time
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return image, str(path)
        except Exception as e:
            print(f"画像の読み込みに失敗しました ({path}): {e}", file=sys.stderr)
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
) -> Tuple[torch.Tensor, List[str]]:
    """
    CLIPを用いて画像群の特徴量を一括抽出する。
    """
    dataset = ImageDataset(image_paths)

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
            image_features = (
                outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_features.append(image_features.cpu())
            valid_paths.extend(batch_paths)

            processed_images += len(batch_paths)
            elapsed = time.time() - start_time
            speed = processed_images / elapsed if elapsed > 0 else 0
            eta = (total_images - processed_images) / speed if speed > 0 else 0

            print(
                f"処理済み {processed_images}/{total_images} 画像 ({processed_images/total_images*100:.1f}%) | "
                f"速度: {speed:.1f} img/s | 残り: {eta:.1f}s",
                end="\r",
            )

    print("\n特徴量抽出が完了しました。")

    if len(all_features) == 0:
        return torch.empty(0), []

    return torch.cat(all_features, dim=0), valid_paths


def evaluate_centroids(
    train_features_illust: torch.Tensor,
    train_features_photo: torch.Tensor,
    val_features_illust: torch.Tensor,
    val_features_photo: torch.Tensor,
) -> Dict[str, float]:

    # 重心（重心ベクトル）を算出し、正規化する
    c_illust = train_features_illust.mean(dim=0)
    c_illust = c_illust / c_illust.norm(dim=-1, keepdim=True)

    c_photo = train_features_photo.mean(dim=0)
    c_photo = c_photo / c_photo.norm(dim=-1, keepdim=True)

    # illust検証データの予測
    sim_illust_to_illust = torch.matmul(val_features_illust, c_illust)
    sim_illust_to_photo = torch.matmul(val_features_illust, c_photo)
    pred_illust_is_illust = sim_illust_to_illust > sim_illust_to_photo
    tp = pred_illust_is_illust.sum().item()  # 正しくイラストと判定
    fn = (~pred_illust_is_illust).sum().item()  # イラストなのに写真と判定

    # photo検証データの予測
    sim_photo_to_illust = torch.matmul(val_features_photo, c_illust)
    sim_photo_to_photo = torch.matmul(val_features_photo, c_photo)
    pred_photo_is_photo = sim_photo_to_photo > sim_photo_to_illust
    tn = pred_photo_is_photo.sum().item()  # 正しく写真と判定
    fp = (~pred_photo_is_photo).sum().item()  # 写真なのにイラストと判定

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def k_fold_cross_validation(
    features_illust: torch.Tensor,
    features_photo: torch.Tensor,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)

    num_illust = len(features_illust)
    num_photo = len(features_photo)

    indices_illust = list(range(num_illust))
    indices_photo = list(range(num_photo))
    random.shuffle(indices_illust)
    random.shuffle(indices_photo)

    fold_size_illust = num_illust // n_splits
    fold_size_photo = num_photo // n_splits

    fold_metrics = []

    for fold in range(n_splits):
        val_start_illust = fold * fold_size_illust
        val_end_illust = (
            (fold + 1) * fold_size_illust if fold < n_splits - 1 else num_illust
        )

        val_start_photo = fold * fold_size_photo
        val_end_photo = (
            (fold + 1) * fold_size_photo if fold < n_splits - 1 else num_photo
        )

        val_idx_illust = indices_illust[val_start_illust:val_end_illust]
        val_idx_photo = indices_photo[val_start_photo:val_end_photo]

        train_idx_illust = (
            indices_illust[:val_start_illust] + indices_illust[val_end_illust:]
        )
        train_idx_photo = (
            indices_photo[:val_start_photo] + indices_photo[val_end_photo:]
        )

        train_feat_illust = features_illust[train_idx_illust]
        train_feat_photo = features_photo[train_idx_photo]

        val_feat_illust = features_illust[val_idx_illust]
        val_feat_photo = features_photo[val_idx_photo]

        metrics = evaluate_centroids(
            train_feat_illust, train_feat_photo, val_feat_illust, val_feat_photo
        )
        fold_metrics.append(metrics)

    return fold_metrics


def train_mode(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = args.model_name
    print(f"Loading CLIP model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 画像ファイル収集
    print("Collecting training images...")
    illust_dir = Path(args.images_dir) / "illust_manga"
    photo_dir = Path(args.images_dir) / "photo"

    illust_paths = [
        p for p in illust_dir.glob("**/*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    photo_paths = [
        p for p in photo_dir.glob("**/*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(
        f"Found {len(illust_paths)} illust_manga images and {len(photo_paths)} photo images."
    )

    if len(illust_paths) == 0 or len(photo_paths) == 0:
        print("Error: No images found. Check your directories.")
        sys.exit(1)

    # 全ての特徴抽出
    print("\n--- Extracting features for illust_manga ---")
    feat_illust, _ = extract_features(
        illust_paths, model, processor, device, args.batch_size
    )

    print("\n--- Extracting features for photo ---")
    feat_photo, _ = extract_features(
        photo_paths, model, processor, device, args.batch_size
    )

    # 交差検証の実行
    if args.n_splits > 1:
        print(f"\nEvaluating model using {args.n_splits}-fold cross-validation...")
        fold_metrics = k_fold_cross_validation(
            feat_illust, feat_photo, n_splits=args.n_splits, seed=args.seed
        )

        avg_metrics = {}
        for key in ["accuracy", "precision", "recall", "f1"]:
            avg_metrics[key] = sum(m[key] for m in fold_metrics) / len(fold_metrics)

        total_tp = sum(m["tp"] for m in fold_metrics)
        total_fp = sum(m["fp"] for m in fold_metrics)
        total_tn = sum(m["tn"] for m in fold_metrics)
        total_fn = sum(m["fn"] for m in fold_metrics)

        print(f"\nCross-Validation Results ({args.n_splits} folds average):")
        print(f"  Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall:    {avg_metrics['recall']:.4f}")
        print(f"  F1-Score:  {avg_metrics['f1']:.4f}")
        print(f"  Total Confusion Matrix (sum of all folds):")
        print(f"    TP={total_tp}, FP={total_fp}, TN={total_tn}, FN={total_fn}")

    # 全データを用いた重心の算出と保存
    print("\nCalculating final centroids using all images...")
    c_illust = feat_illust.mean(dim=0)
    c_illust = c_illust / c_illust.norm(dim=-1, keepdim=True)

    c_photo = feat_photo.mean(dim=0)
    c_photo = c_photo / c_photo.norm(dim=-1, keepdim=True)

    model_data = {"illust_manga": c_illust, "photo": c_photo, "model_name": model_name}

    torch.save(model_data, args.output)
    print(f"Model saved to {args.output}")


def move_images(results: List[Dict[str, Any]], dest_root: Path) -> None:
    today_str = datetime.today().strftime("%Y%m%d")
    illust_dir = dest_root / f"illust_manga_{today_str}"
    photo_dir = dest_root / f"photo_{today_str}"

    illust_dir.mkdir(parents=True, exist_ok=True)
    photo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMoving images to classification directories under {dest_root}...")
    moved_count = 0
    for r in results:
        src_path = Path(r["path"])
        label = r["label"]

        if label == "illust_manga":
            target_dir = illust_dir
        else:
            target_dir = photo_dir

        dst_path = target_dir / src_path.name

        # 同名ファイルが存在する場合の競合対策
        if dst_path.exists():
            base = src_path.stem
            ext = src_path.suffix
            counter = 1
            while dst_path.exists():
                dst_path = target_dir / f"{base}_{counter}{ext}"
                counter += 1

        try:
            shutil.move(str(src_path), str(dst_path))
            moved_count += 1
        except Exception as e:
            print(f"Error moving {src_path} to {dst_path}: {e}")

    print(f"Successfully moved {moved_count}/{len(results)} images.")


def predict_mode(args: argparse.Namespace) -> None:
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} does not exist. Run 'train' mode first.")
        sys.exit(1)

    print(f"Loading centroids from {args.model}...")
    model_data = torch.load(args.model, map_location="cpu")
    c_illust = model_data["illust_manga"]
    c_photo = model_data["photo"]
    model_name = model_data["model_name"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} with CLIP: {model_name}")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 重心をデバイスへ転送
    c_illust = c_illust.to(device)
    c_photo = c_photo.to(device)

    # 対象ファイル収集
    input_path = Path(args.input)
    image_paths = []

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(input_path)
    elif input_path.is_dir():
        image_paths = [
            p
            for p in input_path.glob("**/*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    if len(image_paths) == 0:
        print(f"Error: No valid images found at {args.input}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images to predict.")

    # 推論処理
    dataset = ImageDataset(image_paths)

    def custom_collate(batch):
        return collate_fn_with_processor(batch, processor)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate,
    )

    model.eval()
    results = []

    # 確率計算用のソフトマックス温度スケール (値が大きいほど確信度の高い結果になる)
    temperature = 20.0

    with torch.no_grad():
        for batch_inputs, batch_paths in dataloader:
            if batch_inputs is None:
                continue

            pixel_values = batch_inputs["pixel_values"].to(device)
            outputs = model.get_image_features(pixel_values=pixel_values)
            image_features = outputs.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 各重心とのコサイン類似度を計算
            sim_illust = torch.matmul(image_features, c_illust)
            sim_photo = torch.matmul(image_features, c_photo)

            # ソフトマックスによる確率スケーリング
            logits = torch.stack([sim_illust, sim_photo], dim=-1) * temperature
            probs = torch.softmax(logits, dim=-1)

            for path, prob, sim_i, sim_p in zip(
                batch_paths, probs, sim_illust, sim_photo
            ):
                p_illust = prob[0].item()
                p_photo = prob[1].item()

                label = "illust_manga" if p_illust > p_photo else "photo"
                confidence = p_illust if label == "illust_manga" else p_photo

                results.append(
                    {
                        "path": path,
                        "label": label,
                        "confidence": confidence,
                        "sim_illust": sim_i.item(),
                        "sim_photo": sim_p.item(),
                    }
                )

    # 結果の表示
    print("\n--- Prediction Results ---")
    for r in results[:50]:  # 最大50件表示
        print(
            f"{Path(r['path']).name}: {r['label']} (Confidence: {r['confidence']*100:.1f}%) | "
            f"Sim_Illust: {r['sim_illust']:.3f}, Sim_Photo: {r['sim_photo']:.3f}"
        )

    if len(results) > 50:
        print(f"... and {len(results) - 50} more images.")

    # 結果をテキストやCSVで出力するオプションがあれば便利
    if args.save_results:
        output_csv = Path(args.save_results)
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write(
                "filepath,predicted_label,confidence,similarity_illust,similarity_photo\n"
            )
            for r in results:
                f.write(
                    f'"{r["path"]}",{r["label"]},{r["confidence"]:.4f},{r["sim_illust"]:.4f},{r["sim_photo"]:.4f}\n'
                )
        print(f"Results saved to {output_csv}")

    # 画像の移動処理
    if args.move_to_dir:
        move_images(results, Path(args.move_to_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="画像分類ツール (イラスト・マンガ vs 写真)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="実行モード")

    # 学習モードのパーサー
    train_parser = subparsers.add_parser(
        "train", help="CLIP画像特徴量の重心を抽出・保存してモデルを学習します"
    )
    train_parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="illust_manga/ と photo/ サブディレクトリを含むルートディレクトリ",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default="centroids.pt",
        help="学習済み重心モデルの保存先パス",
    )
    train_parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="使用する事前学習済みCLIPモデル名",
    )
    train_parser.add_argument(
        "--n-splits", type=int, default=5, help="交差検証の分割数 (K-Fold)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64, help="特徴量抽出時のバッチサイズ"
    )
    train_parser.add_argument(
        "--seed", type=int, default=42, help="データ分割用のランダムシード値"
    )

    # 予測モードのパーサー
    predict_parser = subparsers.add_parser(
        "predict", help="保存された重心モデルを使用して画像のクラスを予測します"
    )
    predict_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="予測対象の画像ファイルまたは画像ディレクトリのパス",
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        default="centroids.pt",
        help="保存された重心モデルファイルのパス",
    )
    predict_parser.add_argument(
        "--batch-size", type=int, default=64, help="特徴量抽出時のバッチサイズ"
    )
    predict_parser.add_argument(
        "--save-results",
        type=str,
        default="",
        help="予測結果を保存するCSVファイルのパス",
    )
    predict_parser.add_argument(
        "--move-to-dir",
        type=str,
        default="",
        help="予測結果に基づいて画像を移動する先のルートディレクトリのパス",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "predict":
        predict_mode(args)


if __name__ == "__main__":
    main()
