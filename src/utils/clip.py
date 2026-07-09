"""
CLIP特徴量抽出パイプラインを集約するモジュール。
ImageDataset、バッチ用collate関数、特徴量抽出関数を提供する。
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Any

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from utils.yolo import detect_and_crop_character


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        use_detection: bool = False,
        yolo_model_path: Path = None,
    ):
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
    use_detection: bool = False,
    yolo_model_path: Path = None,
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
            image_features = (
                outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
            )
            # L2正規化して類似度計算をドット積のみにする
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
