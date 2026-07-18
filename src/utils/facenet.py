"""
Facenet (InceptionResnetV1) 関連のユーティリティモジュール。
顔画像から特徴量（512次元の埋め込みベクトル）を抽出する機能を提供する。
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

from utils.clip import ImageDataset


# Facenet用の画像前処理（160x160にリサイズし、テンソル化して[-1, 1]に正規化する）
FACENET_TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def load_facenet_model(device: torch.device) -> InceptionResnetV1:
    """
    Facenet (InceptionResnetV1) モデルをロードし、評価モードにする。
    """
    try:
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        return model
    except Exception as e:
        print(f"Facenetモデルのロード中にエラーが発生しました: {e}", file=sys.stderr)
        raise e


def collate_fn_facenet(
    batch: List[Tuple[Any, str]]
) -> Tuple[torch.Tensor, List[str]]:
    """
    読み込みに失敗した（Noneの）アイテムを除外し、複数画像をまとめてテンソル化（160x160）する。
    """
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
    """
    Facenetを用いて画像群の特徴量（顔埋め込み）を一括抽出する。
    """
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

    total_images = len(image_paths)
    processed_images = 0
    start_time = time.time()

    # InceptionResnetV1モデルを推論モードで使用
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_paths in dataloader:
            if batch_inputs is None:
                continue

            batch_inputs = batch_inputs.to(device)
            # 顔埋め込みベクトルの取得
            outputs = model(batch_inputs)
            
            # L2正規化して類似度計算をドット積のみにする
            image_features = outputs / outputs.norm(dim=-1, keepdim=True)

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
