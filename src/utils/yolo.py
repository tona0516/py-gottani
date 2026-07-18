"""
YOLO関連のユーティリティモジュール。
YOLOモデルのダウンロードおよびキャラクター検出・クロップ機能を提供する。
"""

import sys
import urllib.request
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def download_yolo_model(dest_path: Path) -> Path:
    """
    顔検出用のYOLOv8モデル（または指定されたモデル）をHugging Faceからダウンロードする。
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
    YOLOv8を用いて画像から顔領域（またはオブジェクト等）を検出し、
    最大のバウンディングボックスをクロップした画像を返す。
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
            xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
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

            # マージンを追加してクロップ範囲を設定する（負の値も許容）
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
