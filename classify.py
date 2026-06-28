"""
画像を写真、イラスト、漫画に分類するスクリプト
"""

import argparse
from pathlib import Path
import shutil
import unicodedata
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


def get_east_asian_width(text: str) -> int:
    """文字の表示幅（全角=2, 半角=1）を計算する"""
    width = 0
    for char in text:
        status = unicodedata.east_asian_width(char)
        if status in ("W", "F", "A"):
            width += 2
        else:
            width += 1
    return width


def pad_text(
    text: str, target_width: int, fillchar: str = " ", align: str = "<"
) -> str:
    """マルチバイト文字を考慮してパディングを行う"""
    current_width = get_east_asian_width(text)
    padding_needed = max(0, target_width - current_width)
    if align == "<":
        return text + (fillchar * padding_needed)
    elif align == ">":
        return (fillchar * padding_needed) + text
    else:
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        return (fillchar * left_padding) + text + (fillchar * right_padding)


def get_device() -> torch.device:
    """デバイスの設定 (Apple Silicon Macなら mps, CUDAなら cuda, なければ cpu)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    model_name: str, device: torch.device
) -> tuple[CLIPModel, CLIPProcessor]:
    """CLIPモデルとプロセッサの読み込み"""
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def get_mean_text_feature(
    prompts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """テキスト特徴量の抽出と平均化、およびL2正規化"""
    inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        features = outputs.pooler_output
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        mean_feature = features.mean(dim=0)
        mean_feature = mean_feature / mean_feature.norm(p=2, dim=-1, keepdim=True)
    return mean_feature


def get_image_paths(images_dir: Path) -> list[Path]:
    """画像ディレクトリから対象となる画像ファイルのリストを取得"""
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not images_dir.exists() or not images_dir.is_dir():
        return []
    return sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
    )


def classify_images(
    image_paths: list[Path],
    text_features: torch.Tensor,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> None:
    """画像の読み込み、特徴量抽出、類似度計算、確率計算、結果出力、およびフォルダ移動"""
    print("\n--- 分類結果 ---")
    # ヘッダー出力
    header_name = pad_text("File", 50)
    print(f"{header_name} | {'Photo':<12} | {'Illustration':<12} | {'Manga':<12} | {'Destination':<12}")
    print("-" * 110)

    for path in image_paths:
        try:
            # 画像の読み込みとRGB変換
            with Image.open(path) as img:
                if img.mode == "P" and "transparency" in img.info:
                    img = img.convert("RGBA")
                img_rgb = img.convert("RGB")

                # 画像の前処理
                image_inputs = processor(images=img_rgb, return_tensors="pt").to(device)

                # 特徴量の抽出と類似度の計算
                with torch.no_grad():
                    image_outputs = model.get_image_features(**image_inputs)
                    image_features = image_outputs.pooler_output
                    # L2正規化
                    image_features = image_features / image_features.norm(
                        p=2, dim=-1, keepdim=True
                    )

                    # 類似度の計算（内積）とスケール調整
                    logit_scale = model.logit_scale.exp()
                    logits_per_image = logit_scale * (
                        image_features @ text_features.t()
                    )

                    # 確率の計算 (softmax)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

                photo_prob = probs[0]
                illust_prob = probs[1]
                manga_prob = probs[2]

                # 確率の1位と2位の差が10%以内の場合はpendingフォルダに分類
                classes = ["photo", "illustration", "manga"]
                prob_class_pairs = list(zip(probs, classes))
                prob_class_pairs.sort(key=lambda x: x[0], reverse=True)

                first_prob, first_class = prob_class_pairs[0]
                second_prob, second_class = prob_class_pairs[1]

                if (first_prob - second_prob) <= 0.10:
                    dest_folder = "pending"
                else:
                    dest_folder = first_class

                padded_name = pad_text(path.name, 50)
                print(
                    f"{padded_name} | {photo_prob:12.1%} | {illust_prob:12.1%} | {manga_prob:12.1%} | {dest_folder:<12}"
                )

                # フォルダの作成とファイルの移動
                dest_dir = path.parent / dest_folder
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest_dir / path.name))

        except Exception as e:
            print(f"エラー（ファイル名: {path.name}）: {e}")


def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="CLIPを用いて画像を写真とイラスト、漫画に分類します。"
    )
    parser.add_argument(
        "--dir",
        nargs=1,
        default=["images"],
        help="画像が保存されているディレクトリのパス (デフォルト: images)",
    )
    args = parser.parse_args()

    # 画像ディレクトリの設定
    images_dir = Path(args.dir[0])
    image_paths = get_image_paths(images_dir)

    if not image_paths:
        print(f"'{images_dir}' 内に有効な画像ファイルが見つかりませんでした。")
        return

    print(f"{len(image_paths)} 枚の画像が見つかりました。CLIPモデルをロード中...")

    # デバイスの設定
    device = get_device()
    print(f"使用デバイス: {device}")

    # CLIPモデルの読み込み
    model_name = "openai/clip-vit-base-patch32"
    model, processor = load_model(model_name, device)

    # 判定用のプロンプト定義（プロンプト・アンサンブル）
    photo_prompts = [
        "a photograph",
        "a real life photo",
        "a realistic photograph",
        "a picture taken with a camera",
        "a snapshot of real life",
        "a photo of a real scene",
        "a camera photo",
        "a color photo",
    ]
    illust_prompts = [
        "an illustration",
        "an anime drawing",
        "a digital painting",
        "artwork",
        "a sketch",
        "a cartoon drawing",
        "fanart",
        "a line art drawing",
        "2d cg illustration",
        "a monochrome illustration",
    ]
    manga_prompts = [
        "a manga page with multiple panels",
        "a black and white manga page with panel layout",
        "monochrome comic strip with panel divisions",
        "manga panel with screentone shading",
        "black and white manga drawing with panels",
        "monochrome manga art with screen tones",
    ]

    print("テキスト特徴量を抽出中...")

    # 各プロンプトの特徴量抽出と平均化
    mean_photo_feature = get_mean_text_feature(photo_prompts, model, processor, device)
    mean_illust_feature = get_mean_text_feature(
        illust_prompts, model, processor, device
    )
    mean_manga_feature = get_mean_text_feature(manga_prompts, model, processor, device)

    # 基準テキスト特徴量の結合 (shape: 3, embed_dim)
    text_features = torch.stack(
        [mean_photo_feature, mean_illust_feature, mean_manga_feature]
    )

    # 画像の分類と結果出力
    classify_images(image_paths, text_features, model, processor, device)


if __name__ == "__main__":
    main()
