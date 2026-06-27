"""
画像を写真、イラスト、漫画に分類するスクリプト
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="CLIPを用いて画像を写真とイラストに分類します。"
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
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"エラー: '{images_dir}' ディレクトリが見つかりません。")
        return

    # 対応する画像拡張子
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # 画像ファイルのリストを取得
    image_paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
    )

    if not image_paths:
        print(f"'{images_dir}' 内に有効な画像ファイルが見つかりませんでした。")
        return

    print(f"{len(image_paths)} 枚の画像が見つかりました。CLIPモデルをロード中...")

    # デバイスの設定 (Apple Silicon Macなら mps, CUDAなら cuda, なければ cpu)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用デバイス: {device}")

    # CLIPモデルの読み込み
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

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
        "a manga drawing",
        "a black and white manga page",
        "a monochrome comic page",
        "manga panel",
        "comic book art",
        "japanese manga",
    ]

    print("テキスト特徴量を抽出中...")

    # 写真プロンプトの特徴量抽出と平均化
    photo_inputs = processor(text=photo_prompts, padding=True, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        photo_outputs = model.get_text_features(**photo_inputs)
        photo_features = photo_outputs.pooler_output
        photo_features = photo_features / photo_features.norm(p=2, dim=-1, keepdim=True)
        mean_photo_feature = photo_features.mean(dim=0)
        mean_photo_feature = mean_photo_feature / mean_photo_feature.norm(
            p=2, dim=-1, keepdim=True
        )

    # イラストプロンプトの特徴量抽出と平均化
    illust_inputs = processor(
        text=illust_prompts, padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        illust_outputs = model.get_text_features(**illust_inputs)
        illust_features = illust_outputs.pooler_output
        illust_features = illust_features / illust_features.norm(
            p=2, dim=-1, keepdim=True
        )
        mean_illust_feature = illust_features.mean(dim=0)
        mean_illust_feature = mean_illust_feature / mean_illust_feature.norm(
            p=2, dim=-1, keepdim=True
        )

    # 漫画プロンプトの特徴量抽出と平均化
    manga_inputs = processor(text=manga_prompts, padding=True, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        manga_outputs = model.get_text_features(**manga_inputs)
        manga_features = manga_outputs.pooler_output
        manga_features = manga_features / manga_features.norm(p=2, dim=-1, keepdim=True)
        mean_manga_feature = manga_features.mean(dim=0)
        mean_manga_feature = mean_manga_feature / mean_manga_feature.norm(
            p=2, dim=-1, keepdim=True
        )

    # 基準テキスト特徴量の結合 (shape: 3, embed_dim)
    text_features = torch.stack(
        [mean_photo_feature, mean_illust_feature, mean_manga_feature]
    )

    print("\n--- 分類結果 ---")
    # ヘッダー出力
    print(
        f"{'ファイル名':<50} | {'写真 (Photo)':<12} | {'イラスト (Illustration)':<22} | {'漫画 (Manga)':<12}"
    )
    print("-" * 105)

    for path in image_paths:
        try:
            # 画像の読み込みとRGB変換
            with Image.open(path) as img:
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

                print(
                    f"{path.name:<50} | {photo_prob:12.2%} | {illust_prob:22.2%} | {manga_prob:12.2%}"
                )

        except Exception as e:
            print(f"エラー（ファイル名: {path.name}）: {e}")


if __name__ == "__main__":
    main()
