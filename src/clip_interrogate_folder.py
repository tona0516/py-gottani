"""
フォルダ内の画像から CLIP Interrogator を使用して Stable Diffusion 風プロンプトを生成し、
出現フレーズの頻度を集計するスクリプト。
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import List

from clip_interrogator import Config, Interrogator
from PIL import Image
import torch
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(input_dir: Path, recursive: bool = False) -> List[Path]:
    """指定フォルダから画像ファイルのリストを取得する."""
    if not input_dir.exists() or not input_dir.is_dir():
        print(
            f"エラー: フォルダ '{input_dir}' が存在しないかディレクトリではありません。",
            file=sys.stderr,
        )
        return []

    pattern = "**/*" if recursive else "*"
    image_paths = [
        p
        for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(image_paths)


def parse_prompt(prompt: str) -> List[str]:
    """
    プロンプト文字列からカンマ区切りのフレーズを抽出する.

    Returns:
        List[str]: フレーズのリスト
    """
    return [phrase.strip().lower() for phrase in prompt.split(",") if phrase.strip()]


def analyze_folder(
    input_dir: Path,
    output_file: Path,
    clip_model: str = "ViT-L-14/openai",
    caption_model: str | None = None,
    recursive: bool = False,
) -> None:
    """指定フォルダ内の画像を処理し、プロンプトの抽出と頻度集計を行う."""
    image_paths = find_images(input_dir, recursive=recursive)
    if not image_paths:
        print(f"指定されたフォルダ '{input_dir}' に対象画像が見つかりませんでした。")
        return

    print(f"検出された画像数: {len(image_paths)} 枚")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    print(f"CLIPモデル '{clip_model}' を初期化中...")

    # CLIP Interrogator の初期化
    config_kwargs = {"clip_model_name": clip_model, "device": device}
    if caption_model:
        config_kwargs["caption_model_name"] = caption_model

    config = Config(**config_kwargs)
    ci = Interrogator(config)

    results = []
    phrase_counter = Counter()

    print("画像を解析中...")
    for img_path in tqdm(image_paths, desc="Interrogating"):
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                prompt = ci.interrogate(img_rgb)
        except Exception as e:
            print(
                f"\n画像 '{img_path.name}' の処理中にエラーが発生しました: {e}",
                file=sys.stderr,
            )
            continue

        phrases = parse_prompt(prompt)
        phrase_counter.update(phrases)

        results.append(
            {
                "image_path": str(img_path),
                "image_name": img_path.name,
                "prompt": prompt,
                "phrases": phrases,
            }
        )

    # 結果の表示
    print("\n" + "=" * 60)
    print("【解析結果サマリー】")
    print(f"処理完了画像数: {len(results)} / {len(image_paths)}")
    print("=" * 60)

    print("\n■ カンマ区切りフレーズ / タグの出現頻度:")
    print("-" * 50)
    for rank, (phrase, count) in enumerate(phrase_counter.most_common(), 1):
        print(f"{rank:2d}. {phrase:<40} : {count} 回")

    # ファイル出力が指定されている場合
    output_file.parent.mkdir(parents=True, exist_ok=True)
    export_data = {
        "folder": str(input_dir),
        "total_images": len(results),
        "clip_model": clip_model,
        "phrases": dict(phrase_counter.most_common()),
        "details": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果を JSON ファイルに保存しました: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLIP Interrogator でフォルダ内の画像からSDプロンプトを生成し、出現頻度を集計します。"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="対象の画像が含まれるフォルダのパス",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        required=True,
        help="解析結果（JSON形式）を出力する保存先パス",
    )
    parser.add_argument(
        "-m",
        "--clip-model",
        type=str,
        default="ViT-L-14/openai",
        help="使用するCLIPモデル名 (デフォルト: ViT-L-14/openai)",
    )
    parser.add_argument(
        "--caption-model",
        type=str,
        default=None,
        help="使用するキャプションモデル名 (例: blip-large, blip-base など)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="サブフォルダも含めて再帰的に画像を検索する",
    )

    args = parser.parse_args()
    analyze_folder(
        input_dir=args.input_dir,
        output_file=args.output_file,
        clip_model=args.clip_model,
        caption_model=args.caption_model,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
