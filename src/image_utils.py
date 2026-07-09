"""
画像処理の汎用ユーティリティモジュール。
定数（対応拡張子）や、画像比較に使うハッシュ・ヒストグラム・ピクセル差分などの関数を集約する。
"""

import sys

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def convert_palette_to_rgba_if_needed(img: Image.Image) -> Image.Image:
    """
    Pモードで透過情報がある場合、RGBAに変換する（Pillowの警告回避）。
    """
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA")
    return img


def calculate_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """
    ImageオブジェクトからdHash（Difference Hash）値を計算する。
    """
    try:
        img = convert_palette_to_rgba_if_needed(img)
        img_resized = img.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.BILINEAR
        )
        pixels = list(img_resized.tobytes())

        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                difference.append(pixel_left > pixel_right)

        decimal_value = 0
        for bit in difference:
            decimal_value = (decimal_value << 1) | bit
        return decimal_value
    except Exception as e:
        print(f"dHash計算中にエラーが発生しました: {e}", file=sys.stderr)
        return None


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    2つのハッシュ値のハミング距離（異なるビット数）を計算する。
    """
    return bin(hash1 ^ hash2).count("1")


def compare_color_histograms(img1: Image.Image, img2: Image.Image) -> float:
    """
    Pillowのhistogram()を用いて、2つの画像の正規化ヒストグラム交差（Histogram Intersection）を計算する。
    値は 0.0（全く異なる）から 1.0（完全に一致）の間になる。
    """
    try:
        img1 = convert_palette_to_rgba_if_needed(img1)
        img2 = convert_palette_to_rgba_if_needed(img2)
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        hist1 = img1.histogram()
        hist2 = img2.histogram()

        # 各チャンネル（R, G, B）のピクセル数で割って正規化
        total1 = img1.width * img1.height
        total2 = img2.width * img2.height

        # ピクセル数で正規化したヒストグラムの交差を計算
        intersection = 0.0
        for h1, h2 in zip(hist1, hist2):
            intersection += min(h1 / total1, h2 / total2)

        # 3チャンネル合計なので、3.0で割って 0.0 - 1.0 の範囲にする
        return intersection / 3.0
    except Exception as e:
        print(f"ヒストグラム比較中にエラーが発生しました: {e}", file=sys.stderr)
        return 0.0


def compare_pixel_difference(
    img1: Image.Image, img2: Image.Image, size: int = 128
) -> float:
    """
    2つの画像を中解像度のグレースケールに縮小し、ピクセル間の平均絶対誤差（MAE）を計算する。
    値は 0.0 から 255.0 の間になり、0.0 に近いほどピクセルレベルで同一。
    """
    try:
        img1 = convert_palette_to_rgba_if_needed(img1)
        img2 = convert_palette_to_rgba_if_needed(img2)
        img1_gray = img1.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        img2_gray = img2.convert("L").resize((size, size), Image.Resampling.BILINEAR)

        bytes1 = img1_gray.tobytes()
        bytes2 = img2_gray.tobytes()

        total_diff = 0
        for b1, b2 in zip(bytes1, bytes2):
            total_diff += abs(b1 - b2)

        mae = total_diff / (size * size)
        return mae
    except Exception as e:
        print(f"ピクセル差分比較中にエラーが発生しました: {e}", file=sys.stderr)
        return 255.0
