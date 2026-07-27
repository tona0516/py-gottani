"""
evaluate_images.py
====================
StableDiffusion等で大量生成した画像を自動でスクリーニングするスクリプト。

検出する項目:
  1. ブレ / ぼやけ (Laplacian分散が低い画像)
  2. 顔の破綻疑い (顔の左右非対称が大きい)
  3. 手の破綻疑い (手のランドマーク検出の信頼度が低い / 指先が不自然に密集)
  4. 美的スコア (CLIP + LAION Aesthetic Predictorによる「見栄えの良さ」の推定、
     おおよそ1〜10のスケールで、高いほど好まれやすい構図・色合いとされる)

「破綻がある」と断定するわけではなく、あくまで "目視チェックが必要な候補" を
スコアリングして絞り込むためのツールです。美的スコアも統計的な傾向にすぎず、
好みを断定するものではありません。誤検出・見逃しは必ず発生するので、
最終判断は目視で行ってください。

初回実行時、検出用モデルを自動でダウンロードします
(~/.cache/evaluate_images/ に保存され、2回目以降は再利用されます)。
美的スコアリングはCLIP ViT-L/14 (~900MB) をダウンロードするため、
初回のみ時間がかかります。GPUがあれば自動で使用し高速化されます
(無くてもCPUで動作しますが1枚あたり数秒かかることがあります)。

使い方 (uv):
    uv run evaluate_images.py --input ./images --output result.csv

    # スコアが悪い画像だけ別フォルダにコピーして見やすくする場合
    uv run evaluate_images.py --input ./images --output result.csv --copy-flagged ./review

    # 実行権限を付けていれば ./evaluate_images.py --input ... でも可
    # (先頭のシバン行が `uv run --script` を呼び出すため)

依存関係はこのファイル先頭の `# /// script` ブロックに記載済みなので、
requirements.txtやvenvの準備は不要です。uvが初回実行時に自動で用意します。

オプション:
    --input                画像フォルダ (再帰的に探索)
    --output               結果CSVの出力先 (デフォルト: result.csv)
    --copy-flagged         問題のない画像のうち上位20%をコピーするフォルダ (任意)
    --blur-threshold       ブレ判定のしきい値 (デフォルト: 100.0、低いほど厳しい)
    --aesthetic-threshold  このスコア未満を「問題あり」に含める (任意、指定しなければ採点のみ)
    --no-face              顔チェックを無効化
    --no-hand              手チェックを無効化
    --no-aesthetic         美的スコアリングを無効化 (torch/CLIPのダウンロードが不要になる)
"""

import argparse
import csv
import math
import shutil
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import open_clip
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from torch import nn

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

MODEL_URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
# LAION Aesthetic Predictor (CLIP ViT-L/14 embedding -> MLP -> 1〜10程度のスコア)
# 出典: https://github.com/christophschuhmann/improved-aesthetic-predictor
AESTHETIC_MLP_URL = (
    "https://raw.githubusercontent.com/christophschuhmann/"
    "improved-aesthetic-predictor/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
CACHE_DIR = Path.home() / ".cache" / "evaluate_images"


def download_if_missing(url: str, dest: Path) -> str:
    """ファイルが無ければダウンロードしてパスを返す。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  ダウンロード中: {dest.name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  ダウンロード完了: {dest}")
    return str(dest)


def ensure_model(name: str) -> str:
    """MediaPipe検出モデルが無ければダウンロードしてパスを返す。"""
    return download_if_missing(MODEL_URLS[name], CACHE_DIR / f"{name}_landmarker.task")


# ---------------------------------------------------------------------------
# 1. ブレ / ぼやけ検出
# ---------------------------------------------------------------------------
def compute_blur_score(image_bgr: np.ndarray) -> float:
    """Laplacianの分散を計算。値が低いほどエッジが少なく、ぼやけている可能性が高い。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ---------------------------------------------------------------------------
# 2. 顔の破綻疑い検出 (MediaPipe FaceLandmarker)
# ---------------------------------------------------------------------------
class FaceChecker:
    def __init__(self):
        model_path = ensure_model("face")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_faces=5,
            min_face_detection_confidence=0.4,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def check(self, image_rgb: np.ndarray):
        """
        戻り値: (顔検出数, 最大の左右非対称スコア, 問題フラグ理由リスト)
        非対称スコアは目・口の左右対応点の位置ずれを正規化した値。
        大きいほど顔が歪んでいる可能性が高い。
        """
        h, w, _ = image_rgb.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        issues = []
        max_asym = 0.0

        if not result.face_landmarks:
            return 0, 0.0, issues

        for landmarks in result.face_landmarks:
            pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks])

            # 左右対称性チェック: 左右の目尻/口角/眉のペア
            pairs = [(33, 263), (61, 291), (105, 334)]
            face_width = np.linalg.norm(pts[454] - pts[234]) + 1e-6  # 顔の横幅で正規化
            mid_x = (pts[454][0] + pts[234][0]) / 2
            asym_scores = []
            for left_idx, right_idx in pairs:
                dl = abs(pts[left_idx][0] - mid_x)
                dr = abs(pts[right_idx][0] - mid_x)
                asym_scores.append(abs(dl - dr) / face_width)
            asym = float(np.mean(asym_scores))
            max_asym = max(max_asym, asym)

        if max_asym > 0.18:
            issues.append(f"顔の左右非対称が大きい可能性(score={max_asym:.2f})")

        return len(result.face_landmarks), max_asym, issues


# ---------------------------------------------------------------------------
# 3. 手の破綻疑い検出 (MediaPipe HandLandmarker)
# ---------------------------------------------------------------------------
class HandChecker:
    def __init__(self):
        model_path = ensure_model("hand")
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=4,
            min_hand_detection_confidence=0.3,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def check(self, image_rgb: np.ndarray):
        """
        戻り値: (手検出数, 最低スコア, 問題フラグ理由リスト)
        MediaPipeは指の本数や骨格が破綻した手だと検出自体に失敗したり
        信頼度が下がりやすいため、それを間接的な破綻シグナルとして使う。
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        issues = []
        min_score = 1.0

        if not result.handedness:
            return 0, 1.0, issues

        for handedness, landmarks in zip(result.handedness, result.hand_landmarks):
            score = handedness[0].score
            min_score = min(min_score, score)

            # 指先(4,8,12,16,20)同士の距離が極端に近い/ゼロに潰れていないか
            tips = np.array(
                [[landmarks[i].x, landmarks[i].y] for i in (4, 8, 12, 16, 20)]
            )
            dists = [
                np.linalg.norm(tips[i] - tips[j])
                for i in range(5)
                for j in range(i + 1, 5)
            ]
            if min(dists) < 0.01:
                issues.append("指先の位置が不自然に密集(指の癒着・欠損の疑い)")

        if min_score < 0.55:
            issues.append(f"手の検出信頼度が低い(score={min_score:.2f}) 破綻の疑い")

        return len(result.handedness), min_score, issues


# ---------------------------------------------------------------------------
# 4. 美的スコアリング (CLIP ViT-L/14 + LAION Aesthetic Predictor)
# ---------------------------------------------------------------------------
class AestheticMLP(nn.Module):
    """LAION Aesthetic Predictor (sac+logos+ava1-l14-linearMSE) と同一構造のMLP。"""

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticChecker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"  美的スコアリング用にCLIPモデルを準備しています (device={self.device})..."
        )
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.clip_model.eval().to(self.device)

        mlp_path = download_if_missing(
            AESTHETIC_MLP_URL, CACHE_DIR / "aesthetic_mlp.pth"
        )
        self.mlp = AestheticMLP(768)
        state = torch.load(mlp_path, map_location=self.device)
        self.mlp.load_state_dict(state)
        self.mlp.eval().to(self.device)

    def score(self, image_rgb: np.ndarray) -> float:
        """L2正規化したCLIP埋め込みからおおよそ1〜10の美的スコアを推定する。"""
        pil_image = Image.fromarray(image_rgb)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            pred = self.mlp(emb.float())
        return float(pred.item())


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def find_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description="AI生成画像の自動品質チェック")
    parser.add_argument("-i", "--input", required=True, help="画像フォルダ")
    parser.add_argument("-o", "--output", default="result.csv", help="結果CSVの出力先")
    parser.add_argument(
        "--copy-dir",
        type=Path,
        default=None,
        help="問題のない画像のうち上位20%%をコピーするフォルダ",
    )
    parser.add_argument(
        "--blur-threshold", type=float, default=100.0, help="ブレ判定のしきい値"
    )
    parser.add_argument(
        "--aesthetic-threshold",
        type=float,
        default=None,
        help="このスコア未満を「問題あり」に含める(任意、指定しなければ採点のみ)",
    )
    parser.add_argument("--no-face", action="store_true", help="顔チェックを無効化")
    parser.add_argument("--no-hand", action="store_true", help="手チェックを無効化")
    parser.add_argument(
        "--no-aesthetic", action="store_true", help="美的スコアリングを無効化"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[エラー] フォルダが見つかりません: {input_dir}")
        sys.exit(1)

    images = find_images(input_dir)
    if not images:
        print("[エラー] 画像が見つかりませんでした。")
        sys.exit(1)

    face_checker = None
    if not args.no_face:
        try:
            face_checker = FaceChecker()
        except Exception as e:
            print(
                f"[警告] 顔検出モデルの準備に失敗したため、顔チェックをスキップします: {e}"
            )

    hand_checker = None
    if not args.no_hand:
        try:
            hand_checker = HandChecker()
        except Exception as e:
            print(
                f"[警告] 手検出モデルの準備に失敗したため、手チェックをスキップします: {e}"
            )

    aesthetic_checker = None
    if not args.no_aesthetic:
        try:
            aesthetic_checker = AestheticChecker()
        except Exception as e:
            print(
                f"[警告] 美的スコアリングモデルの準備に失敗したため、スキップします: {e}"
            )

    rows = []
    print(f"{len(images)} 枚の画像を処理します...")

    for i, path in enumerate(images, 1):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"  [スキップ] 読み込み失敗: {path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        reasons = []

        blur_score = compute_blur_score(img_bgr)
        if blur_score < args.blur_threshold:
            reasons.append(f"ブレ/ぼやけの疑い(score={blur_score:.1f})")

        n_faces, asym, face_issues = (0, 0.0, [])
        if face_checker is not None:
            n_faces, asym, face_issues = face_checker.check(img_rgb)
            reasons.extend(face_issues)

        n_hands, hand_score, hand_issues = (0, 1.0, [])
        if hand_checker is not None:
            n_hands, hand_score, hand_issues = hand_checker.check(img_rgb)
            reasons.extend(hand_issues)

        aesthetic_score = None
        if aesthetic_checker is not None:
            aesthetic_score = aesthetic_checker.score(img_rgb)
            if (
                args.aesthetic_threshold is not None
                and aesthetic_score < args.aesthetic_threshold
            ):
                reasons.append(f"美的スコアが低め(score={aesthetic_score:.2f})")

        rows.append(
            {
                "filename": str(path.relative_to(input_dir)),
                "blur_score": round(blur_score, 1),
                "n_faces": n_faces,
                "face_asymmetry": round(asym, 3),
                "n_hands": n_hands,
                "hand_min_confidence": round(hand_score, 3),
                "aesthetic_score": round(aesthetic_score, 3)
                if aesthetic_score is not None
                else "",
                "n_issues": len(reasons),
                "issues": "; ".join(reasons) if reasons else "",
            }
        )

        if i % 20 == 0 or i == len(images):
            print(f"  {i}/{len(images)} 処理完了")

    if not rows:
        print("[エラー] 処理できた画像がありませんでした。")
        sys.exit(1)

    # n_issuesが多い順(問題が多そうな画像を上に)、同数なら美的スコアが低い順でソート
    rows.sort(
        key=lambda r: (
            -r["n_issues"],
            r["aesthetic_score"] if r["aesthetic_score"] != "" else 0,
        )
    )

    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n結果を書き出しました: {out_path.resolve()}")

    flagged = [r for r in rows if r["n_issues"] > 0]
    print(f"問題の可能性がある画像: {len(flagged)} / {len(rows)} 枚")

    if args.copy_dir:
        clean_rows = [r for r in rows if r["n_issues"] == 0]
        if clean_rows:
            clean_rows.sort(
                key=lambda r: (
                    r["aesthetic_score"]
                    if r["aesthetic_score"] != ""
                    else -float("inf"),
                    r["blur_score"],
                ),
                reverse=True,
            )
            top_count = math.ceil(len(clean_rows) * 0.2)
            top_rows = clean_rows[:top_count]

            dest = Path(args.copy_dir)
            dest.mkdir(parents=True, exist_ok=True)
            for r in top_rows:
                src = input_dir / r["filename"]
                dst = dest / Path(r["filename"]).name
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"  コピー失敗: {src} ({e})")
            print(
                f"問題のない画像 ({len(clean_rows)}枚) のうち上位20% ({len(top_rows)}枚) を {dest.resolve()} にコピーしました。"
            )
        else:
            print("問題のない画像が存在しなかったため、コピーをスキップしました。")


if __name__ == "__main__":
    main()
