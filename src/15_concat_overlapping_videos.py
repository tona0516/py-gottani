"""
2つの動画の重複部分（オーバーラップ）を自動検出し、自然に連結して1つの動画を出力するスクリプト。

主な機能:
1. 重複検出（Coarse-to-Fine 粗密フレームマッチング）:
   - 動画1の末尾と動画2の先頭をスキャンし、最も一致する重複区間・接続フレームを特定します。
   - リサイズ画像による高速相関探索（Coarse）と、フレーム単位の高精度SSIM/MSE差分探索（Fine）を組み合わせ。
2. 自然なトランジション処理:
   - "crossfade"（デフォルト）: 重複区間（または指定秒数）で滑らかにアルファブレンド（ディゾルブ）して連結。
   - "cut": 重複を完全に解消し、最良の一致点で瞬時に切り替え。
3. 音声および映像の出力:
   - OpenCVによるダイレクトレンダリング（単体で完結動作可能）。
   - ffmpegが利用可能な場合は、音声トラックも含めた高品質エンコードおよび音声クロスフェード結合に対応。
4. プレビュー＆ログ:
   - 一致した接続点フレームの並列比較画像（--save-preview）を保存可能。
"""

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class VideoMetadata:
    """動画の基本メタデータ。"""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # 秒


@dataclass
class OverlapResult:
    """重複検出の結果。"""

    frame_idx1: int  # 動画1での接続基準フレーム
    frame_idx2: int  # 動画2での接続基準フレーム
    time1: float  # 動画1での接続秒数
    time2: float  # 動画2での接続秒数
    similarity: float  # 類似度スコア (0.0 〜 1.0)
    overlap_duration: float  # 検出された重複秒数


def get_video_metadata(video_path: str) -> VideoMetadata:
    """
    動画ファイルからメタデータを取得する。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けませんでした: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # フォールバック
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0

    cap.release()
    return VideoMetadata(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
    )


def format_timestamp(seconds: float) -> str:
    """
    秒数を hh:mm:ss.ms 形式にフォーマットする。
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def extract_downsampled_frames(
    video_path: str,
    start_frame: int,
    end_frame: int,
    step: int = 1,
    target_size: tuple[int, int] = (64, 64),
) -> tuple[list[int], np.ndarray]:
    """
    指定区間から一定間隔でフレームを読み込み、縮小グレースケール配列として取得する。

    Args:
        video_path: 動画ファイルパス
        start_frame: 開始フレーム番号
        end_frame: 終了フレーム番号
        step: フレームの間引き間隔
        target_size: 縮小サイズ (幅, 高さ)

    Returns:
        (フレーム番号リスト, 形状 (N, H, W) の正規化 float32 配列)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画を開けませんでした: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

    frame_indices = []
    frames = []
    current_idx = max(0, start_frame)

    while current_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_idx - start_frame) % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
            # 正規化 (平均0, 標準偏差1)
            norm = small.astype(np.float32)
            std = np.std(norm)
            if std > 1e-5:
                norm = (norm - np.mean(norm)) / std
            else:
                norm = norm - np.mean(norm)
            frames.append(norm)
            frame_indices.append(current_idx)

        current_idx += 1

    cap.release()
    if not frames:
        return [], np.empty((0, target_size[1], target_size[0]), dtype=np.float32)

    return frame_indices, np.array(frames, dtype=np.float32)


def compute_frame_similarity(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    2つのフレームの類似度（0.0 〜 1.0）を計算する。
    RGBカラーとグレースケール差分を総合して評価。
    """
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(
            frame_b, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_AREA
        )

    # 輝度差分のMSE
    diff = cv2.absdiff(frame_a, frame_b)
    mse = np.mean(diff**2)
    # MSEから類似度（指数減衰）
    sim_mse = np.exp(-mse / 500.0)

    # ヒストグラム相関
    hsv_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [30, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    sim_hist = max(0.0, float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)))

    return float(0.7 * sim_mse + 0.3 * sim_hist)


def detect_overlap(
    meta1: VideoMetadata,
    meta2: VideoMetadata,
    search_window1_sec: float = 60.0,
    search_window2_sec: float = 60.0,
    coarse_step_sec: float = 0.5,
) -> OverlapResult:
    """
    動画1の末尾と動画2の先頭から重複区間を検出する。

    Args:
        meta1: 動画1のメタデータ
        meta2: 動画2のメタデータ
        search_window1_sec: 動画1末尾の探索秒数
        search_window2_sec: 動画2先頭の探索秒数
        coarse_step_sec: 粗探索時のフレーム間隔秒数

    Returns:
        OverlapResult オブジェクト
    """
    # 探索フレーム範囲の決定
    search_frames1 = int(search_window1_sec * meta1.fps)
    start_frame1 = max(0, meta1.frame_count - search_frames1)
    end_frame1 = meta1.frame_count

    search_frames2 = int(search_window2_sec * meta2.fps)
    start_frame2 = 0
    end_frame2 = min(meta2.frame_count, search_frames2)

    step1 = max(1, int(round(coarse_step_sec * meta1.fps)))
    step2 = max(1, int(round(coarse_step_sec * meta2.fps)))

    print(
        f"粗探索（Coarse search）を実行中...\n"
        f"  動画1: 末尾 {search_window1_sec:.1f}s (フレーム {start_frame1} ~ {end_frame1}, step={step1})\n"
        f"  動画2: 先頭 {search_window2_sec:.1f}s (フレーム {start_frame2} ~ {end_frame2}, step={step2})"
    )

    indices1, frames1 = extract_downsampled_frames(
        meta1.path, start_frame1, end_frame1, step=step1
    )
    indices2, frames2 = extract_downsampled_frames(
        meta2.path, start_frame2, end_frame2, step=step2
    )

    if len(frames1) == 0 or len(frames2) == 0:
        raise RuntimeError("探索範囲からフレームを取得できませんでした。")

    # 全ペアの類似度行列を計算 (内積による正規化相互相関)
    n1 = len(frames1)
    n2 = len(frames2)
    dim = frames1.shape[1] * frames1.shape[2]
    flat1 = frames1.reshape(n1, dim)
    flat2 = frames2.reshape(n2, dim)

    # 相関スコア行列: S[i, j] = flat1[i] . flat2[j] / dim
    corr_matrix = np.dot(flat1, flat2.T) / float(dim)

    # 最もスコアの高い候補ペアをトップK件取得
    top_candidates = []
    flat_indices = np.argsort(corr_matrix.ravel())[::-1]
    for idx in flat_indices[:20]:
        i, j = divmod(idx, n2)
        score = corr_matrix[i, j]
        f1 = indices1[i]
        f2 = indices2[j]
        top_candidates.append((score, f1, f2))

    best_coarse_score, best_coarse_f1, best_coarse_f2 = top_candidates[0]
    print(
        f"粗探索ベストマッチ: 動画1 Frame={best_coarse_f1} ({format_timestamp(best_coarse_f1 / meta1.fps)}), "
        f"動画2 Frame={best_coarse_f2} ({format_timestamp(best_coarse_f2 / meta2.fps)}), "
        f"相関スコア={best_coarse_score:.3f}"
    )

    # 密探索（Fine search）: ベスト候補の周辺±2秒を全フレーム精査
    fine_range_sec = 2.0
    fine_half_frames1 = int(fine_range_sec * meta1.fps)
    fine_half_frames2 = int(fine_range_sec * meta2.fps)

    fine_start1 = max(start_frame1, best_coarse_f1 - fine_half_frames1)
    fine_end1 = min(end_frame1, best_coarse_f1 + fine_half_frames1)

    fine_start2 = max(start_frame2, best_coarse_f2 - fine_half_frames2)
    fine_end2 = min(end_frame2, best_coarse_f2 + fine_half_frames2)

    print(
        f"密探索（Fine search）を実行中...\n"
        f"  動画1範囲: {fine_start1} ~ {fine_end1}\n"
        f"  動画2範囲: {fine_start2} ~ {fine_end2}"
    )

    # 動画1のフレームをキャッシュ
    cap1 = cv2.VideoCapture(meta1.path)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, fine_start1)
    fine_frames1 = {}
    for f in range(fine_start1, fine_end1):
        ret, frame = cap1.read()
        if not ret:
            break
        fine_frames1[f] = frame
    cap1.release()

    # 動画2のフレームをキャッシュ
    cap2 = cv2.VideoCapture(meta2.path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, fine_start2)
    fine_frames2 = {}
    for f in range(fine_start2, fine_end2):
        ret, frame = cap2.read()
        if not ret:
            break
        fine_frames2[f] = frame
    cap2.release()

    # フレームペアの類似度を精査
    best_sim = -1.0
    best_f1 = best_coarse_f1
    best_f2 = best_coarse_f2

    # 時間差 (offset) を固定した連続フレームマッチングを評価
    best_offset_sec = (best_coarse_f1 / meta1.fps) - (best_coarse_f2 / meta2.fps)

    for f1, frame_a in fine_frames1.items():
        t1 = f1 / meta1.fps
        target_t2 = t1 - best_offset_sec
        target_f2 = int(round(target_t2 * meta2.fps))

        for f2 in range(target_f2 - 5, target_f2 + 6):
            if f2 in fine_frames2:
                frame_b = fine_frames2[f2]
                sim = compute_frame_similarity(frame_a, frame_b)
                if sim > best_sim:
                    best_sim = sim
                    best_f1 = f1
                    best_f2 = f2

    time1 = best_f1 / meta1.fps
    time2 = best_f2 / meta2.fps
    overlap_dur = (meta1.duration - time1) + time2

    return OverlapResult(
        frame_idx1=best_f1,
        frame_idx2=best_f2,
        time1=time1,
        time2=time2,
        similarity=best_sim,
        overlap_duration=overlap_dur,
    )


def save_preview_image(
    video1_path: str,
    video2_path: str,
    frame_idx1: int,
    frame_idx2: int,
    output_path: str,
    similarity: float,
) -> None:
    """
    接続点の一致フレームを横並びで可視化したプレビュー画像を保存する。
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx1)
    ret1, f1 = cap1.read()
    cap1.release()

    cap2 = cv2.VideoCapture(video2_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx2)
    ret2, f2 = cap2.read()
    cap2.release()

    if not ret1 or not ret2:
        print("警告: プレビュー用フレームの抽出に失敗しました。", file=sys.stderr)
        return

    # 高さを統一
    target_h = 480
    w1 = int(f1.shape[1] * (target_h / f1.shape[0]))
    w2 = int(f2.shape[1] * (target_h / f2.shape[0]))
    f1_resized = cv2.resize(f1, (w1, target_h), interpolation=cv2.INTER_AREA)
    f2_resized = cv2.resize(f2, (w2, target_h), interpolation=cv2.INTER_AREA)

    # アノテーション描画
    cv2.putText(
        f1_resized,
        f"Video 1 (Frame {frame_idx1})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        f2_resized,
        f"Video 2 (Frame {frame_idx2})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    combined = np.hstack([f1_resized, f2_resized])

    # 差分ヒートマップも下部に追加
    diff = cv2.absdiff(cv2.resize(f1, (w1, target_h)), cv2.resize(f2, (w1, target_h)))
    diff_colored = cv2.applyColorMap(
        cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET
    )
    cv2.putText(
        diff_colored,
        f"Absolute Difference (Similarity: {similarity * 100:.1f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 全体レイアウト
    bottom_bar = np.zeros((target_h, combined.shape[1], 3), dtype=np.uint8)
    diff_w = int(diff_colored.shape[1] * (target_h / diff_colored.shape[0]))
    diff_resized = cv2.resize(
        diff_colored, (diff_w, target_h), interpolation=cv2.INTER_AREA
    )
    offset_x = (combined.shape[1] - diff_w) // 2
    bottom_bar[:, offset_x : offset_x + diff_w] = diff_resized

    preview_img = np.vstack([combined, bottom_bar])

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, preview_img)
    print(f"プレビュー画像を保存しました: {output_path}")


def has_ffmpeg() -> bool:
    """システムにffmpegコマンドが存在するか確認する。"""
    return shutil.which("ffmpeg") is not None


def has_audio_stream(video_path: str) -> bool:
    """
    動画ファイルに音声ストリームが存在するか確認する。
    """
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = proc.stderr.decode("utf-8", errors="replace")
        return "Audio:" in output
    except Exception:
        return False


def concat_videos_ffmpeg_video_only(
    video1: str,
    video2: str,
    overlap: OverlapResult,
    output_path: str,
    transition: str = "crossfade",
    transition_sec: float = 1.0,
) -> bool:
    """
    ffmpegを使用して映像のみを連結する（音声ストリームがない場合など）。
    """
    t1_cut = overlap.time1
    t2_start = overlap.time2

    if transition == "cut" or transition_sec <= 0:
        filter_complex = (
            f"[0:v]trim=end={t1_cut:.4f},setpts=PTS-STARTPTS[v0];"
            f"[1:v]trim=start={t2_start:.4f},setpts=PTS-STARTPTS[v1];"
            f"[v0][v1]concat=n=2:v=1:a=0[outv]"
        )
    else:
        half_t = transition_sec / 2.0
        v1_end = t1_cut + half_t
        v2_start = max(0.0, t2_start - half_t)
        actual_xfade_duration = transition_sec
        offset = t1_cut - half_t

        filter_complex = (
            f"[0:v]trim=end={v1_end:.4f},setpts=PTS-STARTPTS[v0];"
            f"[1:v]trim=start={v2_start:.4f},setpts=PTS-STARTPTS[v1];"
            f"[v0][v1]xfade=transition=fade:duration={actual_xfade_duration:.4f}:offset={offset:.4f}[outv]"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video1,
        "-i",
        video2,
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        output_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace")
            print(f"ffmpeg (映像のみ) の実行に失敗しました: {err_msg}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(
            f"ffmpeg (映像のみ) の呼び出し中にエラーが発生しました: {e}",
            file=sys.stderr,
        )
        return False


def concat_videos_ffmpeg(
    video1: str,
    video2: str,
    overlap: OverlapResult,
    output_path: str,
    transition: str = "crossfade",
    transition_sec: float = 1.0,
) -> bool:
    """
    ffmpegを使用して高品質で動画（および音声）を自然に連結する。
    """
    if not has_ffmpeg():
        return False

    has_audio1 = has_audio_stream(video1)
    has_audio2 = has_audio_stream(video2)

    if not (has_audio1 and has_audio2):
        print("ffmpegを使用して映像を連結中（音声ストリームなし）...")
        return concat_videos_ffmpeg_video_only(
            video1, video2, overlap, output_path, transition, transition_sec
        )

    print("ffmpegを使用して動画・音声を連結中...")

    t1_cut = overlap.time1
    t2_start = overlap.time2

    if transition == "cut" or transition_sec <= 0:
        filter_complex = (
            f"[0:v]trim=end={t1_cut:.4f},setpts=PTS-STARTPTS[v0];"
            f"[0:a]atrim=end={t1_cut:.4f},asetpts=PTS-STARTPTS[a0];"
            f"[1:v]trim=start={t2_start:.4f},setpts=PTS-STARTPTS[v1];"
            f"[1:a]atrim=start={t2_start:.4f},asetpts=PTS-STARTPTS[a1];"
            f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"
        )
    else:
        half_t = transition_sec / 2.0
        v1_end = t1_cut + half_t
        v2_start = max(0.0, t2_start - half_t)
        actual_xfade_duration = transition_sec
        offset = t1_cut - half_t

        filter_complex = (
            f"[0:v]trim=end={v1_end:.4f},setpts=PTS-STARTPTS[v0];"
            f"[0:a]atrim=end={v1_end:.4f},asetpts=PTS-STARTPTS[a0];"
            f"[1:v]trim=start={v2_start:.4f},setpts=PTS-STARTPTS[v1];"
            f"[1:a]atrim=start={v2_start:.4f},asetpts=PTS-STARTPTS[a1];"
            f"[v0][v1]xfade=transition=fade:duration={actual_xfade_duration:.4f}:offset={offset:.4f}[outv];"
            f"[a0][a1]acrossfade=d={actual_xfade_duration:.4f}:c1=tri:c2=tri[outa]"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video1,
        "-i",
        video2,
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        output_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace")
            print(
                f"ffmpegの実行に失敗しました:\n{err_msg}",
                file=sys.stderr,
            )
            # フォールバック
            return concat_videos_ffmpeg_video_only(
                video1, video2, overlap, output_path, transition, transition_sec
            )
        return True
    except Exception as e:
        print(f"ffmpeg呼び出し中にエラーが発生しました: {e}", file=sys.stderr)
        return concat_videos_ffmpeg_video_only(
            video1, video2, overlap, output_path, transition, transition_sec
        )


def concat_videos_opencv(
    meta1: VideoMetadata,
    meta2: VideoMetadata,
    overlap: OverlapResult,
    output_path: str,
    transition: str = "crossfade",
    transition_sec: float = 1.0,
) -> None:
    """
    OpenCVを使用してフレーム単位で精密に動画をレンダリング・連結する。
    """
    print("OpenCVを使用して動画フレームを連結中...")

    out_w = meta1.width
    out_h = meta1.height
    fps = meta1.fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"出力動画ファイルを作成できませんでした: {output_path}")

    cap1 = cv2.VideoCapture(meta1.path)
    cap2 = cv2.VideoCapture(meta2.path)

    try:
        if transition == "cut" or transition_sec <= 0:
            total_frames = overlap.frame_idx1 + (meta2.frame_count - overlap.frame_idx2)
            pbar = tqdm(total=total_frames, desc="レンダリング進捗")

            f1_count = 0
            while f1_count < overlap.frame_idx1:
                ret, frame = cap1.read()
                if not ret:
                    break
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_AREA
                    )
                writer.write(frame)
                f1_count += 1
                pbar.update(1)

            cap2.set(cv2.CAP_PROP_POS_FRAMES, overlap.frame_idx2)
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_AREA
                    )
                writer.write(frame)
                pbar.update(1)

            pbar.close()

        else:
            fade_frames = max(1, int(round(transition_sec * fps)))
            half_fade1 = fade_frames // 2

            v1_end_frame = max(0, overlap.frame_idx1 - half_fade1)
            v2_start_frame = max(0, overlap.frame_idx2 - half_fade1)

            total_frames = (
                v1_end_frame
                + fade_frames
                + (meta2.frame_count - (v2_start_frame + fade_frames))
            )
            pbar = tqdm(total=total_frames, desc="レンダリング進捗（クロスフェード）")

            # フェーズ1: 動画1の単独区間
            f1_count = 0
            while f1_count < v1_end_frame:
                ret, frame = cap1.read()
                if not ret:
                    break
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_AREA
                    )
                writer.write(frame)
                f1_count += 1
                pbar.update(1)

            # フェーズ2: クロスフェード区間
            cap2.set(cv2.CAP_PROP_POS_FRAMES, v2_start_frame)
            for i in range(fade_frames):
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()

                if not ret1 and not ret2:
                    break
                elif not ret1:
                    frame = frame2
                elif not ret2:
                    frame = frame1
                else:
                    if frame1.shape[1] != out_w or frame1.shape[0] != out_h:
                        frame1 = cv2.resize(
                            frame1, (out_w, out_h), interpolation=cv2.INTER_AREA
                        )
                    if frame2.shape[1] != out_w or frame2.shape[0] != out_h:
                        frame2 = cv2.resize(
                            frame2, (out_w, out_h), interpolation=cv2.INTER_AREA
                        )

                    alpha = (i + 1) / float(fade_frames + 1)
                    frame = cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0.0)

                writer.write(frame)
                pbar.update(1)

            # フェーズ3: 動画2の残り区間
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_AREA
                    )
                writer.write(frame)
                pbar.update(1)

            pbar.close()

    finally:
        cap1.release()
        cap2.release()
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2つの動画の重複部分を自動検出し、自然に連結して1つの動画を作成します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使い方（デフォルト: クロスフェード結合）
  python 15_concat_overlapping_videos.py -v1 part1.mp4 -v2 part2.mp4 -o merged.mp4

  # カット結合（重複部分を完全に切り詰めてシームレスにジャンプ）
  python 15_concat_overlapping_videos.py -v1 part1.mp4 -v2 part2.mp4 -o merged.mp4 --transition cut

  # 探索範囲やクロスフェード時間をカスタマイズ
  python 15_concat_overlapping_videos.py -v1 part1.mp4 -v2 part2.mp4 -o merged.mp4 --search-window 120 --fade-duration 1.5

  # 一致箇所のプレビュー確認画像を保存
  python 15_concat_overlapping_videos.py -v1 part1.mp4 -v2 part2.mp4 -o merged.mp4 --save-preview preview.jpg
        """,
    )
    parser.add_argument(
        "-v1",
        "--video1",
        required=True,
        help="前半の動画ファイルパス",
    )
    parser.add_argument(
        "-v2",
        "--video2",
        required=True,
        help="後半の動画ファイルパス",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="merged_output.mp4",
        help="出力動画ファイルパス (デフォルト: merged_output.mp4)",
    )
    parser.add_argument(
        "-t",
        "--transition",
        choices=["crossfade", "cut"],
        default="cut",
        help="接続部のトランジション方式 (デフォルト: cut)",
    )
    parser.add_argument(
        "--fade-duration",
        type=float,
        default=1.0,
        help="クロスフェードのブレンド時間（秒） (デフォルト: 1.0)",
    )
    parser.add_argument(
        "--search-window",
        type=float,
        default=60.0,
        help="重複探索を行う秒数（動画1の末尾および動画2の先頭） (デフォルト: 60.0)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.9,
        help="重複検出の最小類似度スコア (0.0 〜 1.0)。この値を下回る場合はエラーで停止します (デフォルト: 0.9)",
    )
    parser.add_argument(
        "--save-preview",
        default=None,
        help="接続点の一致プレビュー画像（左右比較＋差分）の保存先パス",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="レンダリングエンジン (auto: ffmpegがあればffmpeg、無ければopencv) (デフォルト: auto)",
    )

    args = parser.parse_args()

    v1_path = os.path.abspath(args.video1)
    v2_path = os.path.abspath(args.video2)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(v1_path):
        print(f"エラー: 動画1が見つかりません: {v1_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(v2_path):
        print(f"エラー: 動画2が見つかりません: {v2_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("動画メタデータの読み込み中...")
    meta1 = get_video_metadata(v1_path)
    meta2 = get_video_metadata(v2_path)

    print(
        f"動画1: {os.path.basename(v1_path)} "
        f"({meta1.width}x{meta1.height}, {meta1.fps:.2f}fps, 長さ: {format_timestamp(meta1.duration)})"
    )
    print(
        f"動画2: {os.path.basename(v2_path)} "
        f"({meta2.width}x{meta2.height}, {meta2.fps:.2f}fps, 長さ: {format_timestamp(meta2.duration)})"
    )
    print("=" * 60)

    # 重複検出
    overlap = detect_overlap(
        meta1,
        meta2,
        search_window1_sec=args.search_window,
        search_window2_sec=args.search_window,
    )

    print("\n" + "=" * 60)
    print("【重複検出結果】")
    print(
        f"  動画1 接続点: {format_timestamp(overlap.time1)} (Frame {overlap.frame_idx1})"
    )
    print(
        f"  動画2 接続点: {format_timestamp(overlap.time2)} (Frame {overlap.frame_idx2})"
    )
    print(f"  推定重複区間: {overlap.overlap_duration:.2f} 秒")
    print(f"  一致度類似度: {overlap.similarity * 100:.1f} %")
    print("=" * 60 + "\n")

    # プレビュー画像保存
    if args.save_preview:
        save_preview_image(
            v1_path,
            v2_path,
            overlap.frame_idx1,
            overlap.frame_idx2,
            os.path.abspath(args.save_preview),
            overlap.similarity,
        )

    # 類似度チェック
    if overlap.similarity < args.min_similarity:
        print(
            f"エラー: 検出された類似度（{overlap.similarity * 100:.1f}%）が基準値（{args.min_similarity * 100:.1f}%）未満のため、処理を停止しました。\n"
            "動画同士の重複が見つからなかったか、一致度が不十分です。\n"
            "探索範囲（--search-window）を調整するか、--min-similarity で閾値を変更してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    # レンダリングエンジンの選択と実行
    use_ffmpeg = False
    if args.engine == "ffmpeg":
        if not has_ffmpeg():
            print(
                "エラー: --engine ffmpeg が指定されましたが、ffmpegが見つかりません。",
                file=sys.stderr,
            )
            sys.exit(1)
        use_ffmpeg = True
    elif args.engine == "auto":
        use_ffmpeg = has_ffmpeg()

    success = False
    if use_ffmpeg:
        success = concat_videos_ffmpeg(
            v1_path,
            v2_path,
            overlap,
            output_path,
            transition=args.transition,
            transition_sec=args.fade_duration,
        )

    if not success:
        # OpenCVによるフォールバックまたは直接レンダリング
        concat_videos_opencv(
            meta1,
            meta2,
            overlap,
            output_path,
            transition=args.transition,
            transition_sec=args.fade_duration,
        )

    print(f"\n連結動画を出力しました: {output_path}")


if __name__ == "__main__":
    main()
