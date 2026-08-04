import os
import csv
import argparse
import datetime
from pathlib import Path
import cv2
from tqdm import tqdm

# 対象とする動画ファイルの拡張子一覧（大文字小文字を区別せず判定するためすべて小文字で定義）
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".ogg",
    ".ogv",
}


def get_video_metadata(file_path):
    """指定された動画ファイルのメタデータ（解像度、フレーム率、長さ、総ビットレート）を取得します。

    Args:
        file_path (str or Path): 動画ファイルのパス。

    Returns:
        dict: メタデータを含む辞書。取得に失敗した場合は各値が None。
    """
    metadata = {
        "width": None,
        "height": None,
        "fps": None,
        "duration": None,
        "bitrate_kbps": None,
    }
    try:
        path_str = str(Path(file_path).resolve())
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            return metadata

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if width == 0 or height == 0:
            return metadata

        metadata["width"] = width
        metadata["height"] = height
        metadata["fps"] = fps

        # 長さ（秒）の計算
        duration = 0.0
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            metadata["duration"] = duration

        # 総ビットレート（kbps）の計算
        file_size = os.path.getsize(path_str)
        if duration > 0:
            # (バイト数 * 8ビット) / 1000 = キロビット
            # キロビット / 秒 = kbps
            bitrate_kbps = (file_size * 8 / 1000) / duration
            metadata["bitrate_kbps"] = bitrate_kbps

        return metadata
    except Exception as e:
        print(f"警告: {file_path} のメタデータ取得中にエラーが発生しました: {e}")
        return metadata


def is_video_file(file_path):
    """ファイルが動画ファイルであるかを拡張子で判定します。

    Args:
        file_path (Path): 判定対象のファイルパス。

    Returns:
        bool: 動画ファイルの場合は True、そうでない場合は False。
    """
    return file_path.suffix.lower() in VIDEO_EXTENSIONS


def find_video_files(directory_path):
    """指定されたディレクトリを再帰的に探索し、動画ファイルのパスを収集します。

    Args:
        directory_path (str or Path): 探索対象のディレクトリパス。

    Returns:
        list: 動画ファイルの Path オブジェクトのリスト。
    """
    video_files = []
    dir_path = Path(directory_path)

    if not dir_path.is_dir():
        print(f"エラー: {directory_path} は有効なディレクトリではありません。")
        return video_files

    for path in dir_path.rglob("*"):
        try:
            if path.is_file() and is_video_file(path):
                video_files.append(path)
        except Exception as e:
            # アクセス権限のないファイルやシンボリックリンクのエラー対策
            print(f"警告: {path} のアクセス中にエラーが発生しました: {e}")

    return video_files


def collect_video_info(video_files):
    """動画ファイルの一覧から情報を収集します。"""
    results = []
    for path in tqdm(video_files, desc="動画情報抽出中"):
        full_path = str(path.resolve())
        meta = get_video_metadata(full_path)

        res_str = (
            f"{meta['width']}x{meta['height']}"
            if meta["width"] and meta["height"]
            else ""
        )
        total_res = (
            meta["width"] * meta["height"] if meta["width"] and meta["height"] else ""
        )
        fps_str = f"{meta['fps']:.2f}" if meta["fps"] and meta["fps"] > 0 else ""
        dur_str = (
            str(datetime.timedelta(seconds=int(meta["duration"])))
            if meta["duration"] is not None
            else ""
        )
        bitrate_val = (
            int(round(meta["bitrate_kbps"])) if meta["bitrate_kbps"] is not None else ""
        )

        results.append(
            {
                "ファイル名": path.name,
                "フルパス": full_path,
                "解像度(横×縦)": res_str,
                "合計解像度": total_res,
                "フレーム率": fps_str,
                "総ビットレート (kbps)": bitrate_val,
                "長さ": dur_str,
            }
        )

    return results


def write_to_csv(results, output_path):
    """収集した動画情報をCSVファイルに書き出します。

    Windows環境のExcel等で文字化けせずに開けるよう、BOM付きUTF-8 (utf-8-sig) で出力します。

    Args:
        results (list): 動画情報の辞書リスト。
        output_path (str or Path): 出力するCSVファイルのパス。
    """
    fieldnames = [
        "ファイル名",
        "フルパス",
        "解像度(横×縦)",
        "合計解像度",
        "フレーム率",
        "総ビットレート (kbps)",
        "長さ",
    ]
    try:
        with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"成功: CSVファイルを出力しました -> {Path(output_path).resolve()}")
    except Exception as e:
        print(
            f"エラー: CSVファイル {output_path} の書き込み中にエラーが発生しました: {e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="指定ディレクトリ内の動画情報を再帰的に探索し、CSVに出力します。"
    )
    parser.add_argument(
        "-d", "--directory", required=True, help="探索対象のディレクトリパス"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="video_info.csv",
        help="出力するCSVファイルのパス (デフォルト: video_info.csv)",
    )
    args = parser.parse_args()

    print(f"探索を開始します: {args.directory}")
    video_files = find_video_files(args.directory)

    if not video_files:
        print("動画ファイルが見つかりませんでした。")
        return

    print(
        f"{len(video_files)} 個の動画ファイルが見つかりました。情報を抽出しています..."
    )
    results = collect_video_info(video_files)

    output_path = Path(args.output)
    write_to_csv(results, output_path)


if __name__ == "__main__":
    main()
