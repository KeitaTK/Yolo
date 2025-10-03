# -------------------------------------------------------------
# 動画ファイルにYOLOで物体検出を行い、検出結果（矩形・クラス名・確率）を描画して
# GPU/CPU自動判定・並列推論・FFmpegによる高速エンコードで出力するスクリプト
# - ultralytics YOLOモデル使用
# - GPUメモリ量に応じてプロセス数自動調整
# - 検出結果にクラス名と信頼度（確率）を表示
# -------------------------------------------------------------

import os
import cv2
import gc
import torch
import subprocess
import shlex
from ultralytics import YOLO
from multiprocessing import Process, Manager, cpu_count, Value, Lock
import time

# INPUT_PATH  = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
# INPUT_PATH  = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\Guided.mp4"
# INPUT_PATH  = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\1台位置制御.mp4"
INPUT_PATH = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-29\Manual.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_with_boxes.mp4")
MODEL_PATH = r"yolo11n_quadcopter.pt"

# GPU/CPU自動判定
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE_TYPE}")

# GPU使用時はプロセス数を制限（メモリ競合回避）
if DEVICE_TYPE == "cuda":
    # GPUメモリに応じて調整（8GB未満なら1プロセス、以上なら2プロセス）
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    NUM_PROCESSES = min(2, max(1, int(gpu_memory_gb // 4)))
    print(f"GPU memory: {gpu_memory_gb:.1f}GB, using {NUM_PROCESSES} processes")
else:
    NUM_PROCESSES = max(1, cpu_count() - 1)
    print(f"CPU mode: using {NUM_PROCESSES} processes")

# PyTorch スレッド数設定
THREADS_PER_PROCESS = max(1, cpu_count() // NUM_PROCESSES)


def worker(frame_range, boxes_dict, counter, lock, total_frames, model_path, device_type, process_id):
    """ワーカープロセス - GPU/CPU自動切り替え対応"""
    torch.set_num_threads(THREADS_PER_PROCESS)
    
    # GPU使用時はプロセス毎にGPUメモリを管理
    if device_type == "cuda":
        torch.cuda.set_device(0)  # GPU 0を使用
        device = f"cuda:0"
        # GPUメモリクリア
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    
    try:
        model = YOLO(model_path)
        print(f"Process {process_id}: Model loaded on {device}")
    except Exception as e:
        print(f"Process {process_id}: Model loading failed on {device}, falling back to CPU")
        device = "cpu"
        model = YOLO(model_path)

    cap = cv2.VideoCapture(INPUT_PATH)
    start_idx, end_idx = frame_range
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    processed_frames = 0
    total_inference_time = 0.0
    last_log_time = time.time()

    for idx in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO推論（デバイス指定）
        inference_start = time.time()
        try:
            results = model.predict(frame, conf=0.25, verbose=False, device=device)
        except RuntimeError as e:
            if "CUDA" in str(e) and device == "cuda:0":
                print(f"Process {process_id}: CUDA error, falling back to CPU")
                device = "cpu"
                results = model.predict(frame, conf=0.25, verbose=False, device=device)
            else:
                raise e
        
        inference_end = time.time()
        inference_time = inference_end - inference_start
        total_inference_time += inference_time

        detections = []
        boxes = results[0].boxes
        if boxes is not None:
            names = model.names
            for r in boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = r
                class_id = int(cls)
                class_name = names[class_id] if class_id < len(names) else str(class_id)
                detections.append((int(x1), int(y1), int(x2), int(y2), class_id, float(conf), class_name))

        boxes_dict[idx] = detections
        
        # メモリクリーンアップ
        del frame, results
        if device_type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        processed_frames += 1
        now = time.time()

        # 1秒ごとにYOLO推論FPSを表示
        if now - last_log_time >= 1.0 or processed_frames == (end_idx - start_idx):
            yolo_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
            with lock:
                current_progress = counter.value + 1
                print(f"\rProcess {process_id} ({device}): {current_progress}/{total_frames} | YOLO FPS: {yolo_fps:.1f}", end="", flush=True)
            last_log_time = now

        with lock:
            counter.value += 1

    cap.release()
    
    # GPU メモリクリーンアップ
    if device_type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nProcess {process_id} completed using {device}")


def main():
    # 動画プロパティ取得
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("動画ファイルを開けませんでした。")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

    # フレーム範囲を均等分割
    chunk = total_frames // NUM_PROCESSES
    ranges = []
    for i in range(NUM_PROCESSES):
        start = i * chunk
        end = (i + 1) * chunk if i < NUM_PROCESSES - 1 else total_frames
        ranges.append((start, end))

    manager = Manager()
    boxes_dict = manager.dict()
    counter = Value('i', 0)
    lock = Lock()

    # 推論プロセス起動
    processes = []
    for i, fr in enumerate(ranges):
        p = Process(
            target=worker,
            args=(fr, boxes_dict, counter, lock, total_frames, MODEL_PATH, DEVICE_TYPE, i)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print("\nInference complete. Rendering video...")

    # エンコード方式を自動選択
    if DEVICE_TYPE == "cuda":
        print("Using GPU encoding (NVENC)")
        # FFmpeg NVENC コマンド
        ffmpeg_path = "ffmpeg"
        cmd = (
            f"{ffmpeg_path} -y "
            "-hwaccel cuda -hwaccel_output_format cuda "
            f"-f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - "
            "-vf \"format=bgr24,hwupload_cuda\" "
            "-c:v h264_nvenc -preset p7 -rc vbr_hq -cq 19 "
            f"\"{OUTPUT_PATH}\""
        )
    else:
        print("Using CPU encoding (x264)")
        # CPU エンコード（NVENC利用不可時のフォールバック）
        ffmpeg_path = "ffmpeg"
        cmd = (
            f"{ffmpeg_path} -y "
            f"-f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - "
            "-c:v libx264 -preset medium -crf 23 "
            f"\"{OUTPUT_PATH}\""
        )

    try:
        proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg and add to PATH.")
        return

    # 描画＋エンコードループ
    cap = cv2.VideoCapture(INPUT_PATH)
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        dets = boxes_dict.get(idx, [])
        for x1, y1, x2, y2, cls, conf, class_name in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("FFmpeg process terminated unexpectedly")
            break
            
        idx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()

    print(f"Done: {OUTPUT_PATH}")
    
    # 最終GPU メモリクリーンアップ
    if DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()






