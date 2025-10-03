import os
import cv2
import gc
import torch
import subprocess
import shlex
from ultralytics import YOLO
from multiprocessing import Process, Manager, cpu_count, Value, Lock
import time

# パス設定
INPUT_PATH = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-29\Manual.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_with_boxes.mp4")
MODEL_PATH = r"yolo11n_quadcopter.pt"

# GPU/CPU自動判定
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE_TYPE}")

if DEVICE_TYPE == "cuda":
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    NUM_PROCESSES = min(2, max(1, int(gpu_memory_gb // 4)))
    print(f"GPU memory: {gpu_memory_gb:.1f}GB, using {NUM_PROCESSES} processes")
else:
    NUM_PROCESSES = max(1, cpu_count() - 1)
    print(f"CPU mode: using {NUM_PROCESSES} processes")

THREADS_PER_PROCESS = max(1, cpu_count() // NUM_PROCESSES)


def worker(frame_range, boxes_dict, counter, lock, total_frames, model_path, device_type, process_id):
    """最適化されたワーカープロセス"""
    torch.set_num_threads(THREADS_PER_PROCESS)
    
    if device_type == "cuda":
        torch.cuda.set_device(0)
        device = "cuda:0"
        torch.cuda.empty_cache()
        # メモリ使用量最適化
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
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

        # YOLO推論（最適化）
        inference_start = time.time()
        try:
            results = model.predict(frame, conf=0.25, verbose=False, device=device, stream=False)
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
        
        # 効率的なメモリクリーンアップ
        del frame, results
        if device_type == "cuda" and processed_frames % 50 == 0:  # 50フレームごとにクリーンアップ
            torch.cuda.empty_cache()
        
        processed_frames += 1
        now = time.time()

        # ログ表示
        if now - last_log_time >= 1.0 or processed_frames == (end_idx - start_idx):
            yolo_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
            with lock:
                current_progress = counter.value + 1
                print(f"\rProcess {process_id} ({device}): {current_progress}/{total_frames} | YOLO FPS: {yolo_fps:.1f}", end="", flush=True)
            last_log_time = now

        with lock:
            counter.value += 1

    cap.release()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nProcess {process_id} completed using {device}")


def main():
    # GPU最適化設定
    if DEVICE_TYPE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
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

    # フレーム範囲分割
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

    # 推論プロセス実行
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

    print("\nInference complete. Starting optimized encoding...")

    # 最速エンコード設定
    start_encode_time = time.time()
    
    if DEVICE_TYPE == "cuda":
        print("Using GPU encoding (NVENC) - Maximum Speed Mode")
        ffmpeg_path = "ffmpeg"
        cmd = (
            f"{ffmpeg_path} -y "
            "-hwaccel cuda -hwaccel_output_format cuda "
            f"-f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - "
            "-vf \"format=bgr24,hwupload_cuda\" "
            "-c:v h264_nvenc "
            "-preset p2 "                    # 高速設定（p1より品質重視、p7より高速）
            "-rc vbr "                      # シンプルVBR（HQ削除で高速化）
            "-cq 19 "                       # 品質維持
            "-rc-lookahead 8 "              # 先読み削減（高速化）
            "-bf 0 "                        # Bフレーム削除（高速化）
            "-refs 1 "                      # 参照フレーム削減
            f"\"{OUTPUT_PATH}\""
        )
    else:
        print("Using CPU encoding (x264)")
        ffmpeg_path = "ffmpeg"
        cmd = (
            f"{ffmpeg_path} -y "
            f"-f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - "
            "-c:v libx264 -preset faster -crf 19 "
            f"\"{OUTPUT_PATH}\""
        )

    try:
        proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg.")
        return

    # 描画＋エンコードループ（最適化）
    cap = cv2.VideoCapture(INPUT_PATH)
    idx = 0
    frames_processed = 0
    
    print("Rendering and encoding...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 描画処理
        dets = boxes_dict.get(idx, [])
        for x1, y1, x2, y2, cls, conf, class_name in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # エンコード
        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            break
            
        idx += 1
        frames_processed += 1
        
        # 進捗表示
        if frames_processed % 100 == 0:
            print(f"\rEncoding: {frames_processed}/{total_frames}", end="", flush=True)

    cap.release()
    proc.stdin.close()
    proc.wait()

    encode_time = time.time() - start_encode_time
    speed_multiplier = (total_frames / fps) / encode_time
    
    print(f"\nDone: {OUTPUT_PATH}")
    print(f"Encoding time: {encode_time:.2f}s")
    print(f"Encoding speed: {speed_multiplier:.2f}x realtime")
    
    # GPU メモリクリーンアップ
    if DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
