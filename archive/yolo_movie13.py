import os
import cv2
import gc
import torch
import subprocess
import shlex
from ultralytics import YOLO
from multiprocessing import Process, Manager, cpu_count, Value, Lock
import time
import threading
from queue import Queue

# パス設定
INPUT_PATH = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_with_boxes.mp4")
MODEL_PATH = r"yolo11n_quadcopter.pt"

# 4K最適化設定
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE_TYPE}")

if DEVICE_TYPE == "cuda":
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    NUM_PROCESSES = 2
    print(f"GPU memory: {gpu_memory_gb:.1f}GB, using {NUM_PROCESSES} processes for 4K optimization")
else:
    NUM_PROCESSES = max(1, cpu_count() - 1)

THREADS_PER_PROCESS = max(1, cpu_count() // NUM_PROCESSES)

def worker(frame_range, boxes_dict, counter, lock, total_frames, model_path, device_type, process_id, process_stats):
    """4K推論最適化ワーカー（正確なFPS表示版）"""
    torch.set_num_threads(THREADS_PER_PROCESS)
    
    if device_type == "cuda":
        torch.cuda.set_device(0)
        device = "cuda:0"
        torch.cuda.empty_cache()
        
        # 4K推論用メモリ設定
        torch.cuda.set_per_process_memory_fraction(0.4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"
    
    try:
        model = YOLO(model_path)
        print(f"Worker {process_id}: Model loaded on {device}")
    except Exception as e:
        print(f"Worker {process_id}: Error loading model: {e}")
        device = "cpu"
        model = YOLO(model_path)

    cap = cv2.VideoCapture(INPUT_PATH)
    start_idx, end_idx = frame_range
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    # 元画像サイズ取得
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 推論用低画質サイズ（高速化）
    infer_width, infer_height = 640, 384

    processed_frames = 0
    total_inference_time = 0.0
    process_start_time = time.time()

    for idx in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break

        # 推論用にリサイズ
        infer_frame = cv2.resize(frame, (infer_width, infer_height))

        inference_start = time.time()
        try:
            results = model.predict(
                infer_frame,
                conf=0.25,
                verbose=False,
                device=device,
                imgsz=(infer_width, infer_height),
                half=True,
                augment=False,
                agnostic_nms=False
            )
        except RuntimeError as e:
            if "CUDA" in str(e):
                device = "cpu"
                results = model.predict(infer_frame, conf=0.25, verbose=False, device=device)
            else:
                raise e

        inference_end = time.time()
        inference_time = inference_end - inference_start
        total_inference_time += inference_time

        # 検出結果（座標スケーリング）
        detections = []
        boxes = results[0].boxes
        if boxes is not None:
            names = model.names
            for r in boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = r
                # スケーリング
                x1 = int(x1 * orig_width / infer_width)
                y1 = int(y1 * orig_height / infer_height)
                x2 = int(x2 * orig_width / infer_width)
                y2 = int(y2 * orig_height / infer_height)
                class_id = int(cls)
                class_name = names[class_id] if class_id < len(names) else str(class_id)
                detections.append((x1, y1, x2, y2, class_id, float(conf), class_name))

        boxes_dict[idx] = detections
        
        # メモリ管理
        del frame, infer_frame, results
        if device_type == "cuda" and processed_frames % 50 == 0:
            torch.cuda.empty_cache()
        
        processed_frames += 1
        
        # ワーカー統計更新
        now = time.time()
        process_elapsed = now - process_start_time
        if process_elapsed > 0:
            current_fps = processed_frames / process_elapsed
            # リアルタイム統計更新
            process_stats[process_id] = {
                'frames': processed_frames,
                'total_frames': end_idx - start_idx,
                'fps': current_fps,
                'elapsed': process_elapsed
            }
        
        # カウンター更新
        with lock:
            counter.value += 1

    cap.release()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    
    # 最終統計更新
    final_elapsed = time.time() - process_start_time
    final_fps = processed_frames / final_elapsed if final_elapsed > 0 else 0
    process_stats[process_id] = {
        'frames': processed_frames,
        'total_frames': end_idx - start_idx,
        'fps': final_fps,
        'elapsed': final_elapsed,
        'completed': True
    }
    
    print(f"Worker {process_id} completed: {processed_frames}/{end_idx-start_idx} frames at {final_fps:.1f} FPS")

def main():
    # 4K全体最適化
    if DEVICE_TYPE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 動画情報取得
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("Failed to open video file.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"4K Video: {total_frames} frames, {fps} FPS, {width}x{height}")

    # プロセス分割
    chunk = total_frames // NUM_PROCESSES
    ranges = []
    for i in range(NUM_PROCESSES):
        start = i * chunk
        end = (i + 1) * chunk if i < NUM_PROCESSES - 1 else total_frames
        ranges.append((start, end))
        print(f"Worker {i}: frames {start}-{end-1} ({end-start} frames)")

    manager = Manager()
    boxes_dict = manager.dict()
    counter = Value('i', 0)
    lock = Lock()
    process_stats = manager.dict()

    # 推論実行
    print("Starting 4K YOLO inference...")
    print("=" * 70)
    inference_start_time = time.time()
    
    processes = []
    for i, fr in enumerate(ranges):
        p = Process(
            target=worker,
            args=(fr, boxes_dict, counter, lock, total_frames, MODEL_PATH, DEVICE_TYPE, i, process_stats)
        )
        p.start()
        processes.append(p)
    
    # 正確なワーカー別進捗表示
    last_log_time = time.time()
    
    while any(p.is_alive() for p in processes):
        now = time.time()
        if now - last_log_time >= 1.5:  # 1.5秒ごと更新
            
            # ワーカー別統計収集
            worker_info = []
            total_worker_fps = 0
            total_processed_frames = 0
            
            for pid in range(NUM_PROCESSES):
                if pid in process_stats:
                    stats = process_stats[pid]
                    worker_fps = stats.get('fps', 0)
                    completed = stats.get('frames', 0)
                    total_frames_worker = stats.get('total_frames', 1)
                    
                    total_worker_fps += worker_fps
                    total_processed_frames += completed
                    
                    worker_info.append(f"W{pid}: {worker_fps:.1f}fps ({completed}/{total_frames_worker})")
                else:
                    worker_info.append(f"W{pid}: starting...")
            
            # 進捗計算
            progress_pct = 100 * total_processed_frames / total_frames
            
            # 正確な表示（ワーカー別 + 合計）
            workers_display = " | ".join(worker_info)
            status_line = f"🚀 {workers_display} | 💯 Combined: {total_worker_fps:.1f}fps | Progress: {progress_pct:.1f}%"
            print(f"\r{status_line}", end="", flush=True)
            
            last_log_time = now
        
        time.sleep(0.3)
    
    # プロセス終了待機
    for p in processes:
        p.join()

    inference_time = time.time() - inference_start_time
    
    # 最終結果表示（正確な計算）
    print(f"\n{'='*70}")
    print(f"🎉 4K YOLO Inference Complete!")
    print(f"   📊 Total processing time: {inference_time:.2f}s")
    
    # ワーカー別詳細表示
    worker_fps_list = []
    for pid in range(NUM_PROCESSES):
        if pid in process_stats:
            stats = process_stats[pid]
            worker_fps = stats.get('fps', 0)
            frames_processed = stats.get('frames', 0)
            worker_fps_list.append(worker_fps)
            print(f"   🔥 Worker {pid}: {frames_processed} frames at {worker_fps:.1f} FPS")
    
    # 正確な合計計算
    true_combined_fps = sum(worker_fps_list)
    actual_overall_fps = total_frames / inference_time
    
    print(f"   ⚡ True Combined FPS: {true_combined_fps:.1f} (sum of workers)")
    print(f"   📈 Actual Overall FPS: {actual_overall_fps:.1f} (total frames/time)")
    print(f"   🎯 Parallel Efficiency: {(actual_overall_fps/true_combined_fps)*100:.1f}%")
    
    # 並列処理分析
    if len(worker_fps_list) == 2:
        efficiency = (actual_overall_fps/true_combined_fps)*100
        if efficiency > 95:
            analysis = "Excellent parallel efficiency! 🏆"
        elif efficiency > 90:
            analysis = "Very good parallel efficiency! ✅"
        elif efficiency > 80:
            analysis = "Good parallel efficiency ✅"
        else:
            analysis = "Some parallel overhead detected ⚠️"
        print(f"   📊 Analysis: {analysis}")

    # 🚀 超高速エンコード設定
    print(f"\n{'='*70}")
    print("🎬 Starting ULTRA-FAST 4K NVENC Encoding...")
    start_encode_time = time.time()
    
    if DEVICE_TYPE == "cuda":
        ffmpeg_path = "ffmpeg"
        # 超高速4K設定（ビットレート最適化）
        cmd = (
            f"{ffmpeg_path} -y "
            "-hwaccel cuda -hwaccel_output_format cuda "
            f"-f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - "
            "-vf \"format=bgr24,hwupload_cuda\" "
            "-c:v h264_nvenc "
            "-preset p1 "                    # 最高速プリセット
            "-rc cbr "                       # 固定ビットレート
            "-b:v 20M "                      # 20Mbps（高速化）
            "-maxrate 25M "                  # 最大レート制限
            "-bufsize 10M "                  # バッファサイズ
            "-bf 0 "                         # Bフレーム完全無効
            "-refs 1 "                       # 参照フレーム最小
            "-rc-lookahead 0 "               # 先読み完全無効
            "-tune ll "                      # 超低レイテンシ
            "-no-scenecut "                  # シーンカット無効
            f"\"{OUTPUT_PATH}\""
        )
        print("Using NVENC Ultra-Fast mode: 20Mbps CBR")
    else:
        cmd = (
            f"{ffmpeg_path} -y "
            f"-f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - "
            "-c:v libx264 -preset ultrafast -crf 23 -threads 8 "
            f"\"{OUTPUT_PATH}\""
        )

    # 高速非同期エンコード（バッファ拡大）
    frame_queue = Queue(maxsize=120)  # 4秒分バッファ
    encoding_error = threading.Event()
    
    def ultra_fast_encoder():
        try:
            proc = subprocess.Popen(
                shlex.split(cmd), 
                stdin=subprocess.PIPE, 
                stderr=subprocess.DEVNULL,
                bufsize=1024*1024  # 1MBバッファ
            )
            
            frames_written = 0
            while True:
                frame_data = frame_queue.get()
                if frame_data is None:
                    break
                    
                try:
                    proc.stdin.write(frame_data)
                    frames_written += 1
                except (BrokenPipeError, OSError):
                    print(f"\nEncoder pipe broken at frame {frames_written}")
                    encoding_error.set()
                    break
            
            proc.stdin.close()
            return_code = proc.wait()
            if return_code != 0:
                print(f"\nFFmpeg exited with code: {return_code}")
                encoding_error.set()
                
        except Exception as e:
            print(f"\nEncoder critical error: {e}")
            encoding_error.set()

    encoder_thread = threading.Thread(target=ultra_fast_encoder)
    encoder_thread.start()

    # 高速フレーム処理
    cap = cv2.VideoCapture(INPUT_PATH)
    idx = 0
    frames_encoded = 0
    last_fps_time = time.time()
    fps_count = 0
    frame_skip_count = 0

    print("🚀 Ultra-fast 4K frame processing...")

    while True:
        if encoding_error.is_set():
            print("\nEncoding error detected, stopping...")
            break
            
        ret, frame = cap.read()
        if not ret:
            break

        # 描画処理（最適化）
        dets = boxes_dict.get(idx, [])
        if dets:  # 検出がある場合のみ描画
            thickness = 6  # 固定値で高速化
            font_scale = 1.5
            
            for x1, y1, x2, y2, cls, conf, class_name in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # 非同期エンコード送信（タイムアウト拡大）
        try:
            frame_queue.put(frame.tobytes(), timeout=0.5)  # 500msタイムアウト
            frames_encoded += 1
        except:
            frame_skip_count += 1

        idx += 1
        fps_count += 1

        # 高速進捗表示
        now = time.time()
        if now - last_fps_time >= 3.0:  # 3秒ごと
            elapsed = now - last_fps_time
            encode_fps = fps_count / elapsed if elapsed > 0 else 0
            progress = 100 * frames_encoded / total_frames
            realtime_speed = encode_fps / fps if fps > 0 else 0
            
            print(f"\r🔥 Ultra Encoding: {progress:.1f}% | {encode_fps:.1f}fps ({realtime_speed:.1f}x) | Queue: {frame_queue.qsize()}", end="", flush=True)
            last_fps_time = now
            fps_count = 0

    # 終了処理
    frame_queue.put(None)
    encoder_thread.join(timeout=30)
    
    if encoder_thread.is_alive():
        print("\nWarning: Encoder thread did not finish cleanly")
    
    cap.release()

    encode_time = time.time() - start_encode_time
    total_time = inference_time + encode_time
    overall_speed = (total_frames / fps) / total_time
    
    print(f"\n{'='*70}")
    print(f"🚀 4K ULTRA-FAST Processing Complete!")
    print(f"   📊 Inference: {inference_time:.2f}s ({actual_overall_fps:.1f} FPS)")
    print(f"   ⚡ Encoding:  {encode_time:.2f}s ({(total_frames/fps)/encode_time:.1f}x realtime)")
    print(f"   🔥 Total:     {total_time:.2f}s ({overall_speed:.1f}x realtime)")
    print(f"   📉 Skipped frames: {frame_skip_count}")
    print(f"   💾 Output: {OUTPUT_PATH}")
    print(f"{'='*70}")
    
    if DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
