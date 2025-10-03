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
    """4K推論最適化ワーカー"""
    torch.set_num_threads(THREADS_PER_PROCESS)
    
    if device_type == "cuda":
        torch.cuda.set_device(0)
        device = "cuda:0"
        torch.cuda.empty_cache()
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

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    infer_width, infer_height = 640, 384

    processed_frames = 0
    process_start_time = time.time()

    for idx in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break

        infer_frame = cv2.resize(frame, (infer_width, infer_height))

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

        detections = []
        boxes = results[0].boxes
        if boxes is not None:
            names = model.names
            for r in boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = r
                x1 = int(x1 * orig_width / infer_width)
                y1 = int(y1 * orig_height / infer_height)
                x2 = int(x2 * orig_width / infer_width)
                y2 = int(y2 * orig_height / infer_height)
                class_id = int(cls)
                class_name = names[class_id] if class_id < len(names) else str(class_id)
                detections.append((x1, y1, x2, y2, class_id, float(conf), class_name))

        boxes_dict[idx] = detections
        
        del frame, infer_frame, results
        if device_type == "cuda" and processed_frames % 50 == 0:
            torch.cuda.empty_cache()
        
        processed_frames += 1
        
        now = time.time()
        process_elapsed = now - process_start_time
        if process_elapsed > 0:
            current_fps = processed_frames / process_elapsed
            process_stats[process_id] = {
                'frames': processed_frames,
                'total_frames': end_idx - start_idx,
                'fps': current_fps,
                'elapsed': process_elapsed
            }
        
        with lock:
            counter.value += 1

    cap.release()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    
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
    if DEVICE_TYPE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
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
    
    last_log_time = time.time()
    
    while any(p.is_alive() for p in processes):
        now = time.time()
        if now - last_log_time >= 1.5:
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
            
            progress_pct = 100 * total_processed_frames / total_frames
            workers_display = " | ".join(worker_info)
            status_line = f"🚀 {workers_display} | 💯 Combined: {total_worker_fps:.1f}fps | Progress: {progress_pct:.1f}%"
            print(f"\r{status_line}", end="", flush=True)
            
            last_log_time = now
        
        time.sleep(0.3)
    
    for p in processes:
        p.join()

    inference_time = time.time() - inference_start_time
    
    print(f"\n{'='*70}")
    print(f"🎉 4K YOLO Inference Complete!")
    print(f"   📊 Total processing time: {inference_time:.2f}s")
    
    worker_fps_list = []
    for pid in range(NUM_PROCESSES):
        if pid in process_stats:
            stats = process_stats[pid]
            worker_fps = stats.get('fps', 0)
            frames_processed = stats.get('frames', 0)
            worker_fps_list.append(worker_fps)
            print(f"   🔥 Worker {pid}: {frames_processed} frames at {worker_fps:.1f} FPS")
    
    true_combined_fps = sum(worker_fps_list)
    actual_overall_fps = total_frames / inference_time
    
    print(f"   ⚡ True Combined FPS: {true_combined_fps:.1f} (sum of workers)")
    print(f"   📈 Actual Overall FPS: {actual_overall_fps:.1f} (total frames/time)")
    print(f"   🎯 Parallel Efficiency: {(actual_overall_fps/true_combined_fps)*100:.1f}%")

    # 🎬 シンプル・確実なエンコード（修正版）
    print(f"\n{'='*70}")
    print("🎬 Starting Reliable 4K NVENC Encoding...")
    start_encode_time = time.time()
    
    if DEVICE_TYPE == "cuda":
        ffmpeg_path = "ffmpeg"
        # シンプル・確実なNVENC設定
        cmd = [
            ffmpeg_path, "-y",
            "-hwaccel", "cuda",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "fast",        # p1→fast（安定性重視）
            "-rc", "vbr",            # cbr→vbr（品質安定）
            "-cq", "23",             # 品質設定
            "-b:v", "30M",           # 20M→30M（余裕あるビットレート）
            "-maxrate", "40M",
            "-bufsize", "20M",
            OUTPUT_PATH
        ]
        print("Using NVENC Reliable mode: 30Mbps VBR")
    else:
        cmd = [
            ffmpeg_path, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24", 
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            OUTPUT_PATH
        ]

    # 直接的なエンコード（パイプ使用）
    try:
        proc = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0  # バッファリング無効
        )
        print("FFmpeg process started successfully")
        
        # フレーム処理とエンコード
        cap = cv2.VideoCapture(INPUT_PATH)
        idx = 0
        frames_encoded = 0
        last_fps_time = time.time()
        fps_count = 0
        
        print("🚀 Processing and encoding frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 描画処理
            dets = boxes_dict.get(idx, [])
            if dets:
                thickness = 6
                font_scale = 1.5
                
                for x1, y1, x2, y2, cls, conf, class_name in dets:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

            # エンコード送信（エラーハンドリング付き）
            try:
                proc.stdin.write(frame.tobytes())
                proc.stdin.flush()  # 強制フラッシュ
                frames_encoded += 1
            except BrokenPipeError:
                print(f"\nBroken pipe at frame {frames_encoded}")
                break
            except Exception as e:
                print(f"\nWrite error at frame {frames_encoded}: {e}")
                break

            idx += 1
            fps_count += 1

            # 進捗表示
            now = time.time()
            if now - last_fps_time >= 3.0:
                elapsed = now - last_fps_time
                encode_fps = fps_count / elapsed if elapsed > 0 else 0
                progress = 100 * frames_encoded / total_frames
                realtime_speed = encode_fps / fps if fps > 0 else 0
                
                print(f"\r🎬 Encoding: {progress:.1f}% | {encode_fps:.1f}fps ({realtime_speed:.1f}x) | Frame: {frames_encoded}/{total_frames}", end="", flush=True)
                last_fps_time = now
                fps_count = 0

        # 終了処理
        cap.release()
        proc.stdin.close()
        
        # プロセス終了待機
        stdout, stderr = proc.communicate(timeout=60)
        return_code = proc.returncode
        
        if return_code == 0:
            print(f"\n✅ Encoding completed successfully!")
        else:
            print(f"\n❌ FFmpeg error (code {return_code}): {stderr.decode()}")

    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install FFmpeg.")
        return
    except Exception as e:
        print(f"❌ Encoding error: {e}")
        return

    encode_time = time.time() - start_encode_time
    total_time = inference_time + encode_time
    overall_speed = (total_frames / fps) / total_time
    
    print(f"{'='*70}")
    print(f"🚀 4K Processing Complete!")
    print(f"   📊 Inference: {inference_time:.2f}s ({actual_overall_fps:.1f} FPS)")
    print(f"   ⚡ Encoding:  {encode_time:.2f}s ({(total_frames/fps)/encode_time:.1f}x realtime)")
    print(f"   🔥 Total:     {total_time:.2f}s ({overall_speed:.1f}x realtime)")
    print(f"   💾 Output: {OUTPUT_PATH}")
    print(f"{'='*70}")
    
    if DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
