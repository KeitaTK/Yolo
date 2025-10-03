import os
import cv2
import gc
import torch
import subprocess
import numpy as np
import psutil
from ultralytics import YOLO
from multiprocessing import Process, Manager, cpu_count, Value, Lock
import time
import threading

# パス設定
# INPUT_PATH = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
INPUT_PATH = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\1台位置制御.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_with_boxes.mp4")
MODEL_PATH = r"yolo11n_quadcopter.pt"

# 動的最適化設定
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE_TYPE}")

class DynamicOptimizer:
    def __init__(self):
        if DEVICE_TYPE == "cuda":
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.initial_batch_size = min(32, max(8, int(self.gpu_memory_gb * 3)))  # 積極的な初期値
            self.max_batch_size = min(64, int(self.gpu_memory_gb * 4))
            self.initial_processes = min(3, max(2, int(self.gpu_memory_gb // 2.5)))
            self.max_processes = min(4, int(self.gpu_memory_gb // 2))
        else:
            self.initial_batch_size = 4
            self.max_batch_size = 8
            self.initial_processes = max(1, cpu_count() - 1)
            self.max_processes = cpu_count()
        
        self.current_batch_size = self.initial_batch_size
        self.current_processes = self.initial_processes
        self.performance_history = []
        self.optimization_attempts = 0
        self.last_optimization_time = 0
        
        print(f"🔧 Dynamic Optimizer initialized:")
        print(f"   GPU Memory: {self.gpu_memory_gb:.1f}GB")
        print(f"   Initial batch size: {self.initial_batch_size}")
        print(f"   Initial processes: {self.initial_processes}")
        print(f"   Max batch size: {self.max_batch_size}")
        print(f"   Max processes: {self.max_processes}")
    
    def should_optimize(self, current_time):
        """最適化を実行すべきかチェック"""
        return (current_time - self.last_optimization_time) > 15.0  # 15秒間隔
    
    def analyze_performance(self, fps_data, gpu_usage, memory_usage):
        """性能データを分析して最適化提案"""
        current_fps = sum(fps_data.values()) if fps_data else 0
        
        optimization_needed = False
        suggestions = {}
        
        # GPU使用率ベース調整
        if DEVICE_TYPE == "cuda":
            if gpu_usage < 70 and self.current_batch_size < self.max_batch_size:
                # GPU使用率が低い → バッチサイズ増加
                new_batch_size = min(self.max_batch_size, self.current_batch_size * 2)
                if new_batch_size != self.current_batch_size:
                    suggestions['batch_size'] = new_batch_size
                    optimization_needed = True
            
            elif gpu_usage > 95 and memory_usage > 85:
                # GPU/メモリ使用率が高すぎる → バッチサイズ削減
                new_batch_size = max(4, self.current_batch_size // 2)
                if new_batch_size != self.current_batch_size:
                    suggestions['batch_size'] = new_batch_size
                    optimization_needed = True
            
            # FPSベース調整
            if current_fps > 0:
                self.performance_history.append(current_fps)
                if len(self.performance_history) > 3:
                    self.performance_history.pop(0)
                
                # 性能が安定して低い場合
                if len(self.performance_history) >= 3:
                    avg_fps = sum(self.performance_history) / len(self.performance_history)
                    if avg_fps < 60 and gpu_usage < 80 and self.current_processes < self.max_processes:
                        suggestions['processes'] = min(self.max_processes, self.current_processes + 1)
                        optimization_needed = True
        
        return optimization_needed, suggestions
    
    def apply_optimization(self, suggestions):
        """最適化提案を適用"""
        if 'batch_size' in suggestions:
            old_batch = self.current_batch_size
            self.current_batch_size = suggestions['batch_size']
            print(f"\n🔧 Batch size optimized: {old_batch} → {self.current_batch_size}")
        
        if 'processes' in suggestions:
            old_processes = self.current_processes
            self.current_processes = suggestions['processes']
            print(f"\n🔧 Process count optimized: {old_processes} → {self.current_processes}")
        
        self.optimization_attempts += 1
        self.last_optimization_time = time.time()

# グローバル最適化インスタンス
optimizer = DynamicOptimizer()
BATCH_SIZE = optimizer.current_batch_size
NUM_PROCESSES = optimizer.current_processes
THREADS_PER_PROCESS = max(1, cpu_count() // NUM_PROCESSES)

def get_gpu_stats():
    """GPU使用率とメモリ使用率を取得"""
    if DEVICE_TYPE == "cuda":
        try:
            gpu_usage = torch.cuda.utilization()
            memory_used = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_usage = (memory_used / memory_total) * 100
            return gpu_usage, memory_usage
        except:
            return 0, 0
    return 0, 0

def worker_adaptive(frame_range, boxes_dict, counter, lock, total_frames, model_path, device_type, process_id, process_stats, shared_batch_size):
    """動的最適化対応ワーカー"""
    torch.set_num_threads(THREADS_PER_PROCESS)
    
    if device_type == "cuda":
        torch.cuda.set_device(0)
        device = "cuda:0"
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"
    
    try:
        model = YOLO(model_path)
        current_batch = shared_batch_size.value
        print(f"Worker {process_id}: Model loaded on {device} with initial batch size {current_batch}")
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
    total_inference_time = 0.0
    process_start_time = time.time()
    last_batch_check = time.time()
    
    frame_batch = []
    frame_indices = []
    
    for idx in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break

        # 動的バッチサイズ確認（5秒ごと）
        current_time = time.time()
        if current_time - last_batch_check > 5.0:
            new_batch_size = shared_batch_size.value
            if new_batch_size != len(frame_batch) and frame_batch:
                # バッチサイズが変更された場合、現在のバッチを処理
                current_batch_size = len(frame_batch)
            else:
                current_batch_size = new_batch_size
            last_batch_check = current_time
        else:
            current_batch_size = shared_batch_size.value

        infer_frame = cv2.resize(frame, (infer_width, infer_height))
        frame_batch.append(infer_frame)
        frame_indices.append(idx)
        
        # 動的バッチサイズに基づく推論実行
        if len(frame_batch) >= current_batch_size or idx == end_idx - 1:
            batch_inference_start = time.time()
            
            try:
                if len(frame_batch) == 1:
                    batch_results = model.predict(
                        frame_batch[0],
                        conf=0.25,
                        verbose=False,
                        device=device,
                        imgsz=(infer_width, infer_height),
                        half=True,
                        augment=False,
                        agnostic_nms=False
                    )
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                else:
                    batch_results = model.predict(
                        frame_batch,
                        conf=0.25,
                        verbose=False,
                        device=device,
                        imgsz=(infer_width, infer_height),
                        half=True,
                        augment=False,
                        agnostic_nms=False
                    )
                    
            except RuntimeError as e:
                if "CUDA" in str(e) and "out of memory" in str(e).lower():
                    print(f"Worker {process_id}: CUDA OOM, reducing batch size")
                    # OOMエラー時はバッチサイズを半分に
                    with shared_batch_size.get_lock():
                        shared_batch_size.value = max(2, shared_batch_size.value // 2)
                    
                    # 単一フレーム処理にフォールバック
                    device = "cpu"
                    batch_results = []
                    for single_frame in frame_batch:
                        result = model.predict(single_frame, conf=0.25, verbose=False, device=device)
                        batch_results.append(result)
                else:
                    raise e
            
            batch_inference_time = time.time() - batch_inference_start
            total_inference_time += batch_inference_time
            
            # 結果処理
            for i, (frame_idx, result) in enumerate(zip(frame_indices, batch_results)):
                detections = []
                boxes = result.boxes if hasattr(result, 'boxes') else result[0].boxes
                
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
                
                boxes_dict[frame_idx] = detections
            
            processed_frames += len(frame_batch)
            
            # メモリ管理
            del frame_batch, batch_results
            if device_type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            frame_batch = []
            frame_indices = []
        
        # 統計更新
        now = time.time()
        process_elapsed = now - process_start_time
        if process_elapsed > 0 and total_inference_time > 0:
            overall_fps = processed_frames / process_elapsed
            pure_inference_fps = processed_frames / total_inference_time
            process_stats[process_id] = {
                'frames': processed_frames,
                'total_frames': end_idx - start_idx,
                'fps': overall_fps,
                'throughput_fps': pure_inference_fps,
                'elapsed': process_elapsed,
                'inference_time': total_inference_time,
                'current_batch_size': shared_batch_size.value
            }
        
        with lock:
            counter.value += 1

    cap.release()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    
    # 最終統計
    final_elapsed = time.time() - process_start_time
    final_overall_fps = processed_frames / final_elapsed if final_elapsed > 0 else 0
    final_inference_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
    
    process_stats[process_id] = {
        'frames': processed_frames,
        'total_frames': end_idx - start_idx,
        'fps': final_overall_fps,
        'throughput_fps': final_inference_fps,
        'elapsed': final_elapsed,
        'inference_time': total_inference_time,
        'completed': True
    }
    
    print(f"\nWorker {process_id} completed: {processed_frames} frames at {final_overall_fps:.1f} FPS (inference: {final_inference_fps:.1f} FPS)")

def performance_monitor(process_stats, shared_batch_size, monitor_active):
    """性能監視とリアルタイム最適化"""
    print("🔍 Performance monitor started")
    
    while monitor_active.value:
        time.sleep(10)  # 10秒ごとに監視
        
        current_time = time.time()
        if not optimizer.should_optimize(current_time):
            continue
        
        # 現在の性能データ収集
        fps_data = {}
        for pid in range(NUM_PROCESSES):
            if pid in process_stats:
                stats = process_stats[pid]
                fps_data[pid] = stats.get('fps', 0)
        
        # GPU使用率取得
        gpu_usage, memory_usage = get_gpu_stats()
        
        # 最適化判断
        optimization_needed, suggestions = optimizer.analyze_performance(fps_data, gpu_usage, memory_usage)
        
        if optimization_needed:
            print(f"\n🔍 Performance Analysis:")
            print(f"   Current FPS: {sum(fps_data.values()):.1f}")
            print(f"   GPU Usage: {gpu_usage:.1f}%")
            print(f"   Memory Usage: {memory_usage:.1f}%")
            print(f"   Current Batch Size: {shared_batch_size.value}")
            
            if 'batch_size' in suggestions:
                with shared_batch_size.get_lock():
                    shared_batch_size.value = suggestions['batch_size']
                print(f"   🔧 Batch size adjusted to: {suggestions['batch_size']}")
            
            optimizer.apply_optimization(suggestions)
    
    print("🔍 Performance monitor stopped")

def main():
    # GPU最適化
    if DEVICE_TYPE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')
    
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
    print("🚀 Starting ADAPTIVE batch inference with real-time optimization...")

    # プロセス分割
    chunk = total_frames // NUM_PROCESSES
    ranges = []
    for i in range(NUM_PROCESSES):
        start = i * chunk
        end = (i + 1) * chunk if i < NUM_PROCESSES - 1 else total_frames
        ranges.append((start, end))
        print(f"Worker {i}: frames {start}-{end-1} ({end-start} frames)")

    # 共有リソース
    manager = Manager()
    boxes_dict = manager.dict()
    counter = Value('i', 0)
    lock = Lock()
    process_stats = manager.dict()
    shared_batch_size = Value('i', optimizer.current_batch_size)  # 共有バッチサイズ
    monitor_active = Value('b', True)

    print("=" * 80)
    inference_start_time = time.time()
    
    # 性能監視スレッド開始
    monitor_thread = threading.Thread(
        target=performance_monitor,
        args=(process_stats, shared_batch_size, monitor_active)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 推論プロセス開始
    processes = []
    for i, fr in enumerate(ranges):
        p = Process(
            target=worker_adaptive,
            args=(fr, boxes_dict, counter, lock, total_frames, MODEL_PATH, DEVICE_TYPE, i, process_stats, shared_batch_size)
        )
        p.start()
        processes.append(p)
    
    # 進捗監視
    last_log_time = time.time()
    
    while any(p.is_alive() for p in processes):
        now = time.time()
        if now - last_log_time >= 2.0:
            worker_info = []
            total_worker_fps = 0
            total_throughput_fps = 0
            total_processed_frames = 0
            
            for pid in range(NUM_PROCESSES):
                if pid in process_stats:
                    stats = process_stats[pid]
                    worker_fps = stats.get('fps', 0)
                    throughput_fps = stats.get('throughput_fps', 0)
                    completed = stats.get('frames', 0)
                    batch_size = stats.get('current_batch_size', shared_batch_size.value)
                    
                    total_worker_fps += worker_fps
                    total_throughput_fps += throughput_fps
                    total_processed_frames += completed
                    
                    worker_info.append(f"W{pid}: {worker_fps:.1f}fps (T:{throughput_fps:.1f})")
                else:
                    worker_info.append(f"W{pid}: starting...")
            
            progress_pct = 100 * total_processed_frames / total_frames
            gpu_usage, memory_usage = get_gpu_stats()
            
            workers_display = " | ".join(worker_info)
            status_line = f"🔄 {workers_display} | 💯 Combined: {total_worker_fps:.1f}fps | 🚀 Throughput: {total_throughput_fps:.1f}fps"
            status_line += f" | Batch: {shared_batch_size.value} | GPU: {gpu_usage:.0f}% | Mem: {memory_usage:.0f}% | Progress: {progress_pct:.1f}%"
            
            print(f"\r{status_line}", end="", flush=True)
            last_log_time = now
        
        time.sleep(0.5)
    
    # 監視停止
    monitor_active.value = False
    monitor_thread.join(timeout=5)
    
    for p in processes:
        p.join()

    inference_time = time.time() - inference_start_time
    
    print(f"\n{'='*80}")
    print(f"🎉 Adaptive 4K YOLO Inference Complete!")
    print(f"   📊 Total processing time: {inference_time:.2f}s")
    
    worker_fps_list = []
    throughput_fps_list = []
    for pid in range(NUM_PROCESSES):
        if pid in process_stats:
            stats = process_stats[pid]
            worker_fps = stats.get('fps', 0)
            throughput_fps = stats.get('throughput_fps', 0)
            frames_processed = stats.get('frames', 0)
            
            worker_fps_list.append(worker_fps)
            throughput_fps_list.append(throughput_fps)
            
            print(f"   🔥 Worker {pid}: {frames_processed} frames at {worker_fps:.1f} FPS (inference: {throughput_fps:.1f} FPS)")
    
    true_combined_fps = sum(worker_fps_list)
    true_throughput_fps = sum(throughput_fps_list)
    actual_overall_fps = total_frames / inference_time
    
    print(f"   ⚡ Final Combined FPS: {true_combined_fps:.1f}")
    print(f"   🚀 Final Throughput FPS: {true_throughput_fps:.1f}")
    print(f"   📈 Actual Overall FPS: {actual_overall_fps:.1f}")
    print(f"   🎯 Final Batch Size: {shared_batch_size.value}")
    print(f"   🔧 Optimizations Applied: {optimizer.optimization_attempts}")

    # 高速エンコード
    print(f"\n{'='*80}")
    print("🎬 Starting optimized encoding...")
    start_encode_time = time.time()
    
    if DEVICE_TYPE == "cuda":
        ffmpeg_path = "ffmpeg"
        cmd = [
            ffmpeg_path, "-y",
            "-hwaccel", "cuda",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-rc", "vbr",
            "-cq", "23",
            "-b:v", "30M",
            "-maxrate", "40M",
            "-bufsize", "20M",
            OUTPUT_PATH
        ]
        print("Using NVENC optimized encoding")
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

    try:
        proc = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
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

            try:
                proc.stdin.write(frame.tobytes())
                proc.stdin.flush()
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

        cap.release()
        proc.stdin.close()
        stdout, stderr = proc.communicate(timeout=60)
        
        if proc.returncode == 0:
            print(f"\n✅ Encoding completed successfully!")
        else:
            print(f"\n❌ FFmpeg error (code {proc.returncode}): {stderr.decode()}")

    except Exception as e:
        print(f"❌ Encoding error: {e}")
        return

    encode_time = time.time() - start_encode_time
    total_time = inference_time + encode_time
    overall_speed = (total_frames / fps) / total_time
    
    print(f"{'='*80}")
    print(f"🏆 ADAPTIVE 4K Processing Complete!")
    print(f"   📊 Inference: {inference_time:.2f}s ({actual_overall_fps:.1f} FPS)")
    print(f"   ⚡ Encoding:  {encode_time:.2f}s ({(total_frames/fps)/encode_time:.1f}x realtime)")
    print(f"   🔥 Total:     {total_time:.2f}s ({overall_speed:.1f}x realtime)")
    print(f"   🎯 Peak Performance: {true_throughput_fps:.1f} FPS (pure inference)")
    print(f"   🔧 Final Optimizations: Batch={shared_batch_size.value}")
    print(f"   💾 Output: {OUTPUT_PATH}")
    print(f"{'='*80}")
    
    if DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
