import os
import cv2
import gc
import torch
from ultralytics import YOLO
from multiprocessing import Process, Manager, cpu_count, Value, Lock

# 元のパス設定
INPUT_PATH  = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_with_boxes.mp4")
MODEL_PATH  = r"yolo11n_quadcopter.pt"

# 並列プロセス数
NUM_PROCESSES = max(1, cpu_count() - 1)

# PyTorch スレッド数設定（各プロセス内）
THREADS_PER_PROCESS = max(1, cpu_count() // NUM_PROCESSES)
torch.set_num_threads(THREADS_PER_PROCESS)

def worker(frame_range, boxes_dict, counter, lock, total_frames, model_path):
    torch.set_num_threads(THREADS_PER_PROCESS)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(INPUT_PATH)
    start_idx, end_idx = frame_range
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    for idx in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        detections = []
        for r in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, _, cls = r
            detections.append((int(x1), int(y1), int(x2), int(y2), int(cls)))

        boxes_dict[idx] = detections
        del frame, results
        gc.collect()

        # 進捗更新
        with lock:
            counter.value += 1
            print(f"\rInference progress: {counter.value}/{total_frames}", end="", flush=True)

    cap.release()

def main():
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("動画ファイルを開けませんでした。")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # フレーム範囲を均等分割
    chunk = total_frames // NUM_PROCESSES
    ranges = []
    for i in range(NUM_PROCESSES):
        start = i * chunk
        end   = (i + 1) * chunk if i < NUM_PROCESSES - 1 else total_frames
        ranges.append((start, end))

    manager    = Manager()
    boxes_dict = manager.dict()
    counter    = Value('i', 0)
    lock       = Lock()

    processes = []
    for fr in ranges:
        p = Process(
            target=worker,
            args=(fr, boxes_dict, counter, lock, total_frames, MODEL_PATH)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nInference complete. Rendering video...")

    cap = cv2.VideoCapture(INPUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = boxes_dict.get(idx, [])
        for x1, y1, x2, y2, cls in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    print(f"Done: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
