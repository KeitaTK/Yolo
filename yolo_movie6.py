import os
import time
import cv2
import gc
import torch
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count

# モデルとパス（元のまま）
MODEL_PATH  = r"yolo11n_quadcopter.pt"
INPUT_PATH  = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_detected.mp4")

# バッチサイズと並列度設定
BATCH_SIZE = 8
CPU_CORES = cpu_count()
THREADS_PER_WORKER = max(1, CPU_CORES // 2)
WORKERS = max(1, CPU_CORES // THREADS_PER_WORKER)

# プロセス起動時にモデルを一度だけロード
def init_worker(model_path):
    torch.set_num_threads(THREADS_PER_WORKER)
    global model
    model = YOLO(model_path)

# バッチ推論：[(idx, frame), ...] -> [(idx, result), ...]
def infer_batch(batch):
    idxs, frames = zip(*batch)
    resized = [cv2.resize(f, (1280, 720)) for f in frames]
    results = model.predict(resized, conf=0.25, verbose=False)
    return list(zip(idxs, results))

def main():
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("動画を開けませんでした。")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    pool = Pool(processes=WORKERS, initializer=init_worker, initargs=(MODEL_PATH,))

    start_time = time.time()
    processed = 0
    batch = []
    idx = 0
    in_flight = []

    # フレームをストリーミング読み込み＆逐次バッチ送信
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append((idx, frame))
        idx += 1

        if len(batch) >= BATCH_SIZE:
            # apply_async でバッチ送信
            fut = pool.apply_async(infer_batch, (batch.copy(),))
            in_flight.append(fut)
            batch.clear()

            # 同時 in_flight を WORKERS*2 に制限
            while len(in_flight) > WORKERS * 2:
                old = in_flight.pop(0)
                for i, res in old.get():
                    img = res.plot()
                    out.write(img)
                    del img, res
                    gc.collect()
                    processed += 1
                    print(f"\r{processed}/{total} フレーム処理完了", end="", flush=True)

    # 余りのバッチを送信
    if batch:
        fut = pool.apply_async(infer_batch, (batch.copy(),))
        in_flight.append(fut)

    cap.release()
    pool.close()

    # 残 in_flight を回収
    for fut in in_flight:
        for i, res in fut.get():
            img = res.plot()
            out.write(img)
            del img, res
            gc.collect()
            processed += 1
            print(f"\r{processed}/{total} フレーム処理完了", end="", flush=True)

    pool.join()
    out.release()

    elapsed = time.time() - start_time
    print(f"\n完了。結果は {OUTPUT_PATH} に保存されました。")
    print(f"処理時間: {elapsed:.2f} 秒")

if __name__ == "__main__":
    main()
