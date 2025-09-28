import os
import time
import cv2
import gc
import torch
import logging
from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor, as_completed

# 元のパス設定
MODEL_PATH = r"yolo11n_quadcopter.pt"
INPUT_PATH = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_detected.mp4")

# ハードウェアリソース設定
NUM_CORES = os.cpu_count() or 1
THREADS_PER_WORKER = max(1, NUM_CORES // 2)
MAX_WORKERS = max(1, NUM_CORES // THREADS_PER_WORKER)
BATCH_SIZE = 8

torch.set_num_threads(THREADS_PER_WORKER)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Frame %(frame)d processed in batch %(batch)d",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# プロセス起動時にモデルを一度だけロード
def init_worker(model_path):
    global model
    model = YOLO(model_path)

# バッチ推論処理：フレームのリストを受け取り、(idx, results) ペアを返す
def infer_batch(args):
    batch_indices, batch_frames, batch_id = args
    results = model.predict(batch_frames, conf=0.25, verbose=False)
    return batch_id, list(zip(batch_indices, results))

if __name__ == "__main__":
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("動画を開けませんでした。")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    start_time = time.time()
    processed = 0

    # ProcessPoolExecutor の準備
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
        initargs=(MODEL_PATH,)
    ) as executor:
        futures = []
        batch = []
        batch_id = 0
        idx = 0

        # フレーム読み込みとバッチのキューイング
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append((idx, frame))
            idx += 1

            if len(batch) >= BATCH_SIZE:
                futures.append(executor.submit(infer_batch, ( [i for i, _ in batch], [f for _, f in batch], batch_id )))
                batch = []
                batch_id += 1

        # 残りのフレーム
        if batch:
            futures.append(executor.submit(infer_batch, ( [i for i, _ in batch], [f for _, f in batch], batch_id )))

        cap.release()

        # 推論結果受信、描画・書き出し、ログ出力
        for future in as_completed(futures):
            bid, results = future.result()
            for idx, res in results:
                img = res.plot()
                out.write(img)

                # ログ出力
                logger.info("", extra={"frame": idx, "batch": bid})

                del img, res
                gc.collect()

                processed += 1
                print(f"\r{processed}/{total_frames} フレーム処理完了", end="", flush=True)

    out.release()
    elapsed = time.time() - start_time

    print(f"\n完了。結果: {OUTPUT_PATH}")
    print(f"処理時間: {elapsed:.2f} 秒")
