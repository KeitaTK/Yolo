import os
import time
import cv2
import gc
import torch
from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor, as_completed

# モデルパスと動画パスは元のまま
MODEL_PATH = r"yolo11n_quadcopter.pt"
INPUT_PATH = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_detected.mp4")

# CPUコア数に応じたスレッド／プロセス数設定
NUM_CORES = os.cpu_count() or 1
THREADS_PER_WORKER = max(1, NUM_CORES // 2)
MAX_WORKERS = max(1, NUM_CORES // THREADS_PER_WORKER)
torch.set_num_threads(THREADS_PER_WORKER)

# ワーカー初期化：各プロセスで一度だけモデルを読み込む
def init_worker(model_path):
    global model
    model = YOLO(model_path)

# 各フレームを処理する関数
def process_frame(idx, frame, conf=0.25):
    results = model.predict(frame, conf=conf, verbose=False)
    img = results[0].plot()
    return idx, img

if __name__ == "__main__":
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("動画ファイルを開けませんでした。")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    start_time = time.time()
    processed = 0

    # 同時キューに積む最大タスク数
    max_queue = MAX_WORKERS * 2

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
        initargs=(MODEL_PATH,)
    ) as executor:
        futures = {}
        idx = 0

        while True:
            # タスク投入フェーズ
            while len(futures) < max_queue:
                ret, frame = cap.read()
                if not ret:
                    break
                fut = executor.submit(process_frame, idx, frame)
                futures[fut] = idx
                idx += 1

            # すべて処理済みならループ終了
            if not futures:
                break

            # 完了したタスクを１つ回収
            for fut in as_completed(futures):
                i, img = fut.result()
                out.write(img)
                processed += 1

                # メモリ解放
                del img
                gc.collect()

                print(f"\r{processed}/{total_frames} フレーム処理完了", end='', flush=True)

                # 回収済みの Future を削除して再度投入へ
                del futures[fut]
                break

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f"\n推論完了。結果は {OUTPUT_PATH} に保存されています。")
    print(f"処理時間: {elapsed:.2f} 秒")
