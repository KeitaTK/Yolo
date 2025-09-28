import os
import cv2
import gc
import torch
import subprocess
import shlex
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count, Manager

# 入力・出力パス（元のまま）
# INPUT_PATH  = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
INPUT_PATH  = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\Guided.mp4"
OUTPUT_PATH = os.path.join(os.path.dirname(INPUT_PATH), "output_with_boxes.mp4")
MODEL_PATH  = r"yolo11n_quadcopter.pt"

# バッチサイズとプロセス数
BATCH_SIZE    = 8
CPU_CORES     = cpu_count()
THREADS_PER_P = max(1, CPU_CORES // 2)
NUM_PROCESSES = max(1, CPU_CORES // THREADS_PER_P)

def init_worker(model_path):
    torch.set_num_threads(THREADS_PER_P)
    global model
    model = YOLO(model_path)

def infer_batch(args):
    """
    batch_frames: [(idx, frame), ...]
    戻り: [(idx, [ (x1,y1,x2,y2,cls), ... ]), ...]
    """
    (batch_frames,) = args
    idxs, frames = zip(*batch_frames)
    # 必要ならリサイズ（ここでは元解像度をそのまま使用）
    # resized = [cv2.resize(f, (1280,720)) for f in frames]
    results = model.predict(frames, conf=0.25, verbose=False)
    out = []
    for idx, res in zip(idxs, results):
        dets = []
        for *xy, _, cls in res.boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, xy)
            dets.append((x1, y1, x2, y2, int(cls)))
        out.append((idx, dets))
    del frames, results
    gc.collect()
    return out

def main():
    # 元動画プロパティ取得
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print("動画を開けませんでした。")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # バッチ単位でフレーム収集
    cap = cv2.VideoCapture(INPUT_PATH)
    batches, batch = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append((idx, frame))
        idx += 1
        if len(batch) == BATCH_SIZE:
            batches.append((batch.copy(),))
            batch.clear()
    if batch:
        batches.append((batch.copy(),))
    cap.release()

    # 共有辞書にバウンディングボックス保存
    manager    = Manager()
    boxes_dict = manager.dict()

    # プロセスプールでバッチ推論
    pool = Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(MODEL_PATH,))
    for batch_out in pool.imap_unordered(infer_batch, batches):
        for idx, dets in batch_out:
            boxes_dict[idx] = dets
        del batch_out
        gc.collect()
    pool.close()
    pool.join()

    # 推論完了後に GPU エンコードで書き出し
    # ffmpeg NVENC コマンド
    ffmpeg_cmd = (
        f"ffmpeg -y "
        f"-f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - "
        f"-c:v h264_nvenc -preset p7 -rc vbr_hq -cq 19 "
        f"-pix_fmt yuv420p \"{OUTPUT_PATH}\""
    )
    proc = subprocess.Popen(shlex.split(ffmpeg_cmd), stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(INPUT_PATH)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = boxes_dict.get(idx, [])
        for x1, y1, x2, y2, cls in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        proc.stdin.write(frame.tobytes())
        idx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()

    print(f"完了: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
