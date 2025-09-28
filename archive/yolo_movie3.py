from ultralytics import YOLO
import torch
import os
import time
import cv2
import concurrent.futures
import gc

# 入力動画パス
input_path = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"
output_dir = os.path.dirname(input_path)
output_video_path = os.path.join(output_dir, "output_detected.mp4")

# スレッド数を最大数-2で設定
num_threads = max(1, os.cpu_count() - 2)
torch.set_num_threads(num_threads)

# モデルのパスを指定してロード
model = YOLO('yolo11n_quadcopter.pt')

# 使用中のデバイスを表示
print(f'使用デバイス: {model.device}')
print(f'スレッド数: {num_threads}')

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("動画ファイルを開けませんでした。")
    exit(1)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps if fps > 0 else 0
print(f'動画の長さ: {duration:.2f} 秒')
print(f'フレーム数: {frame_count}')
print(f'解像度: {width}x{height}')

# 出力動画の準備
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def process_batch(batch_indices, batch_frames):
    from ultralytics import YOLO  # 各プロセスでimport
    local_model = YOLO('yolo11n_quadcopter.pt')
    results_list = []
    for idx, frame in zip(batch_indices, batch_frames):
        results = local_model.predict(frame, conf=0.25, verbose=False)
        result_img = results[0].plot()
        results_list.append((idx, result_img))
    # バッチ内で何フレーム処理したかも返す
    return results_list, len(batch_indices)

BATCH_SIZE = 10  # 必要に応じて調整

if __name__ == "__main__":
    start_time = time.time()
    idx = 0
    processed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        while True:
            batch_indices = []
            batch_frames = []
            for _ in range(BATCH_SIZE):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_indices.append(idx)
                batch_frames.append(frame)
                idx += 1
            if not batch_indices:
                break
            futures.append(executor.submit(process_batch, batch_indices, batch_frames))
        cap.release()

        # 結果を順次取得・保存
        for future in futures:
            results, batch_processed = future.result()
            for i, result_img in sorted(results):
                out.write(result_img)
            processed_count += batch_processed
            print(f"{processed_count}/{frame_count} フレーム処理完了", end='\r', flush=True)
            del results
            gc.collect()
    out.release()

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"{processed_count}/{frame_count} フレーム処理完了")
    print(f'推論完了。結果は {output_video_path} に保存されています。')
    print(f'処理時間: {elapsed:.2f} 秒')
