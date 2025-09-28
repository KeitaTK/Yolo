from ultralytics import YOLO
import torch
import os
import time
import cv2

# 入力動画パス
input_path = r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4"

# 出力先ディレクトリを入力動画と同じ場所に設定
output_dir = os.path.dirname(input_path)

# スレッド数を最大数-2で設定
num_threads = max(1, os.cpu_count() - 2)
torch.set_num_threads(num_threads)

# モデルのパスを指定してロード
model = YOLO('yolo11n_quadcopter.pt')

# 使用中のデバイスを表示
print(f'使用デバイス: {model.device}')
print(f'スレッド数: {num_threads}')

# 動画の長さを取得
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("動画ファイルを開けませんでした。")
    exit(1)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps if fps > 0 else 0
print(f'動画の長さ: {duration:.2f} 秒')
cap.release()

# 推論処理の時間計測
start_time = time.time()
results = model.predict(
    source=input_path,
    save=True,
    project=output_dir,
    name="runs",
    conf=0.25
)
end_time = time.time()
elapsed = end_time - start_time
print(f'推論完了。結果は {output_dir} に保存されています。')
print(f'処理時間: {elapsed:.2f} 秒')
