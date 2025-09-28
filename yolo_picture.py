from ultralytics import YOLO
import torch
import os

# CPUのコア数を取得し、2引いた値をスレッド数に設定（最低1）
cpu_count = os.cpu_count()
num_threads = max(1, cpu_count - 2)
torch.set_num_threads(num_threads)

# デバイスは必ずCPU
device = 'cpu'

# モデルのパスを指定してロード
# model = YOLO('yolo11n.pt')
model = YOLO("yolo11s.pt")
model.to(device)

# 使用中のデバイスとスレッド数を表示
print(f'使用デバイス: {model.device}')
print(f'使用スレッド数: {num_threads}')

# 画像ファイルを指定して推論を実行
# results = model.predict(source='Test1.jpg', save=True, conf=0.25)
results = model.predict(source='frame_00036.jpg', save=True, conf=0.25)

print('推論完了。結果は保存されています。')
