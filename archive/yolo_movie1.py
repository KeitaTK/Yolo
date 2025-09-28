from ultralytics import YOLO
import torch

# CPU利用時にスレッド数を指定（例: 4コア使用）
torch.set_num_threads(10)

# モデルのパスを指定してロード
model = YOLO('yolo11n.pt')

# 使用中のデバイスを表示
print(f'使用デバイス: {model.device}')

# ローカル動画ファイルを指定して推論を実行（全てのフレームを処理）
results = model.predict(source='input_video3.mp4', save=True, conf=0.25)

print('推論完了。結果は保存されています。')
