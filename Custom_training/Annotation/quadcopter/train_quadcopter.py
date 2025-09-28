from ultralytics import YOLO

def main():
    # Nanoモデルをロード（自動ダウンロード）
    model = YOLO('yolo11n.pt')

    # 学習実行
    model.train(
        data='data.yaml',          # データ設定ファイル
        epochs=30,                 # エポック数
        imgsz=640,                 # 短辺を640pxにリサイズ
        batch=16,                  # バッチサイズ（CPU: Core i7-13700KF, RAM 48GB）
        device='cpu',              # CPU学習
        project='runs/train',      # 出力先フォルダ
        name='quad_nano_cpu_30ep'  # 実験名
    )

if __name__ == '__main__':
    main()
