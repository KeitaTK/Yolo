import cv2
import time
from ultralytics import YOLO
import psutil  # 追加

def main():
    device_id = 1
    width, height, display_fps, inference_fps = 1920, 1200, 30, 10

    # VideoCapture初期化
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"キャプチャーデバイス {device_id} を開けません")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, display_fps)

    # 設定反映のため数フレーム捨てる
    for _ in range(5):
        cap.read()

    # YOLOモデル読み込み
    model = YOLO("yolo11n.pt")

    # 推論タイミング管理
    last_infer_time = 0
    infer_interval = 1.0 / inference_fps
    latest_results = None

    window_name = "YOLO Streaming 1920x1200"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"取得した解像度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"取得したFPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレーム取得に失敗しました")
            break

        now = time.time()
        # 推論は10fpsで実施
        if now - last_infer_time >= infer_interval:
            results = model(frame, device="cpu", verbose=False)
            latest_results = results[0] if results and len(results) > 0 else None
            last_infer_time = now

            # 4コア分のCPU使用率を表示
            cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_percents = cpu_percents[:4]
            avg_cpu = sum(cpu_percents) / len(cpu_percents)
            print(f"4コア平均CPU使用率: {avg_cpu:.1f}%")

        # バウンディングボックス描画
        annotated = frame.copy()
        if latest_results is not None and hasattr(latest_results, "boxes") and latest_results.boxes is not None:
            for box in latest_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class{cls}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow(window_name, annotated)

        # 30fpsで表示
        if cv2.waitKey(int(1000 / display_fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
