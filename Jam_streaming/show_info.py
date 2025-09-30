import cv2
from ultralytics import YOLO
import time
import psutil
import threading

class DynamicFPSController:
    def __init__(self, base_fps=5, min_fps=2, max_fps=15):
        self.base_fps = base_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.current_fps = base_fps
        self.cpu_history = []
        self.adjustment_interval = 2.0
        self.last_adjustment = time.time()

    def update_fps(self):
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return self.current_fps
        cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_percents = cpu_percents[:4]
        cpu_percent = sum(cpu_percents) / len(cpu_percents)
        self.cpu_history.append(cpu_percent)
        if len(self.cpu_history) > 5:
            self.cpu_history.pop(0)
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        if avg_cpu > 85:
            self.current_fps = max(self.min_fps, self.current_fps - 2)
        elif avg_cpu < 60:
            self.current_fps = min(self.max_fps, self.current_fps + 1)
        self.last_adjustment = now
        print(f"CPU: {avg_cpu:.1f}%, 推論FPS: {self.current_fps}")
        return self.current_fps

def resize_for_inference(frame, max_size=640):
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return frame

def scale_boxes(boxes, resized_shape, orig_shape):
    # バウンディングボックスを元解像度にスケール
    scaled = []
    resized_h, resized_w = resized_shape
    orig_h, orig_w = orig_shape
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
        cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
        scaled.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
    return scaled

def main():
    device_id = 1
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"キャプチャーデバイス {device_id} を開けません")
        return

    # HD 30fpsに設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(50):
        cap.read()

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"取得した解像度: {width}x{height}, FPS: {fps}")

    model = YOLO("yolo11n.pt")
    print(model.names)  # クラス名一覧を表示
    # クラス名リストは必ずモデルから取得
    class_names = model.names
    window_name = f"YOLO {width}x{height}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_controller = DynamicFPSController(base_fps=5)
    latest_results = []
    lock = threading.Lock()

    # 推論スレッド
    def inference_loop():
        nonlocal latest_results
        last_infer_time = 0
        while running[0]:
            frame_copy = None
            with lock:
                if shared_frame[0] is not None:
                    frame_copy = shared_frame[0].copy()
            if frame_copy is None:
                time.sleep(0.01)
                continue

            target_fps = fps_controller.update_fps()
            infer_interval = 1.0 / target_fps
            now = time.time()
            if now - last_infer_time < infer_interval:
                time.sleep(0.005)
                continue

            resized_frame = resize_for_inference(frame_copy)
            results = model(resized_frame, device="cpu", verbose=False)
            boxes = []
            if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                boxes = scale_boxes(results[0].boxes, resized_frame.shape[:2], (height, width))
            with lock:
                latest_results = boxes
            last_infer_time = now

    shared_frame = [None]
    running = [True]
    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    infer_thread.start()

    # 表示ループ（HD 30FPSで滑らかに）
    display_interval = 1.0 / 30
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレーム取得に失敗しました")
            break

        with lock:
            shared_frame[0] = frame

        annotated = frame.copy()
        with lock:
            boxes = latest_results.copy()
        for x1, y1, x2, y2, conf, cls in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # クラス名でラベル表示（タグ付けされたものを必ず使用）
            class_name = class_names[cls] if cls < len(class_names) else f"Class{cls}"
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow(window_name, annotated)

        # 30FPSで表示
        key = cv2.waitKey(int(display_interval * 1000)) & 0xFF
        if key == ord('q'):
            break

    running[0] = False
    infer_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    running[0] = False
    infer_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
