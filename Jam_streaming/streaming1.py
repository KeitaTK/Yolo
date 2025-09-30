# YOLO Live Streaming with Dynamic FPS Control
# キャプチャーボード → YOLO推論 → バウンディングボックス描画 → OBS仮想カメラ配信

import cv2
import numpy as np
import threading
import time
import queue
import pyvirtualcam
import psutil
from ultralytics import YOLO
from typing import Optional, Tuple
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicFPSController:
    """CPU使用率に基づいて推論FPSを動的に調整するクラス"""
    
    def __init__(self, base_fps: int = 10, min_fps: int = 5, max_fps: int = 20):
        self.base_fps = base_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.current_fps = base_fps
        self.cpu_history = []
        self.adjustment_interval = 2.0  # 2秒ごとに調整
        self.last_adjustment = time.time()
    
    def update_fps(self) -> int:
        """CPU使用率に基づいてFPSを調整"""
        current_time = time.time()
        
        if current_time - self.last_adjustment < self.adjustment_interval:
            return self.current_fps
            
        # CPU使用率を取得（4コア使用を想定）
        cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
        # 4コア分だけ使う（それ以上あっても4つまで）
        cpu_percents = cpu_percents[:4]
        cpu_percent = sum(cpu_percents) / len(cpu_percents)
        self.cpu_history.append(cpu_percent)
        
        # 過去5回の平均CPU使用率で判定
        if len(self.cpu_history) > 5:
            self.cpu_history.pop(0)
        
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        
        # CPU使用率に基づいてFPS調整
        if avg_cpu > 85:  # 高負荷時はFPSを下げる
            self.current_fps = max(self.min_fps, self.current_fps - 2)
        elif avg_cpu < 60:  # 低負荷時はFPSを上げる
            self.current_fps = min(self.max_fps, self.current_fps + 1)
        
        self.last_adjustment = current_time
        logger.info(f"CPU: {avg_cpu:.1f}%, Inference FPS: {self.current_fps}")
        
        return self.current_fps

class FrameCapture:
    """キャプチャーボードからのフレーム取得を担当するクラス"""
    
    def __init__(self, device_id: int = 0, buffer_size: int = 5):
        self.device_id = device_id
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.capture: Optional[cv2.VideoCapture] = None  # 型ヒントを追加
        self.running = False
        self.thread = None
        
        # カメラ初期化
        self._init_camera()
    
    def _init_camera(self):
        """キャプチャーボードの初期化"""
        self.capture = cv2.VideoCapture(self.device_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"キャプチャーボード（デバイス{self.device_id}）を開けません")
        
        # キャプチャー設定（30fps, 1920x1080）
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # バッファサイズを最小に
        
        # 実際の設定を確認
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"キャプチャー設定: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    def start(self):
        """フレーム取得スレッドを開始"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("フレーム取得スレッド開始")
    
    def stop(self):
        """フレーム取得スレッドを停止"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.capture:
            self.capture.release()
        logger.info("フレーム取得スレッド停止")
    
    def _capture_loop(self):
        """フレーム取得のメインループ"""
        if self.capture is None:
            logger.error("captureが初期化されていません")
            return
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                logger.warning("フレーム取得失敗")
                continue
            
            timestamp = time.time()
            
            # キューが満杯の場合、古いフレームを削除
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # 新しいフレームを追加
            try:
                self.frame_queue.put((timestamp, frame), block=False)
            except queue.Full:
                pass  # キューが満杯の場合は無視
            
            frame_count += 1
            
            # 1秒ごとにFPS計算してログ出力
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                logger.debug(f"キャプチャーFPS: {actual_fps:.1f}")
    
    def get_latest_frame(self) -> Optional[Tuple[float, np.ndarray]]:
        """最新のフレームを取得"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

class YOLOInferenceEngine:
    """YOLO推論エンジン"""
    
    def __init__(self, model_path: str = "yolo11n.pt", device: str = "cpu"):
        self.model = YOLO(model_path)
        self.device = device
        self.latest_results = None
        self.inference_count = 0
        self.total_inference_time = 0
        
        logger.info(f"YOLOモデル読み込み完了: {model_path} on {device}")
    
    def predict(self, frame: np.ndarray, confidence: float = 0.5) -> Optional[object]:
        """YOLO推論を実行"""
        start_time = time.time()
        
        try:
            # 推論実行（CPUで実行）
            results = self.model(frame, device=self.device, verbose=False, conf=confidence)
            
            if results and len(results) > 0:
                self.latest_results = results[0]
            
            # 推論時間統計
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            if self.inference_count % 10 == 0:
                avg_time = self.total_inference_time / self.inference_count
                logger.debug(f"平均推論時間: {avg_time*1000:.1f}ms")
            
            return self.latest_results
            
        except Exception as e:
            logger.error(f"推論エラー: {e}")
            return self.latest_results
    
    def get_latest_results(self):
        """最新の推論結果を取得"""
        return self.latest_results

class YOLOLiveStreamer:
    """メインのライブ配信クラス"""
    
    def __init__(self, 
                 capture_device: int = 0,
                 model_path: str = "yolo11n.pt",
                 inference_fps: int = 10,
                 output_fps: int = 30,
                 output_width: int = 1920,
                 output_height: int = 1200):
        
        self.output_width = output_width
        self.output_height = output_height
        self.output_fps = output_fps
        self.running = False
        
        # コンポーネント初期化
        self.frame_capture = FrameCapture(capture_device)
        self.yolo_engine = YOLOInferenceEngine(model_path, device="cpu")
        self.fps_controller = DynamicFPSController(base_fps=inference_fps)
        
        # 推論制御用変数
        self.last_inference_time = 0
        self.inference_thread = None
        self.streaming_thread = None
        
        # 過去のバウンディングボックスを保持
        self.past_boxes = []
        self.max_past = 30  # 30フレーム分保持
    
    def _resize_for_inference(self, frame: np.ndarray, max_size: int = 640) -> np.ndarray:
        """推論用にフレームをリサイズ（画質を落とす）"""
        h, w = frame.shape[:2]
        
        # 長辺をmax_sizeに合わせてリサイズ
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def _resize_to_original(self, boxes, original_shape: Tuple[int, int], 
                          resized_shape: Tuple[int, int]) -> list:
        """バウンディングボックスを元の解像度に戻す"""
        if boxes is None:
            return []
        
        orig_h, orig_w = original_shape[:2]
        resized_h, resized_w = resized_shape[:2]
        
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h
        
        scaled_boxes = []
        for box in boxes:
            if hasattr(box, 'xyxy'):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                scaled_box = {
                    'xyxy': [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                    'conf': float(box.conf[0]) if hasattr(box, 'conf') else 0.0,
                    'cls': int(box.cls[0]) if hasattr(box, 'cls') else 0
                }
                scaled_boxes.append(scaled_box)
        
        return scaled_boxes
    
    def _draw_boxes(self, frame: np.ndarray, boxes: list) -> np.ndarray:
        """バウンディングボックスを描画"""
        if not boxes:
            return frame
        
        annotated_frame = frame.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box['xyxy'])
            conf = box['conf']
            cls = box['cls']
            
            # バウンディングボックス描画
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ラベル描画
            label = f"Class{cls}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def _inference_loop(self):
        """推論専用スレッド"""
        logger.info("推論スレッド開始")
        
        while self.running:
            current_time = time.time()
            target_fps = self.fps_controller.update_fps()
            inference_interval = 1.0 / target_fps
            
            if current_time - self.last_inference_time >= inference_interval:
                # 最新フレームを取得
                frame_data = self.frame_capture.get_latest_frame()
                if frame_data:
                    timestamp, frame = frame_data
                    
                    # 推論用にリサイズ
                    resized_frame = self._resize_for_inference(frame)
                    
                    # YOLO推論実行
                    self.yolo_engine.predict(resized_frame)
                    self.last_inference_time = current_time
            
            # 短時間スリープ
            time.sleep(0.01)
    
    def _streaming_loop(self):
        """配信専用スレッド"""
        logger.info("配信スレッド開始")
        
        try:
            with pyvirtualcam.Camera(width=self.output_width, 
                                   height=self.output_height, 
                                   fps=self.output_fps) as cam:
                
                logger.info(f"仮想カメラ開始: {cam.device}")
                
                frame_interval = 1.0 / self.output_fps
                last_frame_time = time.time()
                
                while self.running:
                    current_time = time.time()
                    
                    if current_time - last_frame_time >= frame_interval:
                        frame_data = self.frame_capture.get_latest_frame()
                        if not frame_data:
                            time.sleep(0.005)
                            continue
                        timestamp, frame = frame_data
                        if frame is None or frame.shape is None:
                            time.sleep(0.005)
                            continue
                        
                        results = self.yolo_engine.get_latest_results()
                        boxes_to_add = []
                        if results and hasattr(results, 'boxes') and results.boxes is not None:
                            resized_frame = self._resize_for_inference(frame)
                            if resized_frame is None or resized_frame.shape is None:
                                time.sleep(0.005)
                                continue
                            scaled_boxes = self._resize_to_original(
                                results.boxes, frame.shape, resized_frame.shape
                            )
                            boxes_to_add = scaled_boxes

                        # 過去のバウンディングボックスを更新
                        if boxes_to_add:
                            self.past_boxes.append(boxes_to_add)
                            if len(self.past_boxes) > self.max_past:
                                self.past_boxes.pop(0)

                        # 過去分も含めて全て描画
                        annotated_frame = frame.copy()
                        for boxes in self.past_boxes:
                            annotated_frame = self._draw_boxes(annotated_frame, boxes)
                        
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        cam.send(rgb_frame)
                        last_frame_time = current_time
                    
                    cam.sleep_until_next_frame()
                        
        except Exception as e:
            logger.error(f"配信エラー: {e}")
    
    def start(self):
        """ライブ配信開始"""
        logger.info("YOLO Live Streaming 開始...")
        
        self.running = True
        
        # フレームキャプチャ開始
        self.frame_capture.start()
        
        # 推論スレッド開始
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        # 配信スレッド開始
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()
        
        logger.info("全スレッド開始完了")
    
    def stop(self):
        """ライブ配信停止"""
        logger.info("YOLO Live Streaming 停止中...")
        
        self.running = False
        
        # スレッド終了待ち
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2.0)
        
        # フレームキャプチャ停止
        self.frame_capture.stop()
        
        logger.info("YOLO Live Streaming 停止完了")

def main():
    """メイン関数"""
    # 設定
    CAPTURE_DEVICE = 0  # キャプチャーボードのデバイス番号
    MODEL_PATH = "yolo11n.pt"  # YOLOモデルパス
    INFERENCE_FPS = 10  # 初期推論FPS
    OUTPUT_FPS = 30  # 配信FPS
    OUTPUT_WIDTH = 1920  # 配信解像度幅
    OUTPUT_HEIGHT = 1200  # 配信解像度高さ ← 1080→1200に変更

    # ライブストリーマー初期化
    streamer = YOLOLiveStreamer(
        capture_device=CAPTURE_DEVICE,
        model_path=MODEL_PATH,
        inference_fps=INFERENCE_FPS,
        output_fps=OUTPUT_FPS,
        output_width=OUTPUT_WIDTH,
        output_height=OUTPUT_HEIGHT
    )
    
    try:
        # 配信開始
        streamer.start()
        
        logger.info("配信中... Ctrl+Cで停止")
        logger.info("OBSで仮想カメラソースを追加してください")
        
        # メインスレッドでキー入力待ち
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("停止シグナル受信")
    
    finally:
        # 配信停止
        streamer.stop()

if __name__ == "__main__":
    main()