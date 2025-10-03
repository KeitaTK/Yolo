import torch
from ultralytics import YOLO
import os
import time
from datetime import datetime

def get_optimal_settings():
    """環境に応じた最適設定を自動決定"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory_gb >= 12:
            batch_size = 32
            workers = 8
        elif gpu_memory_gb >= 8:
            batch_size = 24
            workers = 6
        elif gpu_memory_gb >= 6:
            batch_size = 16
            workers = 4
        else:
            batch_size = 12
            workers = 2
        
        gpu_capability = torch.cuda.get_device_capability(0)
        use_amp = gpu_capability[0] >= 7
        
        print(f"🚀 GPU: {gpu_name} ({gpu_memory_gb:.1f}GB) | Batch: {batch_size} | AMP: {'ON' if use_amp else 'OFF'}")
        return device, batch_size, workers, use_amp
        
    else:
        device = 'cpu'
        cpu_count = os.cpu_count()
        
        if cpu_count >= 16:
            batch_size = 16
            workers = min(8, cpu_count // 2)
        elif cpu_count >= 12:
            batch_size = 12
            workers = min(6, cpu_count // 2)
        else:
            batch_size = 8
            workers = min(4, cpu_count // 2)
        
        print(f"💻 CPU: {cpu_count} cores | Batch: {batch_size}")
        return device, batch_size, workers, False

class SimpleTimer:
    """シンプル学習時間計測"""
    def __init__(self):
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        print(f"⏱️ Started: {datetime.now().strftime('%H:%M:%S')}")
        
    def finish(self):
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print(f"⏱️ Finished: {datetime.now().strftime('%H:%M:%S')} | Duration: {minutes:02d}:{seconds:02d}")
            return total_time
        return 0

def main():
    timer = SimpleTimer()
    
    # 環境設定
    device, batch_size, workers, use_amp = get_optimal_settings()
    
    # GPU最適化
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
    
    # モデルロード
    model = YOLO('yolo11s.pt')
    print(f"📦 Model: YOLO11s loaded")

    # データセット確認
    data_path = 'Custom_training/Annotation/quadcopter/data.yaml'
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return
    
    print(f"📊 Dataset: {data_path}")
    
    # 学習設定
    epochs = 30
    training_config = {
        'data': data_path,
        'epochs': epochs,
        'imgsz': 640,
        'batch': batch_size,
        'device': device,
        'workers': workers,
        'project': 'runs/train',
        'name': f'quad_nano_{"gpu" if device=="cuda" else "cpu"}_{batch_size}b_{epochs}ep_{datetime.now().strftime("%m%d_%H%M")}',
        'save_period': 10,
        'patience': 100,
        'verbose': False,  # YOLOの詳細ログを非表示
        'plots': True,
        'cache': 'ram' if device == 'cuda' else False,
    }
    
    if device == 'cuda':
        training_config.update({
            'amp': use_amp,
            'close_mosaic': 10,
            'copy_paste': 0.1,
            'mixup': 0.1,
        })
    
    print(f"🏋️ Training: {epochs} epochs | Batch: {batch_size} | Device: {device}")
    
    try:
        timer.start()
        
        # 学習実行
        results = model.train(**training_config)
        
        # 学習完了
        total_time = timer.finish()
        
        # 結果表示
        print(f"✅ Training completed successfully")
        
        # 性能サマリー
        epoch_time = total_time / epochs
        print(f"📈 Performance: {epoch_time:.1f}s/epoch | {epochs/total_time*3600:.0f} epochs/hour")
        
        # メトリクス表示
        try:
            if hasattr(results, 'metrics') and hasattr(results.metrics, 'box'):
                metrics = results.metrics.box
                print(f"📊 Results: mAP50={metrics.map50:.3f} | mAP50-95={metrics.map:.3f}")
        except:
            pass
        
        # ベストモデル
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            file_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
            print(f"🏆 Best model: {file_size_mb:.1f}MB | {best_model_path}")
        
        print(f"📁 Results: {results.save_dir}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")
        timer.finish()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ GPU OOM! Try batch size: {batch_size//2}")
        else:
            print(f"❌ Error: {e}")
        timer.finish()
    except Exception as e:
        print(f"❌ Error: {e}")
        timer.finish()
    finally:
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
