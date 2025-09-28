import cv2
import os

video_paths = [
    r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\2台上下.mp4",
    r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\Guided.mp4",
    r"C:\Users\Umemoto\Downloads\OneDrive_1_2025-9-28\1台位置制御.mp4"
]
output_dir = 'study1'

os.makedirs(output_dir, exist_ok=True)

frame_global_idx = 0

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)

    for sec in range(duration + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f'frame_{frame_global_idx:05d}.jpg')
            cv2.imwrite(out_path, frame)
            print(f'Saved: {out_path}')
            frame_global_idx += 1
        else:
            break

    cap.release()

print('完了')
