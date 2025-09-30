import cv2

def select_resolution():
    print("解像度を選択してください:")
    print("1: HHD (1920x1080)")
    print("2: HD  (1280x720)")
    print("3: VGA (640x480)")
    print("4: 1920x1200")
    choice = input("番号を入力: ")
    if choice == "1":
        return 1920, 1080
    elif choice == "2":
        return 1280, 720
    elif choice == "3":
        return 640, 480
    elif choice == "4":
        return 1920, 1200
    else:
        print("無効な選択です。デフォルトでHHDを使用します。")
        return 1920, 1080

def main():
    # キャプチャーデババイスID（0は最初のキャプチャーボード）
    device_id = 1

    # 解像度選択
    width, height = select_resolution()

    # VideoCapture オブジェクトの生成
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"キャプチャーデバイス {device_id} を開けません")
        return

    # キャプチャ設定（オプション：解像度・FPS）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 設定反映のため数フレーム捨てる
    for _ in range(5):
        cap.read()

    # 最初のフレームで解像度・FPSを取得
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得に失敗しました")
        cap.release()
        return

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"取得した解像度: {width}x{height}")
    print(f"取得したFPS: {fps}")

    # ウィンドウサイズを解像度に合わせる
    window_name = "Capture Board Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Press 'q' to quit")

    while True:
        # フレーム取得
        ret, frame = cap.read()
        if not ret:
            print("フレーム取得に失敗しました")
            break

        # ウィンドウに表示
        cv2.imshow(window_name, frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
