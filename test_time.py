import cv2
import mediapipe as mp
import time

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# グーの判定
def is_fist(landmarks):
    tips = [4, 8, 12, 16, 20]  # 親指、人差し指、中指、薬指、小指の先端
    for tip in tips[1:]:  # 親指を除く指
        if landmarks[tip].y < landmarks[tip - 2].y:  # 指が曲がっていないとグーではない
            return False
    return True

# チョキの判定 (人差し指と中指が伸びている状態)
def is_scissors(landmarks):
    tips = [8, 12]  # 人差し指と中指
    for tip in tips:
        if landmarks[tip].y > landmarks[tip - 2].y:  # どちらかが曲がっているとチョキではない
            return False
    # 薬指と小指が曲がっているかを確認
    if landmarks[16].y < landmarks[14].y or landmarks[20].y < landmarks[18].y:
        return False
    return True

# パーの判定 (全ての指が伸びている状態)
def is_palm(landmarks):
    tips = [4, 8, 12, 16, 20]  # 親指、人差し指、中指、薬指、小指
    for tip in tips:
        if landmarks[tip].y > landmarks[tip - 2].y:  # どれかの指が曲がっているとパーではない
            return False
    return True

cap = cv2.VideoCapture(0)  # カメラ映像をキャプチャ

# 各ジェスチャーの計測時間と最後に認識されたジェスチャー
gesture_start_time = None
gesture_last = None
total_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR画像をRGBに変換
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 手のランドマークを検出
        results = hands.process(image)

        # 画像を再び書き込み可能にして、BGRに戻す
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture_detected = None  # 認識されたジェスチャー

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # グー、チョキ、パーの判定
                if is_fist(hand_landmarks.landmark):
                    gesture_detected = 'Fist'
                elif is_scissors(hand_landmarks.landmark):
                    gesture_detected = 'Scissors'
                elif is_palm(hand_landmarks.landmark):
                    gesture_detected = 'Palm'

                # 前回のジェスチャーと異なる場合、トータル時間をリセット
                if gesture_detected != gesture_last:
                    total_time = 0
                    gesture_start_time = time.time()
                    gesture_last = gesture_detected
                else:
                    # 同じジェスチャーの場合、gesture_start_timeがNoneでないことを確認してから経過時間を計算
                    if gesture_start_time is None:
                        gesture_start_time = time.time()  # ジェスチャーが初めて認識された場合に現在の時刻を設定
                    total_time = time.time() - gesture_start_time

        # トータルの時間を表示
        cv2.putText(image, f'Total {gesture_last} Time: {total_time:.2f}s', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if gesture_detected:
            cv2.putText(image, f'Gesture: {gesture_detected}', 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 画面に出力
        cv2.imshow('Gesture Detection (Fist, Scissors, Palm)', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
