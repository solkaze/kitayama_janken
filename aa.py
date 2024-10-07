import cv2
import mediapipe as mp
import time
import random

# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# カメラキャプチャを初期化
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラを表します

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipeで手を検出
        results = hands.process(frame_rgb)

        # 検出した手の情報を取得
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # 各指の位置を取得
                flm = landmarks.landmark
                
                # 修正点1　べつに人差し指と中指を立たせていなくても判定されてしまう

                finger_count = 0
                # 人差し指が立っている
                if flm[8].y < flm[7].y < flm[6].y < flm[0].y:
                    finger_count += 1
                # 中指が立っている場合
                if flm[12].y < flm[11].y < flm[10].y < flm[0].y:
                    finger_count += 1
                # 薬指が立っている場合
                if flm[16].y < flm[15].y < flm[14].y < flm[0].y:
                    finger_count += 1
                # 小指が立っている場合
                if flm[20].y < flm[19].y < flm[18].y < flm[0].y:
                    finger_count += 1

                # 親指が立っている場合
                if (flm[4].x < flm[3].x < flm[5].x < flm[17].x or
                    flm[4].x > flm[3].x > flm[5].x > flm[17].x):
                    finger_count += 1

                # プレイヤーの手を判定
                if finger_count == 0:
                    user_hands = "guu"
                elif finger_count == 2:
                    user_hands = "tyoki"
                elif finger_count == 5:
                    user_hands = "pa"
                else:
                    user_hands = "humei"

                cv2.putText(frame, f"user_hands: {user_hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                # 結果を画面に表示
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # スタート
        if cv2.waitKey(1) & 0xFF == ord('m'):
            # cv2.putText(frame, f"Jyanken", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # AIの手をランダムに選択
            ai_hands = random.choice(["guu", "tyoki", "pa"])

            # time.sleep(3)

            # ゲームの結果を判定
            if user_hands != "humei":
                if user_hands == ai_hands:
                    result = "Draw"
                elif (
                    (user_hands == "guu" and ai_hands == "tyoki")
                    or (user_hands == "tyoki" and ai_hands == "pa")
                    or (user_hands == "pa" and ai_hands == "guu")
                ):
                    result = "You Win"
                else:
                    result = "AI Wins"


                cv2.putText(frame, f"AI hands: {ai_hands}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Result: {result}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # time.sleep(3)

        # フレームを表示
        cv2.imshow('Hand Count', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
