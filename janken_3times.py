import cv2
import mediapipe as mp
import time
import random
import numpy as np
import tkinter as tk  # Tkinterをインポート
import queue
from PIL import Image, ImageTk
from threading import Thread

#多層ニューラルネットワーク
#from sklearn.neural_network import MLPClassifier
#単純パーセプトロン
from sklearn.linear_model import Perceptron
#テストコメントアウト
# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# カメラキャプチャを初期化
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラを表します

# カウンタダウンを表示するTkinterウィンドウを初期化
root = tk.Tk()
root.title("じゃんけん")

# ウィンドウのサイズを設定
root.geometry("699x400")  # 幅x高さ

countdown_label = tk.Label(root, font=("Helvetica", 48))
countdown_label.pack()

result_label = tk.Label(root, font=("Helvetica", 24))
result_label.pack()

image_guu = Image.open('./ml-images/human_gu.png')
photo_guu = ImageTk.PhotoImage(image_guu)
image_label0 = tk.Label(root, image=photo_guu)
image_label0.pack(side="left")
# image_label0.config(image=photo_guu)

image_label1 = tk.Label(root, image=photo_guu)
image_label1.pack(side="right")

left_label = tk.Label(root, text="AI")
left_label.pack(side='left')

right_label = tk.Label(root, text="あなた")
right_label.pack(side='right')

root.update()

#じゃんけんの手のベクトル形式を格納した配列
#グー[1,0,0], チョキ[0,1,0], パー[0,0,1]
janken_array = np.array([[1,0,0], [0,1,0], [0,0,1]])

#グー, チョキ, パーの名称を格納した配列
janken_class = ['グー', 'チョキ', 'パー']

#過去何回分の手を覚えているか
n = 30

#じゃんけんの過去の手の初期化
#人間の手とコンピュータの手をそれぞれn回分.(1回分につき3個の数字が必要)
Jprev = np.zeros(3*n*2)

#過去の手をグー、チョキ、パーが同じ数になるように初期化
for i in range(2*n):
    j = i % 3
    Jprev[3*i:3*i+3] = janken_array[j]

#現在の手(0~2の整数)をランダムに初期化
j = np.random.randint(0,3)

#過去の手(入力データ)をscikit_learn用の配列に変換
Jprev_set = np.array([Jprev])
#現在の手(ターゲット)をscikit_learn用の配列に変換
jnow_set = np.array([j])

#三層ニューラルネットワークを定義
#clf = MLPClassifier(hidden_layer_sizes=(100, ), random_state=None)
#単純パーセプトロンを定義
clf = Perceptron(random_state=None)

#ランダムな入力でオンライン学習を1回行う
#初回の学習では、あり得るターゲット(0,1,2)を分類器に知らせる必要がある
clf.partial_fit(Jprev_set, jnow_set, classes=[0,1,2])

def display_image(image_path_ai,image_path_user):
    
    # 画像を読み込み、Tkinterラベルに表示する
    #ai
    image0 = Image.open(image_path_ai)
    photo0 = ImageTk.PhotoImage(image0)
    
    # image_label0 = tk.Label(root, image=photo0)
    # image_label0.image = photo0
    image_label0.config(image=photo0)
    # image_label0.update()
    # root.update()
    # image_label0.pack(side="left")


    #user
    image1 = Image.open(image_path_user)
    photo1 = ImageTk.PhotoImage(image1)
    
    # image_label1 = tk.Label(root, image=photo1)
    image_label1.config(image=photo1)
    # image_label1.image = photo1
    # root.update()
    # image_label1.pack(side="right")
    root.update()
    time.sleep(1)

# 勝敗を判定する関数
def determine_winner(user_hands, ai_hands):
    if user_hands ==  "humei":
        return "Booooooooooo"

    if user_hands == ai_hands:
        return "Draw"
    elif (
        (user_hands == "guu" and ai_hands == "tyoki")
        or (user_hands == "tyoki" and ai_hands == "pa")
        or (user_hands == "pa" and ai_hands == "guu")
    ):
        return "You Win"
    else:
        return "AI Wins"
    

# じゃんけんの結果を表示
# じゃんけんの手を表示
def display_janken_result(result):
    # result_label = tk.Label(root, text=result, font=("Helvetica", 24))
    # root.update()
    # result_label.pack()
    result_label.config(text=result)
    root.update()

def countdown(user_hands,ai_hands):
    image_path_ai = "null"
    image_path_user = "null"

    print(user_hands)
    print(ai_hands)

    #for i in range(3, 0, -1):
        #countdown_label.config(text=str(i))
        #root.update()
        #time.sleep(1)
    countdown_label.config(text="じゃん!")
    root.update()
    time.sleep(1)
    countdown_label.config(text="けん!!")
    root.update()
    time.sleep(1)
    countdown_label.config(text="ぽんっ!!!")
    root.update()
    time.sleep(1) 

    
    if user_hands == "humei":
        # root.destroy()
        # AI
        if ai_hands == "guu":   #グ-のとき
            image_path_ai = './ml-images/human_gu.png'  # 画像ファイルのパスを実際のファイルパスに変更
        if ai_hands == "tyoki":   #チョキのとき
            image_path_ai = './ml-images/human_choki.png'
        if ai_hands == "pa":  #パーのとき
            image_path_ai = './ml-images/human_pa.png'
        ## AI 不明画像を出力したいとき
        # image_path_ai = './ml-images/hatena.png'
        #######

        # User
        image_path_user = './ml-images/hatena.png'
        # root.update()
        print()

    if user_hands != "humei":
        # root.update()

        # AIの手を表示する 左側
        if ai_hands == "guu":   #グ-のとき
            image_path_ai = './ml-images/human_gu.png'  # 画像ファイルのパスを実際のファイルパスに変更
        if ai_hands == "tyoki":   #チョキのとき
            image_path_ai = './ml-images/human_choki.png'
        if ai_hands == "pa":  #パーのとき
            image_path_ai = './ml-images/human_pa.png'


        # ユーザの手を表示する 右側
        if user_hands == "guu":   #グ-のとき
            image_path_user = './ml-images/human_gu.png'  # 画像ファイルのパスを実際のファイルパスに変更
        if user_hands == "tyoki":   #チョキのとき
            image_path_user = './ml-images/human_choki.png'
        if user_hands == "pa":  #パーのとき
            image_path_user = './ml-images/human_pa.png'
            print("a")
        
    display_image(image_path_ai,image_path_user)

    #じゃんけんの勝敗
    # left_label = tk.Label(root, text="AI")
    # left_label.pack(side='left')

    # right_label = tk.Label(root, text="あなた")
    # right_label.pack(side='right')

    result = determine_winner(user_hands, ai_hands)
    display_janken_result(result)  # じゃんけんの結果を表示

    root.update()

    # root.update()

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

def player_hands_thinking(q):

    #idと手をセットでプッシュする
    result = (1, user_hands)

    # 決定した手をqueueに保存
    q.put(result)

def ai_hands_thinking(q):
    #過去のじゃんけんの手(ベクトル形式)をscikit_learn形式に
    Jprev_set = np.array([Jprev])
    #現在のじゃんけんの手(0~2の整数)をscikit_learn形式に
    jnow_set = np.array([j])

    #コンピュータが過去の手から人間の現在の手を予測
    jpredict = clf.predict(Jprev_set)

    #予測を元にコンピュータが決めた手
    #予測がグーならパー, チョキならグー, パーならチョキ
    comp_choice = (jpredict[0]+2)%3

    clf.partial_fit(Jprev_set, jnow_set)
    
    if(comp_choice == 0):
        ai_hands = 'guu'
    elif(comp_choice == 1):
        ai_hands = 'tyoki'
    elif(comp_choice == 2):
        ai_hands = 'pa'
    
    #idと手をセットでプッシュする
    result = (2, ai_hands)
    
    # 決定した手をqueueに保存
    q.put(result)

key = 0
# 各ジェスチャーの計測時間と最後に認識されたジェスチャー
gesture_start_time = None
gesture_last = None
total_time = 0
#メイン処理
if __name__ == "__main__":
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7) as hands:
        q = queue.Queue()
        user_hands = 'humei'

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # フレームをRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipeで手を検出
            results = hands.process(frame_rgb)

            # 検出した手の情報を取得
            # 開始前の表示処理
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 各指の位置を取得
                    flm = hand_landmarks.landmark

                    # グー、チョキ、パーの判定
                    if is_fist(hand_landmarks.landmark):
                        user_hands = "guu"
                    elif is_scissors(hand_landmarks.landmark):
                        user_hands = "tyoki"
                    elif is_palm(hand_landmarks.landmark):
                        user_hands = "pa"
                    else:
                        user_hands = "humei"
                    
                    # 前回のジェスチャーと異なる場合、トータル時間をリセット
                    if user_hands != gesture_last:
                        total_time = 0
                        gesture_start_time = time.time()
                        gesture_last = user_hands
                    else:
                        # 同じジェスチャーの場合、gesture_start_timeがNoneでないことを確認してから経過時間を計算
                        if gesture_start_time is None:
                            gesture_start_time = time.time()  # ジェスチャーが初めて認識された場合に現在の時刻を設定
                        total_time = time.time() - gesture_start_time

                    cv2.putText(frame, f"user_hands: {user_hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # 結果を画面に表示
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
            # フレームを表示
            cv2.imshow('Hand Count', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # スタート
            #　グーを三秒間認識した場合スタート
            if user_hands == "guu" and total_time >= 3:
                total_time = 0
                gesture_start_time = None
                #ai_hands = inference(user_hands) #推論を実行
                #推論
                if(user_hands == 'guu'):
                    j = 0
                elif(user_hands == 'tyoki'):
                    j = 1
                elif(user_hands == 'pa'):
                    j = 2
                elif(user_hands == 'humei'):
                    j = -1
                p1 = Thread(target=player_hands_thinking, args=(q,))
                p2 = Thread(target=ai_hands_thinking, args=(q,))
                p1.start()
                p2.start()
                res = [q.get(), q.get()]
                p1.join()
                p2.join()
                res.sort()
                print(res[0][1] , res[1][1])

    cap.release()
    cv2.destroyAllWindows()
    if key == ord('q'):
        root.destroy()
    else:
        root.mainloop()  # Tkinterウィンドウを表示