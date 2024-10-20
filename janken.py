import cv2
import mediapipe as mp
import time
import random
import numpy as np
import pygame
import tkinter as tk  # Tkinterをインポート
from PIL import Image, ImageTk

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
def display_janken_result(result):
    # result_label = tk.Label(root, text=result, font=("Helvetica", 24))
    # root.update()
    # result_label.pack()
    result_label.config(text=result)

    #音をならす。

    # 初期化
    pygame.mixer.init()

    # 音楽ファイルのロードと再生
    if result == "You Win":
        pygame.mixer.music.load('./ml-music/kati.mp3')  # 勝ちの場合の音
    elif result == "AI Wins":
        pygame.mixer.music.load('./ml-music/make.mp3')    # 負けの場合の音（適切なファイル名に変更）
    elif result == "Draw":
        pygame.mixer.music.load('./ml-music/hikiwake.mp3')    # あいこの場合の音（適切なファイル名に変更）
    else:
        print("不明な結果:", result)

        return  # 不明な結果の場合は処理を終了

    # 再生
    pygame.mixer.music.play()

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
    # 初期化
    # 音楽ファイルのロードと再生（ポンの音）
    pygame.mixer.init()
    pygame.mixer.music.load('./ml-music/pon.mp3')  
    pygame.mixer.music.play()
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
key = 0
#メイン処理
with mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipeで手を検出
        results = hands.process(frame_rgb)
        tyoki_count =0  # チョキの指のカウント

        # 検出した手の情報を取得
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # 各指の位置を取得
                flm = landmarks.landmark

                finger_count = 0
                # 人差し指が立っている
                if (flm[8].y < flm[7].y < flm[6].y < flm[0].y or
                    flm[8].y > flm[7].y > flm[6].y > flm[0].y):
                    finger_count += 1
                    tyoki_count += 1
                # 中指が立っている場合
                if (flm[12].y < flm[11].y < flm[10].y < flm[0].y or
                    flm[12].y > flm[11].y > flm[10].y > flm[0].y):
                    finger_count += 1
                    tyoki_count += 1
                # 薬指が立っている場合
                if (flm[16].y < flm[15].y < flm[14].y < flm[0].y or
                    flm[16].y > flm[15].y > flm[14].y > flm[0].y):
                    finger_count += 1
                # 小指が立っている場合
                if (flm[20].y < flm[19].y < flm[18].y < flm[0].y or
                    flm[20].y > flm[19].y > flm[18].y > flm[0].y):
                    finger_count += 1

                # 親指が立っている場合
                if (flm[4].x < flm[3].x < flm[5].x < flm[17].x or
                    flm[4].x > flm[3].x > flm[5].x > flm[17].x):
                    finger_count += 1

                # プレイヤーの手を判定
                if finger_count == 0:
                    user_hands = "guu"
                elif finger_count == 2 & tyoki_count == 2:
                    user_hands = "tyoki"
                elif finger_count == 5:
                    user_hands = "pa"
                else:
                    user_hands = "humei"

                cv2.putText(frame, f"user_hands: {user_hands}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 結果を画面に表示
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # スタート
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
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
        
            #不明の場合は出された手を記憶せずランダムで手を返す
            if(j == -1):
                ai_hands = random.choice(['guu', 'tyoki', 'pa'])
            else:
                #過去のじゃんけんの手(ベクトル形式)をscikit_learn形式に
                Jprev_set = np.array([Jprev])
                #現在のじゃんけんの手(0~2の整数)をscikit_learn形式に
                jnow_set = np.array([j])

                #コンピュータが過去の手から人間の現在の手を予測
                jpredict = clf.predict(Jprev_set)

                #人間の手
                your_choice = j
                #予測を元にコンピュータが決めた手
                #予測がグーならパー, チョキならグー, パーならチョキ
                comp_choice = (jpredict[0]+2)%3

                clf.partial_fit(Jprev_set, jnow_set)

                #過去の手の末尾に現在のコンピュータの手を追加
                Jprev = np.append(Jprev[3:], janken_array[comp_choice])
                #過去の手の末尾に現在の人間の手を追加
                Jprev = np.append(Jprev[3:], janken_array[your_choice])

                if(comp_choice == 0):
                    ai_hands = 'guu'
                elif(comp_choice == 1):
                    ai_hands = 'tyoki'
                elif(comp_choice == 2):
                    ai_hands = 'pa'

            countdown(user_hands,ai_hands)  # カウンタダウンを開始

        # フレームを表示
        cv2.imshow('Hand Count', frame)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
if key == ord('q'):
    root.destroy()
else:
    root.mainloop()  # Tkinterウィンドウを表示