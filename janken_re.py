import cv2
import mediapipe as mp
import numpy as np
import pygame
from sklearn.linear_model import Perceptron
import sys
import math

# --- 基本定数 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
WHITE, BLACK, GRAY, LIGHT_GRAY = (255, 255, 255), (0, 0, 0), (200, 200, 200), (220, 220, 220)
GREEN, RED, BLUE, ORANGE = (50, 200, 50), (250, 50, 50), (50, 50, 250), (255, 165, 0)


# サイズ
CAM_DISPLAY_SIZE = 480
HAND_IMAGE_SIZE = 250
PROGRESS_BAR_HEIGHT = 15
# 垂直位置 (Y座標)
HEADER_Y = 50
PLAYER_NAME_Y = 100
CONTENT_Y = 150 # カメラや画像の描画を開始するY座標
PROMPT_Y = SCREEN_HEIGHT - 70 # 下部の指示テキストのY座標
PROGRESS_BAR_Y = SCREEN_HEIGHT - 35 # プログレスバーのY座標
# 水平位置 (X座標)
AI_AREA_X = SCREEN_WIDTH * 0.25
USER_AREA_X = SCREEN_WIDTH * 0.75

CENTER_TEXT_Y = 550

# --- アセットパス定義 ---
IMAGE_PATH = {"guu": './ml-images/human_gu.png',
              "tyoki": './ml-images/human_choki.png',
              "pa": './ml-images/human_pa.png',
              "hatena": './ml-images/hatena.png'
}
SOUND_PATH = {"win": './ml-music/kati.mp3',
              "lose": './ml-music/make.mp3',
              "draw": './ml-music/hikiwake.mp3',
              "pon": './ml-music/pon.mp3'
}
JANKEN_HANDS = ['guu', 'tyoki', 'pa']
FONT_PATH = "./fonts/migu-1m-regular.ttf"

# --- Pygameと関連モジュールの初期化 ---
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pygame AI じゃんけん")
clock = pygame.time.Clock()
# フォント読み込み (try-exceptでフォールバック)
try:
    font_small = pygame.font.Font(FONT_PATH, 24)
    font_medium = pygame.font.Font(FONT_PATH, 36)
    font_large = pygame.font.Font(FONT_PATH, 60)
except FileNotFoundError:
    print(f"フォントファイルが見つかりません: {FONT_PATH}. デフォルトフォントを使用します。")
    font_small = pygame.font.Font(None, 30)
    font_medium = pygame.font.Font(None, 50)
    font_large = pygame.font.Font(None, 80)
# ★変更: 定数を使用
images = {key: pygame.transform.scale(pygame.image.load(path), (HAND_IMAGE_SIZE, HAND_IMAGE_SIZE)) for key, path in IMAGE_PATH.items()}
sounds = {key: pygame.mixer.Sound(path) for key, path in SOUND_PATH.items()}
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# --- AIの初期化 ---
n_history = 30
JANKEN_VECTOR = {'guu': np.array([1, 0, 0]), 'tyoki': np.array([0, 1, 0]), 'pa': np.array([0, 0, 1])}
def initialize_ai_history():
    jprev = np.zeros(3 * n_history * 2)
    for i in range(n_history * 2):
        jprev[3*i : 3*i+3] = JANKEN_VECTOR[JANKEN_HANDS[i % 3]]
    return jprev
jprev = initialize_ai_history()
clf = Perceptron(random_state=None)
clf.partial_fit(np.array([jprev]), np.array([0]), classes=[0, 1, 2])

# --- 関数の定義 ---
def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
def get_hand_sign(landmarks):
    try:
        wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
        tips = {id: landmarks.landmark[id] for id in [4, 8, 12, 16, 20]}
        mcps = {id: landmarks.landmark[id] for id in [2, 5, 9, 13, 17]}
        fingers_open = [
            get_distance(tips[4], wrist) > get_distance(mcps[2], wrist),
            get_distance(tips[8], wrist) > get_distance(mcps[5], wrist),
            get_distance(tips[12], wrist) > get_distance(mcps[9], wrist),
            get_distance(tips[16], wrist) > get_distance(mcps[13], wrist),
            get_distance(tips[20], wrist) > get_distance(mcps[17], wrist)
        ]
        if fingers_open[1] and fingers_open[2] and not fingers_open[3] and not fingers_open[4]: return "tyoki"
        if sum(fingers_open) >= 4: return "pa"
        if sum(fingers_open) <= 1: return "guu"
        return "humei"
    except: return "humei"
def draw_text(text, font, color, surface, x, y, center=True, bg_color=None):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(**{('center' if center else 'topleft'): (x, y)})
    if bg_color:
        bg_rect = text_rect.inflate(20, 10)
        pygame.draw.rect(surface, bg_color, bg_rect, border_radius=10)
    surface.blit(text_surface, text_rect)

# --- ゲームの状態変数 ---
game_state = "waiting"
current_user_hand, final_user_hand, ai_hand = "humei", "humei", "humei"
result_text, countdown_text = "", ""
result_color = BLACK
rock_timer_start, countdown_timer, result_display_timer = 0, 0, 0

# --- メインループ ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x, start_y = (w - crop_size) // 2, (h - crop_size) // 2
        frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        current_user_hand = "humei"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_user_hand = get_hand_sign(hand_landmarks)
        frame = np.rot90(frame); frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_surface = pygame.surfarray.make_surface(frame)
        cam_surface = pygame.transform.scale(cam_surface, (CAM_DISPLAY_SIZE, CAM_DISPLAY_SIZE)) # ★変更

    if game_state == "waiting":
        if current_user_hand == "guu":
            if rock_timer_start == 0: rock_timer_start = pygame.time.get_ticks()
            if pygame.time.get_ticks() - rock_timer_start > 3000:
                game_state, countdown_timer, rock_timer_start = "countdown", pygame.time.get_ticks(), 0
        else: rock_timer_start = 0
    elif game_state == "countdown":
        elapsed = pygame.time.get_ticks() - countdown_timer
        if elapsed < 1000: countdown_text = "じゃん！"
        elif elapsed < 2000: countdown_text = "けん！"
        elif elapsed < 3000:
            if countdown_text != "ぽん！": sounds["pon"].play()
            countdown_text, final_user_hand = "ぽん！", current_user_hand
        else:
            if final_user_hand == 'humei': ai_choice_idx = np.random.randint(0, 3)
            else:
                user_choice_idx = JANKEN_HANDS.index(final_user_hand)
                jpredict = clf.predict(np.array([jprev])); ai_choice_idx = (jpredict[0] + 1) % 3
                clf.partial_fit(np.array([jprev]), np.array([user_choice_idx]))
                new_ai_vec, new_user_vec = JANKEN_VECTOR[JANKEN_HANDS[ai_choice_idx]], JANKEN_VECTOR[final_user_hand]
                jprev = np.roll(jprev, -6); jprev[-6:] = np.concatenate([new_ai_vec, new_user_vec])
            ai_hand = JANKEN_HANDS[ai_choice_idx]
            if final_user_hand == "humei": result_text, result_color = "判定不能です...", ORANGE
            elif final_user_hand == ai_hand: result_text, result_color = "あいこ", GRAY; sounds["draw"].play()
            elif (JANKEN_HANDS.index(final_user_hand) - JANKEN_HANDS.index(ai_hand)) % 3 == 2: result_text, result_color = "あなたの勝ち！", GREEN; sounds["win"].play()
            else: result_text, result_color = "あなたの負け！", RED; sounds["lose"].play()
            game_state, result_display_timer = "result", pygame.time.get_ticks()
    elif game_state == "result":
        if pygame.time.get_ticks() - result_display_timer > 3000: game_state = "waiting"; countdown_text = ""

    # --- ★★★ 描画処理（定数を使用） ★★★ ---
    screen.fill(WHITE)
    draw_text("AI じゃんけん", font_medium, BLACK, screen, SCREEN_WIDTH / 2, HEADER_Y)
    draw_text("AI", font_medium, BLACK, screen, AI_AREA_X, PLAYER_NAME_Y)
    draw_text("あなた", font_medium, BLACK, screen, USER_AREA_X, PLAYER_NAME_Y)
    draw_text(f"認識中の手: {current_user_hand}", font_small, RED, screen, 10, 10, center=False)

    cam_pos = (USER_AREA_X - CAM_DISPLAY_SIZE / 2, CONTENT_Y)
    hand_image_pos = (AI_AREA_X - HAND_IMAGE_SIZE / 2, CONTENT_Y + (CAM_DISPLAY_SIZE - HAND_IMAGE_SIZE) / 2)
    user_hand_image_pos = (USER_AREA_X - HAND_IMAGE_SIZE / 2, CONTENT_Y + (CAM_DISPLAY_SIZE - HAND_IMAGE_SIZE) / 2)

    # 各要素のX, Y座標を定数から計算
    cam_pos = (USER_AREA_X - CAM_DISPLAY_SIZE / 2, CONTENT_Y)
    hand_image_pos = (AI_AREA_X - HAND_IMAGE_SIZE / 2, CONTENT_Y + (CAM_DISPLAY_SIZE - HAND_IMAGE_SIZE) / 2)
    user_hand_image_pos = (USER_AREA_X - HAND_IMAGE_SIZE / 2, CONTENT_Y + (CAM_DISPLAY_SIZE - HAND_IMAGE_SIZE) / 2)

    if game_state == "waiting":
        if 'cam_surface' in locals(): screen.blit(cam_surface, cam_pos)
        screen.blit(images["hatena"], hand_image_pos)
        if rock_timer_start > 0:
            elapsed_sec = (pygame.time.get_ticks() - rock_timer_start) / 1000.0
            status_text = f"グーを認識中... {elapsed_sec:.1f} 秒"
            draw_text(status_text, font_small, WHITE, screen, SCREEN_WIDTH / 2, PROMPT_Y, bg_color=BLACK)
            progress_bar_pos_x = SCREEN_WIDTH / 2 - CAM_DISPLAY_SIZE / 2
            pygame.draw.rect(screen, GRAY, (progress_bar_pos_x, PROGRESS_BAR_Y, CAM_DISPLAY_SIZE, PROGRESS_BAR_HEIGHT), border_radius=5)
            pygame.draw.rect(screen, GREEN, (progress_bar_pos_x, PROGRESS_BAR_Y, CAM_DISPLAY_SIZE * (elapsed_sec / 3.0), PROGRESS_BAR_HEIGHT), border_radius=5)
        else:
            draw_text("グーの形で3秒間キープしてね！", font_medium, BLUE, screen, SCREEN_WIDTH / 2, PROMPT_Y)
    elif game_state == "countdown":
        draw_text(countdown_text, font_large, BLACK, screen, SCREEN_WIDTH / 2, CENTER_TEXT_Y)
        if 'cam_surface' in locals(): screen.blit(cam_surface, cam_pos)
        screen.blit(images["hatena"], hand_image_pos)
    elif game_state == "result":
        draw_text(result_text, font_large, result_color, screen, SCREEN_WIDTH / 2, CENTER_TEXT_Y)
        screen.blit(images[ai_hand], hand_image_pos)
        user_img = images[final_user_hand] if final_user_hand != 'humei' else images['hatena']
        screen.blit(user_img, user_hand_image_pos)

    pygame.display.flip()
    clock.tick(30)

cap.release()
hands.close()
pygame.quit()
sys.exit()