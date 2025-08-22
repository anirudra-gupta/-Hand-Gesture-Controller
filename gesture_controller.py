import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import time
import os
import absl.logging

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera Error: Check if camera is connected and not in use by other apps")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Click gesture variables
click_threshold = 0.05  # Distance between thumb tip and palm for click
last_click_time = 0
click_delay = 0.5  # Seconds between clicks

def execute_action(action):
    if action == "youtube":
        webbrowser.open("youtube.com")
    elif action == "instagram":
        webbrowser.open("instagram.com")
    elif action == "scroll_up":
        pyautogui.scroll(100)
    elif action == "scroll_down":
        pyautogui.scroll(-100)

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                # Get key landmarks
                lm = hand.landmark
                wrist = lm[0]   # Wrist/palm base
                thumb_tip = lm[4]
                index_tip = lm[8]
                
                # Finger states (1=extended, 0=folded)
                fingers = [
                    1 if lm[i].y < lm[i-2].y else 0
                    for i in [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
                ]
                thumb_state = 1 if thumb_tip.x < lm[3].x else 0
                total_fingers = sum(fingers)
                
                # Cursor control (index finger only)
                if total_fingers == 1 and fingers[0] == 1:
                    x = int(index_tip.x * pyautogui.size()[0])
                    y = int(index_tip.y * pyautogui.size()[1])
                    pyautogui.moveTo(pyautogui.size()[0] - x, y)
                
                # NEW CLICK GESTURE: Thumb taps palm (wrist)
                thumb_to_palm_dist = ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)**0.5
                if thumb_to_palm_dist < click_threshold and time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.putText(frame, "CLICK", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
                # YouTube (peace sign)
                if total_fingers == 2 and fingers[0] == fingers[1] == 1:
                    execute_action("youtube")
                    cv2.putText(frame, "YOUTUBE", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                
                # Instagram (middle+ring)
                if total_fingers == 2 and fingers[1] == fingers[2] == 1:
                    execute_action("instagram")
                    cv2.putText(frame, "INSTAGRAM", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                
                # Scroll up (open hand)
                if total_fingers == 4:
                    execute_action("scroll_up")
                    cv2.putText(frame, "SCROLL UP", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                
                # Scroll down (fist)
                if total_fingers == 0:
                    execute_action("scroll_down")
                    cv2.putText(frame, "SCROLL DOWN", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # Draw hand
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

        # Display instructions
        instructions = [
            "1 Finger: Move Cursor",
            "Thumb to Palm: Click",
            "Peace Sign (2 fingers): YouTube",
            "Middle+Ring: Instagram",
            "Open Hand: Scroll Up",
            "Fist: Scroll Down",
            "Press Q to quit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10,30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow('Gesture Control (Thumb Tap to Click)', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()