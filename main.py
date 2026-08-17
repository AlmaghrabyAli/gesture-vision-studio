import cv2
import time
from utils.hand_tracker import HandTracker
from modes import menu
from modes import ghost
from modes import canvas

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

tracker = HandTracker()

cv2.namedWindow("Gesture Vision Studio", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gesture Vision Studio", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

current_mode = "menu"
previous_mode = "menu"
 
prev_pinching = False

hand_was_open = False
closure_events = []
CLOSURE_WINDOW = 1.2  # لازم القفلتين يحصلوا خلال أقل من 1.2 ثانية

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = tracker.find_hand(frame)

    h, w, _ = frame.shape
    cursor_pos = tracker.get_index_finger_position(w, h)
    pinching = tracker.is_pinching(w, h)
    fist = tracker.is_fist()
    open_palm = tracker.is_open_palm()

    just_pinched = pinching and not prev_pinching
    prev_pinching = pinching

    # ---- رجوع للمينو: افتح كفك بالكامل، بعدين اقفله (قبضة) - كرر مرتين ----
    if open_palm:
        hand_was_open = True

    if fist and hand_was_open:
        hand_was_open = False  # لازم تفتح تاني عشان تسجل قفلة جديدة
        now = time.time()
        closure_events.append(now)
        closure_events = [t for t in closure_events if now - t < CLOSURE_WINDOW]
        if len(closure_events) >= 2:
            current_mode = "menu"
            closure_events = []

    # ------- الدخول لوضع جديد: نعمل reset لو لسه دخلناه -------
    if current_mode == "ghost" and previous_mode != "ghost":
        ghost.reset()
    if current_mode == "canvas" and previous_mode != "canvas":
        canvas.reset()

    key = cv2.waitKey(1) & 0xFF

    # ------- منطق الأوضاع -------
    if current_mode == "menu":
        frame = menu.draw_menu(frame, cursor_pos)

        if just_pinched:
            selected = menu.check_selection(cursor_pos, w)
            if selected:
                current_mode = selected

    elif current_mode == "ghost":
        hands_points = tracker.get_two_hand_points(w, h)
        frame = ghost.run(frame, hands_points)
        if key == ord('r'):
            ghost.reset()

    # elif current_mode == "canvas":
    #     frame = canvas.run(frame, cursor_pos, pinching, open_palm)

    elif current_mode == "canvas":
        finger_spread = tracker.get_index_middle_spread(w, h)
        middle_up = tracker.is_middle_up(w, h)
        frame = canvas.run(frame, cursor_pos, pinching, open_palm, middle_up, finger_spread)

    elif current_mode == "game":
        cv2.putText(frame, f"Mode: {current_mode} (not built yet)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Open hand then close fist twice to go back", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    previous_mode = current_mode

    if cursor_pos and current_mode == "menu":
        color = (0, 0, 255) if pinching else (255, 0, 0)
        cv2.circle(frame, cursor_pos, 9, color, -1)
        cv2.circle(frame, cursor_pos, 11, (255, 255, 255), 1)

    cv2.imshow("Gesture Vision Studio", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  