import cv2
import time

background = None
state = "calibrating"
calib_start_time = None
CALIB_DURATION = 5

smooth_box = None
BOX_SMOOTHING = 0.5

hands_lost_frames = 0
MAX_HANDS_LOST_FRAMES = 0  # قللناها عشان يمسح بسرعة أكبر

def reset():
    global background, state, calib_start_time, smooth_box, hands_lost_frames
    background = None
    state = "calibrating" 
    calib_start_time = None
    smooth_box = None
    hands_lost_frames = 0

def run(frame, hands_points):
    global background, state, calib_start_time, smooth_box, hands_lost_frames
    h, w, _ = frame.shape

    if state == "calibrating":
        if calib_start_time is None:
            calib_start_time = time.time()

        remaining = CALIB_DURATION - (time.time() - calib_start_time)

        if remaining > 0:
            cv2.putText(frame, "Step out of frame!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Capturing background in {int(remaining) + 1}...", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return frame

        background = frame.copy()
        state = "active"
        return frame

    if background is None:
        state = "calibrating"
        return frame

    if len(hands_points) >= 4:
        hands_lost_frames = 0
        xs = [p[0] for p in hands_points]
        ys = [p[1] for p in hands_points]
        raw_box = (max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys)))

        if smooth_box is None:
            smooth_box = raw_box
        else:
            smooth_box = tuple(
                int(smooth_box[i] * BOX_SMOOTHING + raw_box[i] * (1 - BOX_SMOOTHING))
                for i in range(4)
            )
    else:
        hands_lost_frames += 1
        if hands_lost_frames > MAX_HANDS_LOST_FRAMES:
            smooth_box = None

    if smooth_box:
        x1, y1, x2, y2 = smooth_box
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = background[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)

    return frame 