import cv2
import numpy as np

canvas = None
prev_point = None
prev_erase_point = None

smooth_draw_x = None
smooth_draw_y = None
DRAW_SMOOTHING = 0.55

colors = [
    {"name": "Red",   "bgr": (0, 0, 255)},
    {"name": "Green", "bgr": (0, 255, 0)},
    {"name": "Blue",  "bgr": (255, 0, 0)},
    {"name": "White", "bgr": (255, 255, 255)},
]
current_color_index = 0

BRUSH_THICKNESS = 6
TOOLBAR_HEIGHT = 70
CIRCLE_RADIUS = 22

ERASER_MIN_SIZE = 30
ERASER_MAX_SIZE = 160
SPREAD_MIN = 20
SPREAD_MAX = 120
smooth_eraser_size = ERASER_MIN_SIZE
ERASER_SMOOTHING = 0.7


def reset():
    global canvas, prev_point, prev_erase_point, smooth_eraser_size
    global smooth_draw_x, smooth_draw_y
    canvas = None
    prev_point = None
    prev_erase_point = None
    smooth_eraser_size = ERASER_MIN_SIZE
    smooth_draw_x = None
    smooth_draw_y = None


def draw_rounded_rect(img, pt1, pt2, color, radius):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)


def get_toolbar_buttons(frame_width):
    buttons = {}
    spacing = 70
    start_x = 60
    for i, c in enumerate(colors):
        cx = start_x + i * spacing
        cy = TOOLBAR_HEIGHT // 2
        buttons[f"color_{i}"] = {"center": (cx, cy), "radius": CIRCLE_RADIUS, "color": c["bgr"]}
    return buttons


def check_toolbar_click(cursor_pos):
    global current_color_index
    if cursor_pos is None:
        return
    cx, cy = cursor_pos
    buttons = get_toolbar_buttons(0)
    for key, data in buttons.items():
        bx, by = data["center"]
        r = data["radius"]
        if (cx - bx) ** 2 + (cy - by) ** 2 <= r ** 2:
            current_color_index = int(key.split("_")[1])


def draw_toolbar(frame):
    h, w, _ = frame.shape
    buttons = get_toolbar_buttons(w)
    bar_w = 60 + len(colors) * 70

    overlay = frame.copy()
    draw_rounded_rect(overlay, (10, 8), (bar_w, TOOLBAR_HEIGHT - 8), (30, 30, 30), 22)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    for i, c in enumerate(colors):
        key = f"color_{i}"
        cx, cy = buttons[key]["center"]
        cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, c["bgr"], -1)
        if i == current_color_index:
            cv2.circle(frame, (cx, cy), CIRCLE_RADIUS + 4, (255, 255, 255), 2)

    return frame


def run(frame, cursor_pos, is_pinching, is_open_palm, is_middle_up, finger_spread=0):
    global canvas, prev_point, prev_erase_point, smooth_eraser_size
    global smooth_draw_x, smooth_draw_y

    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    in_toolbar_zone = cursor_pos is not None and cursor_pos[1] < TOOLBAR_HEIGHT
    eraser_active = False

    if is_open_palm and cursor_pos and not in_toolbar_zone:
        eraser_active = True
        smooth_draw_x, smooth_draw_y = None, None

        spread_clamped = max(SPREAD_MIN, min(SPREAD_MAX, finger_spread))
        ratio = (spread_clamped - SPREAD_MIN) / (SPREAD_MAX - SPREAD_MIN)
        target_size = ERASER_MIN_SIZE + ratio * (ERASER_MAX_SIZE - ERASER_MIN_SIZE)
        smooth_eraser_size = smooth_eraser_size * ERASER_SMOOTHING + target_size * (1 - ERASER_SMOOTHING)
        eraser_size = int(smooth_eraser_size)

        if prev_erase_point is not None:
            cv2.line(canvas, prev_erase_point, cursor_pos, (0, 0, 0), eraser_size)
        else:
            cv2.circle(canvas, cursor_pos, eraser_size // 2, (0, 0, 0), -1)
        prev_erase_point = cursor_pos
        prev_point = None

    elif in_toolbar_zone:
        if is_pinching:
            check_toolbar_click(cursor_pos)
        prev_point = None
        prev_erase_point = None
        smooth_draw_x, smooth_draw_y = None, None

    else:
        prev_erase_point = None
        if cursor_pos is not None and not is_middle_up:
            raw_x, raw_y = cursor_pos

            if smooth_draw_x is None:
                smooth_draw_x, smooth_draw_y = raw_x, raw_y
            else:
                smooth_draw_x = smooth_draw_x * DRAW_SMOOTHING + raw_x * (1 - DRAW_SMOOTHING)
                smooth_draw_y = smooth_draw_y * DRAW_SMOOTHING + raw_y * (1 - DRAW_SMOOTHING)

            smoothed_point = (int(smooth_draw_x), int(smooth_draw_y))

            if prev_point is not None:
                cv2.line(canvas, prev_point, smoothed_point, colors[current_color_index]["bgr"], BRUSH_THICKNESS)
            prev_point = smoothed_point
        else:
            prev_point = None
            smooth_draw_x, smooth_draw_y = None, None

    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY_INV)
    mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

    frame_bg = cv2.bitwise_and(frame, mask_inv_3ch)
    frame = cv2.add(frame_bg, canvas)

    frame = draw_toolbar(frame)

    if eraser_active:
        cv2.circle(frame, cursor_pos, int(smooth_eraser_size) // 2, (200, 200, 200), 2)
    elif cursor_pos and not in_toolbar_zone:
        color = colors[current_color_index]["bgr"] if not is_middle_up else (150, 150, 150)
        cv2.circle(frame, cursor_pos, 8, color, -1)

    return frame