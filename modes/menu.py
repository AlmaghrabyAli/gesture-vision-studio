import cv2

buttons_config = [
    {"key": "ghost",  "label": "Ghost Mode"},
    {"key": "canvas", "label": "Air Canvas"},
    {"key": "game",   "label": "RPS Game"},
]

SIDEBAR_WIDTH = 200
BUTTON_HEIGHT = 60
BUTTON_MARGIN = 15

def get_buttons(frame_width):
    buttons = {}
    start_y = 100
    sidebar_x1 = frame_width - SIDEBAR_WIDTH
    for i, btn in enumerate(buttons_config):
        y1 = start_y + i * (BUTTON_HEIGHT + BUTTON_MARGIN)
        y2 = y1 + BUTTON_HEIGHT
        x1 = sidebar_x1 + 15
        x2 = frame_width - 15
        buttons[btn["key"]] = {"pos": (x1, y1, x2, y2), "label": btn["label"]}
    return buttons

def draw_menu(frame, cursor_pos):
    h, w, _ = frame.shape
    sidebar_x1 = w - SIDEBAR_WIDTH
    buttons = get_buttons(w)

    overlay = frame.copy()
    cv2.rectangle(overlay, (sidebar_x1, 0), (w, h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    cv2.putText(frame, "MENU", (sidebar_x1 + 15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, "pinch to select", (sidebar_x1 + 15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    for name, data in buttons.items():
        x1, y1, x2, y2 = data["pos"]
        color = (60, 60, 60)
        if cursor_pos:
            cx, cy = cursor_pos
            if x1 < cx < x2 and y1 < cy < y2:
                color = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        text_size = cv2.getTextSize(data["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2
        cv2.putText(frame, data["label"], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def check_selection(cursor_pos, frame_width):
    if cursor_pos is None:
        return None
    buttons = get_buttons(frame_width)
    cx, cy = cursor_pos
    for name, data in buttons.items():
        x1, y1, x2, y2 = data["pos"]
        if x1 < cx < x2 and y1 < cy < y2:
            return name
    return None