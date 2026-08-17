import cv2
import time
import mediapipe as mp
import math
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
DETECTION_SCALE = 0.6

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

class HandTracker:
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.landmarks = None
        self.multi_landmarks = []
        self._middle_state = False

        self.smooth_x = None
        self.smooth_y = None
        self.smoothing = 0.6

        self.lost_frames = 0
        self.max_lost_frames = 4

        self.multi_lost_frames = 0
        self.max_multi_lost_frames = 8  # نسمح بفقد إيد لحظي قبل ما نمسح المستطيل

        self._pinch_state = False

        self._start_time = time.time()

    def find_hand(self, frame):
        small = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        # ---- إيد واحدة (للمؤشر) ----
        if result.hand_landmarks:
            self.landmarks = result.hand_landmarks[0]
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.landmarks = None

        # ---- إيدين (للمستطيل) - ميتحدثش إلا لو الاتنين ظاهرين ----
        if result.hand_landmarks and len(result.hand_landmarks) >= 2:
            self.multi_landmarks = result.hand_landmarks
            self.multi_lost_frames = 0
        else:
            self.multi_lost_frames += 1
            if self.multi_lost_frames > self.max_multi_lost_frames:
                self.multi_landmarks = result.hand_landmarks if result.hand_landmarks else []

        if result.hand_landmarks:
            h, w, _ = frame.shape
            for hand in result.hand_landmarks:
                for lm in hand:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                for start, end in HAND_CONNECTIONS:
                    x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
                    x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return frame

    def get_index_finger_position(self, frame_width, frame_height):
        if self.landmarks is None:
            return None
        index_tip = self.landmarks[8]
        x = int(index_tip.x * frame_width)
        y = int(index_tip.y * frame_height)

        if self.smooth_x is None:
            self.smooth_x, self.smooth_y = x, y
        else:
            self.smooth_x = int(self.smooth_x * self.smoothing + x * (1 - self.smoothing))
            self.smooth_y = int(self.smooth_y * self.smoothing + y * (1 - self.smoothing))

        return (self.smooth_x, self.smooth_y)

    # def is_pinching(self, frame_width, frame_height):
    #     if self.landmarks is None:
    #         return False
    #     thumb_tip = self.landmarks[4]
    #     index_tip = self.landmarks[8]
    #     x1, y1 = thumb_tip.x * frame_width, thumb_tip.y * frame_height
    #     x2, y2 = index_tip.x * frame_width, index_tip.y * frame_height
    #     return math.hypot(x2 - x1, y2 - y1) < 60

    def is_pinching(self, frame_width, frame_height):
        if self.landmarks is None:
            self._pinch_state = False
            return False
        thumb_tip = self.landmarks[4]
        index_tip = self.landmarks[8]
        x1, y1 = thumb_tip.x * frame_width, thumb_tip.y * frame_height
        x2, y2 = index_tip.x * frame_width, index_tip.y * frame_height
        distance = math.hypot(x2 - x1, y2 - y1)

        if not self._pinch_state:
            if distance < 55:
                self._pinch_state = True
        else:
            if distance > 75:
                self._pinch_state = False

        return self._pinch_state

    def is_fist(self):
        if self.landmarks is None:
            return False
        tips_ids = [8, 12, 16, 20]
        folded = sum(1 for tip_id in tips_ids if self.landmarks[tip_id].y > self.landmarks[tip_id - 2].y)
        return folded == 4

    def is_open_palm(self):
        if self.landmarks is None:
            return False
        tips_ids = [8, 12, 16, 20]
        fingers_up = sum(1 for tip_id in tips_ids if self.landmarks[tip_id].y < self.landmarks[tip_id - 2].y)
        return fingers_up == 4
        
    def get_two_hand_points(self, frame_width, frame_height):  
        points = []
        for hand in self.multi_landmarks: 
            thumb = hand[4]
            index = hand[8]
            points.append((int(thumb.x * frame_width), int(thumb.y * frame_height)))
            points.append((int(index.x * frame_width), int(index.y * frame_height)))
        return points

    def get_index_middle_spread(self, frame_width, frame_height):
        if self.landmarks is None:
            return 0
        index_tip = self.landmarks[8]
        middle_tip = self.landmarks[12]
        x1, y1 = index_tip.x * frame_width, index_tip.y * frame_height
        x2, y2 = middle_tip.x * frame_width, middle_tip.y * frame_height
        return math.hypot(x2 - x1, y2 - y1)


    def is_middle_up(self, frame_width, frame_height):
        if self.landmarks is None:
            self._middle_state = False
            return False
        tip_y = self.landmarks[12].y * frame_height
        pip_y = self.landmarks[10].y * frame_height
        diff = pip_y - tip_y  # لو موجب، يبقى رأس الوسطى فوق مفصلها = طالعة

        if not self._middle_state:
            if diff > 25:
                self._middle_state = True
        else:
            if diff < 10:
                self._middle_state = False

        return self._middle_state