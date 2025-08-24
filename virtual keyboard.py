import cv2, time, math
import numpy as np
import mediapipe as mp
import pyautogui as pag
from dataclasses import dataclass

# ---------------- CONFIG ----------------
FRAME_W, FRAME_H = 1280, 720
KEY_H, BASE_KEY_W, GAP = 62, 62, 10
OVERLAY_ALPHA, HOVER_ALPHA = 0.35, 0.65
CLICK_COOLDOWN, SMOOTHING = 0.28, 0.35
CLICK_THRESH_PX, DRAW_MIRROR = 35, True   # distance between index & middle finger

ROW1 = [(c,1) for c in "1234567890"]
ROW2 = [(c,1) for c in "QWERTYUIOP"]
ROW3 = [(c,1) for c in "ASDFGHJKL"]
ROW4 = [(c,1) for c in "ZXCVBNM"] + [("SPACE",3), ("BKSP",2), ("ENTER",2)]
LAYOUT = [ROW1, ROW2, ROW3, ROW4]

pag.FAILSAFE = False


# ---------------- DATA MODELS ----------------
@dataclass
class KeyBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int


# ---------------- HAND TRACKER ----------------
class HandTracker:
    def __init__(self):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(max_num_hands=1,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6)
        self.last_index = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def get_fingers(self, res, w, h):
        if not res.multi_hand_landmarks:
            return None, None, None, None

        pts = [(int(lm.x * w), int(lm.y * h))
               for lm in res.multi_hand_landmarks[0].landmark]

        ix, iy = pts[8]   # index tip
        mx, my = pts[12]  # middle tip

        # smoothing
        if self.last_index is None:
            self.last_index = (ix, iy)
        else:
            lix, liy = self.last_index
            ix = int(SMOOTHING * ix + (1 - SMOOTHING) * lix)
            iy = int(SMOOTHING * iy + (1 - SMOOTHING) * liy)
            self.last_index = (ix, iy)

        return (ix, iy), (mx, my), pts


# ---------------- KEYBOARD RENDERER ----------------
class KeyRenderer:
    def __init__(self, layout, w, h):
        self.boxes = self._build_keyboard(layout, w, h)

    def _row_width(self, row):
        return sum(m for _, m in row) * BASE_KEY_W + (len(row) - 1) * GAP

    def _build_keyboard(self, layout, w, h):
        boxes, y = [], int(h * 0.18)
        for row in layout:
            x = (w - self._row_width(row)) // 2
            for label, mul in row:
                boxes.append(KeyBox(label, x, y, x + BASE_KEY_W * mul, y + KEY_H))
                x += BASE_KEY_W * mul + GAP
            y += KEY_H + GAP
        return boxes

    def draw(self, frame, hover_idx=None, pressed_idx=None):
        for i, b in enumerate(self.boxes):
            color, alpha = (203, 41, 214), OVERLAY_ALPHA
            if i == hover_idx:
                color, alpha = (0, 255, 255), HOVER_ALPHA
            if i == pressed_idx:
                color, alpha = (0, 200, 0), 0.85

            rect = np.zeros_like(frame, dtype=np.uint8)
            cv2.rectangle(rect, (b.x1, b.y1), (b.x2, b.y2), color, -1)
            frame = cv2.addWeighted(frame, 1, rect, alpha, 0)

            (tw, th), _ = cv2.getTextSize(b.label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = b.x1 + (b.x2 - b.x1 - tw) // 2
            ty = b.y1 + (b.y2 - b.y1 + th) // 2
            cv2.putText(frame, b.label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        return frame


# ---------------- KEYBOARD LOGIC ----------------
class GestureKeyboard:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, FRAME_W)
        self.cap.set(4, FRAME_H)
        self.hand = HandTracker()
        self.renderer = KeyRenderer(LAYOUT, FRAME_W, FRAME_H)
        self.last_press = 0
        self.pressed_idx = None
        self.pressed_until = 0

    def _distance(self, p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def _send_key(self, label):
        mapping = {"SPACE": "space", "ENTER": "enter", "BKSP": "backspace"}
        pag.press(mapping.get(label, label.lower()))

    def run(self):
        fps_time = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            if DRAW_MIRROR: frame = cv2.flip(frame, 1)

            res = self.hand.process(frame)
            hover_idx, click_active, click_dist = None, False, 999

            if res.multi_hand_landmarks:
                index, middle, pts = self.hand.get_fingers(res, FRAME_W, FRAME_H)
                if index and middle:
                    cv2.circle(frame, index, 10, (0,255,255), -1)
                    cv2.circle(frame, middle, 8, (0,200,0), -1)

                    click_dist = self._distance(index, middle)
                    click_active = click_dist < CLICK_THRESH_PX

                    for i, b in enumerate(self.renderer.boxes):
                        if b.x1 <= index[0] <= b.x2 and b.y1 <= index[1] <= b.y2:
                            hover_idx = i
                            break

            # Click detection
            now = time.time()
            if click_active and hover_idx is not None and now - self.last_press > CLICK_COOLDOWN:
                label = self.renderer.boxes[hover_idx].label
                self._send_key(label)
                self.last_press, self.pressed_idx, self.pressed_until = now, hover_idx, now + 0.2

            if self.pressed_idx is not None and now > self.pressed_until:
                self.pressed_idx = None

            # Draw keyboard
            frame = self.renderer.draw(frame, hover_idx, self.pressed_idx)

            # HUD
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            cv2.putText(frame, f"FPS: {int(fps)} | ClickDist: {int(click_dist)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Gesture Keyboard (Click)", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GestureKeyboard().run()
