import cv2
import mediapipe as mp
import numpy as np
from toolbar import (
    draw_toolbar, check_toolbar, get_color, get_button_type, draw_ocr_overlay,
    layout_buttons
)
from shapes_ocr import draw_clean_shape, run_ocr


def finger_up(hand_landmarks, tip, pip):
    return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y


# ----------------------------------------
# Fingertip smoothing (reduces jitter -> smoother lines/shapes)
# ----------------------------------------

class PointSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.x = None
        self.y = None

    def update(self, x, y):
        if self.x is None:
            self.x, self.y = x, y
        else:
            self.x = self.alpha * x + (1 - self.alpha) * self.x
            self.y = self.alpha * y + (1 - self.alpha) * self.y
        return int(self.x), int(self.y)

    def reset(self):
        self.x = None
        self.y = None


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("Camera not found")
    exit()

h, w, _ = frame.shape

# Make sure all toolbar buttons (including SHAPE/OCR/CLEAR) fit on screen,
# regardless of your webcam's resolution.
layout_buttons(w)

canvas = np.zeros((h, w, 3), dtype=np.uint8)
current_tool = "RED"
brush_color = (0, 0, 255)
brush_size = 5

prev_x = None
prev_y = None
smoother = PointSmoother(alpha=0.5)

# ---- Feature state ----
shape_mode = False          # toggled by the SHAPE button
stroke_points = []          # points of the *current* in-progress stroke
was_drawing = False         # was the previous frame in DRAW mode?

ocr_text = ""
ocr_text_frames_left = 0    # how many more frames to show the OCR overlay

action_cooldown = 0         # debounce for toggle/action buttons (SHAPE/OCR/CLEAR)
ACTION_COOLDOWN_FRAMES = 20

while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hover_xy = None  # default: no finger hovering toolbar
    drawing_this_frame = False

    if action_cooldown > 0:
        action_cooldown -= 1

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            index_up = finger_up(hand_landmarks, 8, 6)
            middle_up = finger_up(hand_landmarks, 12, 10)
            ring_up = finger_up(hand_landmarks, 16, 14)
            pinky_up = finger_up(hand_landmarks, 20, 18)

            index_tip = hand_landmarks.landmark[8]
            raw_x = int(index_tip.x * w)
            raw_y = int(index_tip.y * h)

            # Smooth the fingertip position to reduce jitter
            x, y = smoother.update(raw_x, raw_y)

            hover_xy = (x, y)

            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1, cv2.LINE_AA)

            # ---------------- DRAW MODE ----------------
            if index_up and not middle_up and not ring_up and not pinky_up:

                drawing_this_frame = True
                mode_label = "DRAW MODE" + (" (SHAPE)" if shape_mode else "")
                cv2.putText(frame, mode_label, (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if prev_x is None:
                    prev_x, prev_y = x, y

                if shape_mode:
                    # Don't commit to the real canvas yet — collect points and
                    # preview the raw stroke, then snap to a clean shape when done.
                    stroke_points.append((x, y))
                    cv2.line(frame, (prev_x, prev_y), (x, y), brush_color, brush_size, cv2.LINE_AA)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), brush_color, brush_size, cv2.LINE_AA)

                prev_x, prev_y = x, y

            # ---------------- SELECTION MODE ----------------
            elif index_up and middle_up and not ring_up and not pinky_up:

                cv2.putText(frame, "SELECTION MODE", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                prev_x = None
                prev_y = None
                smoother.reset()

                tool = check_toolbar(x, y)
                if tool is not None:
                    btn_type = get_button_type(tool)

                    if btn_type == "color":
                        current_tool = tool
                        brush_color = get_color(tool)
                        brush_size = 5

                    elif btn_type == "tool":  # ERASER
                        current_tool = tool
                        brush_color = get_color(tool)
                        brush_size = 40

                    elif btn_type == "toggle" and action_cooldown == 0:  # SHAPE
                        shape_mode = not shape_mode
                        stroke_points = []
                        action_cooldown = ACTION_COOLDOWN_FRAMES

                    elif btn_type == "action" and action_cooldown == 0:
                        if tool == "CLEAR":
                            canvas[:] = 0
                            stroke_points = []
                            ocr_text = ""
                            ocr_text_frames_left = 0
                        elif tool == "OCR":
                            ocr_text = run_ocr(canvas)
                            ocr_text_frames_left = 150  # ~5 sec at 30fps
                        action_cooldown = ACTION_COOLDOWN_FRAMES

            # ---------------- STOP (Fist) ----------------
            elif not index_up and not middle_up and not ring_up and not pinky_up:

                cv2.putText(frame, "STOP", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                prev_x = None
                prev_y = None
                smoother.reset()

            else:
                prev_x = None
                prev_y = None
                smoother.reset()

    else:
        prev_x = None
        prev_y = None
        smoother.reset()

    # ---- Stroke just ended: if shape mode is on, snap it to a clean shape ----
    if was_drawing and not drawing_this_frame and shape_mode and stroke_points:
        draw_clean_shape(canvas, stroke_points, brush_color, brush_size)
        stroke_points = []

    was_drawing = drawing_this_frame

    # ---- Toolbar draw ----
    frame = draw_toolbar(frame, current_tool, brush_size=brush_size,
                          hover_xy=hover_xy, shape_mode=shape_mode, brush_color=brush_color)

    # ---- Merge canvas with live frame ----
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, canvas_fg)

    # ---- OCR result overlay (fades out after a few seconds) ----
    if ocr_text_frames_left > 0:
        combined = draw_ocr_overlay(combined, ocr_text)
        ocr_text_frames_left -= 1

    cv2.imshow("Air Canvas", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()