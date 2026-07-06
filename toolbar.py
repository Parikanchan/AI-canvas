import cv2
import numpy as np

# ----------------------------------------
# Toolbar Config
# ----------------------------------------

BUTTON_HEIGHT = 92   # a bit taller now: icon + label stacked
PADDING = 8
BTN_WIDTH = 100      # default/preferred width; shrinks automatically if frame is narrow
MIN_BTN_WIDTH = 55   # never shrink smaller than this (text stays readable)
RADIUS = 16

# Neutral "card" backgrounds — the colorful part now lives in the icon/swatch,
# not the whole button, which reads as a lot cleaner on camera.
CARD_BG = (42, 42, 46)
CARD_HOVER = (60, 60, 68)
CARD_SELECTED = (74, 74, 84)

COLOR_TOOL_NAMES = {"RED", "BLUE", "GREEN", "YELLOW", "BLACK"}

# "type" tells main.py how to treat a button:
#   color   -> selects a drawing color / becomes current_tool
#   tool    -> special drawing tool (eraser) / becomes current_tool
#   toggle  -> flips a persistent on/off mode (doesn't change current_tool)
#   action  -> one-shot trigger (doesn't change current_tool)
# "color" here = the accent color used for the icon/swatch and glow, not a
# full background fill anymore.
buttons = [
    {"name": "RED",    "color": (60, 60, 235),   "type": "color"},
    {"name": "BLUE",   "color": (235, 110, 60),  "type": "color"},
    {"name": "GREEN",  "color": (90, 200, 90),   "type": "color"},
    {"name": "YELLOW", "color": (0, 210, 235),   "type": "color"},
    {"name": "BLACK",  "color": (210, 210, 210), "type": "color"},
    {"name": "ERASER", "color": (235, 235, 235), "type": "tool"},
    {"name": "SHAPE",  "color": (255, 150, 40),  "type": "toggle"},
    {"name": "OCR",    "color": (0, 210, 255),   "type": "action"},
    {"name": "CLEAR",  "color": (190, 190, 190), "type": "action"},
]

TOOLBAR_WIDTH = 0  # set for real by layout_buttons() once frame width is known


def layout_buttons(frame_width):
    """
    Recompute each button's x1/x2 so all buttons always fit within the
    actual camera frame width, however wide/narrow it is. Call this once
    (per resolution) right after you know your frame's width, e.g.:
        h, w, _ = frame.shape
        layout_buttons(w)
    """
    global TOOLBAR_WIDTH

    n = len(buttons)
    available = frame_width - PADDING * (n + 1)
    btn_width = min(BTN_WIDTH, max(MIN_BTN_WIDTH, available // n))

    _x = PADDING
    for b in buttons:
        b["x1"] = _x
        b["x2"] = _x + btn_width
        _x += btn_width + PADDING

    TOOLBAR_WIDTH = _x
    return TOOLBAR_WIDTH


# Give buttons a sane default layout immediately (in case someone calls
# check_toolbar/draw_toolbar before layout_buttons is called explicitly).
layout_buttons(9 * (BTN_WIDTH + PADDING) + PADDING)


# ----------------------------------------
# Helper: rounded rectangle (anti-aliased)
# ----------------------------------------

def rounded_rect(img, pt1, pt2, color, radius, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)


# ----------------------------------------
# Helper: per-tool icon glyph (drawn in the button's accent color)
# ----------------------------------------

def draw_icon(frame, tool_name, accent, cx, cy, r):
    if tool_name in COLOR_TOOL_NAMES:
        real_color = get_color(tool_name)
        cv2.circle(frame, (cx, cy), r, real_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)

    elif tool_name == "ERASER":
        w_, h_ = int(r * 1.7), int(r * 1.15)
        pt1 = (cx - w_ // 2, cy - h_ // 2)
        pt2 = (cx + w_ // 2, cy + h_ // 2)
        rounded_rect(frame, pt1, pt2, (235, 235, 235), 6, -1)
        rounded_rect(frame, pt1, pt2, (150, 150, 150), 6, 1)
        cv2.line(frame, (pt1[0] + 6, pt2[1] - 4), (pt2[0] - 4, pt1[1] + 6), (190, 190, 195), 1, cv2.LINE_AA)

    elif tool_name == "SHAPE":
        offset = max(2, r // 3)
        sq_size = int(r * 1.15)

        # square drawn behind, shifted up-left
        sq_top_left = (cx - sq_size // 2 - offset, cy - sq_size // 2 - offset)
        sq_bottom_right = (cx + sq_size // 2 - offset, cy + sq_size // 2 - offset)
        rounded_rect(frame, sq_top_left, sq_bottom_right, accent, 4, 2)

        # circle drawn in front, shifted down-right, so the two overlap clearly
        circle_center = (cx + offset, cy + offset)
        circle_r = max(4, int(r * 0.62))
        cv2.circle(frame, circle_center, circle_r, (0, 210, 255), 2, cv2.LINE_AA)

    elif tool_name == "OCR":
        cv2.putText(frame, "Aa", (cx - r, cy + r // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, accent, 1, cv2.LINE_AA)

    elif tool_name == "CLEAR":
        body_w, body_h = int(r * 1.15), int(r * 1.3)
        top = cy - body_h // 2 + 4
        bottom = cy + body_h // 2
        left = cx - body_w // 2
        right = cx + body_w // 2
        cv2.line(frame, (left - 3, top), (right + 3, top), accent, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 4, top - 3), (cx + 4, top - 3), accent, 1, cv2.LINE_AA)
        rounded_rect(frame, (left, top), (right, bottom), (0, 0, 0), 2, 1)
        for lx in (left + body_w // 4, cx, right - body_w // 4):
            cv2.line(frame, (lx, top + 5), (lx, bottom - 3), accent, 1, cv2.LINE_AA)


# ----------------------------------------
# Draw Toolbar (modern glass card style)
# ----------------------------------------

def draw_toolbar(frame, current_tool, brush_size=5, hover_xy=None, shape_mode=False, brush_color=None):

    h, w, _ = frame.shape
    bar_h = BUTTON_HEIGHT + 18

    # Subtle vertical gradient background instead of a flat dark rectangle
    overlay = frame.copy()
    top_shade, bottom_shade = 18, 34
    for i in range(0, bar_h, 2):
        shade = top_shade + int((bottom_shade - top_shade) * (i / bar_h))
        cv2.rectangle(overlay, (0, i), (w, i + 2), (shade, shade, shade + 5), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # crisp accent line separating toolbar from the canvas below
    cv2.line(frame, (0, bar_h), (w, bar_h), (90, 90, 100), 1, cv2.LINE_AA)

    for button in buttons:

        top_left = (button["x1"], PADDING)
        bottom_right = (button["x2"], BUTTON_HEIGHT)
        btn_width = button["x2"] - button["x1"]
        cx = (button["x1"] + button["x2"]) // 2

        is_selected = current_tool == button["name"]
        is_toggled_on = button["name"] == "SHAPE" and shape_mode

        is_hover = False
        if hover_xy is not None:
            hx, hy = hover_xy
            is_hover = button["x1"] <= hx <= button["x2"] and 0 <= hy <= BUTTON_HEIGHT

        # ---- card background ----
        if is_selected or is_toggled_on:
            card_color = CARD_SELECTED
        elif is_hover:
            card_color = CARD_HOVER
        else:
            card_color = CARD_BG

        rounded_rect(frame, top_left, bottom_right, card_color, RADIUS, -1)

        # ---- soft outer glow for selected / toggled-on state ----
        if is_selected or is_toggled_on:
            glow_color = button["color"] if is_toggled_on else (255, 255, 255)
            for i, thickness in enumerate((5, 3, 1)):
                alpha_rect = frame.copy()
                rounded_rect(alpha_rect, (top_left[0] - 2, top_left[1] - 2),
                             (bottom_right[0] + 2, bottom_right[1] + 2),
                             glow_color, RADIUS + 2, thickness)
                cv2.addWeighted(alpha_rect, 0.35, frame, 0.65, 0, frame)
        elif is_hover:
            rounded_rect(frame, top_left, bottom_right, (150, 150, 155), RADIUS, 1)

        # ---- icon ----
        icon_cy = PADDING + (BUTTON_HEIGHT - PADDING) // 2 - 8
        icon_r = max(9, min(15, btn_width // 5))
        draw_icon(frame, button["name"], button["color"], cx, icon_cy, icon_r)

        # ---- label (auto-shrinks to fit narrow buttons) ----
        label_color = (225, 225, 230)
        if button["name"] in ("SHAPE", "OCR"):
            label_color = button["color"]

        font_scale = 0.42
        text_size = cv2.getTextSize(button["name"], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        while text_size[0] > btn_width - 8 and font_scale > 0.26:
            font_scale -= 0.03
            text_size = cv2.getTextSize(button["name"], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

        text_x = cx - text_size[0] // 2
        text_y = BUTTON_HEIGHT - 8

        cv2.putText(frame, button["name"], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, 1, cv2.LINE_AA)

        # small green "on" dot badge for the SHAPE toggle
        if is_toggled_on:
            cv2.circle(frame, (button["x2"] - 8, top_left[1] + 8), 4, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(frame, (button["x2"] - 8, top_left[1] + 8), 4, (20, 20, 20), 1, cv2.LINE_AA)

    # ---- Brush size / current color indicator (right side) ----
    indicator_x = TOOLBAR_WIDTH + 30
    if indicator_x + 90 < w:
        cv2.putText(frame, "BRUSH", (indicator_x, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 175), 1, cv2.LINE_AA)

        swatch_color = brush_color if brush_color is not None else (255, 255, 255)
        card_pt1 = (indicator_x, 32)
        card_pt2 = (indicator_x + 84, BUTTON_HEIGHT - 6)
        rounded_rect(frame, card_pt1, card_pt2, CARD_BG, 10, -1)
        rounded_rect(frame, card_pt1, card_pt2, (80, 80, 88), 10, 1)

        swatch_cy = (card_pt1[1] + card_pt2[1]) // 2
        cv2.circle(frame, (indicator_x + 22, swatch_cy), max(min(brush_size, 16), 3),
                   swatch_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (indicator_x + 22, swatch_cy), max(min(brush_size, 16), 3),
                   (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(brush_size) + "px", (indicator_x + 42, swatch_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 215), 1, cv2.LINE_AA)

    return frame


# ----------------------------------------
# Overlay: OCR result text box
# ----------------------------------------

def draw_ocr_overlay(frame, text):
    """Draws a semi-transparent box at the bottom of the frame with recognized text."""
    if not text:
        return frame

    h, w, _ = frame.shape
    box_h = 90
    overlay = frame.copy()
    rounded_rect(overlay, (10, h - box_h + 10), (w - 10, h - 10), (18, 18, 20), 18, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    rounded_rect(frame, (10, h - box_h + 10), (w - 10, h - 10), (0, 210, 255), 18, 1)

    cv2.putText(frame, "OCR Result", (26, h - box_h + 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 210, 255), 1, cv2.LINE_AA)

    # simple word-wrap so long text doesn't run off screen
    max_chars_per_line = max((w - 60) // 11, 10)
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > max_chars_per_line:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    for i, line in enumerate(lines[:2]):  # show up to 2 lines
        cv2.putText(frame, line, (26, h - box_h + 60 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 240), 1, cv2.LINE_AA)

    return frame


# ----------------------------------------
# Detect Selected Tool
# ----------------------------------------

def check_toolbar(x, y):

    if y > BUTTON_HEIGHT:
        return None

    for button in buttons:
        if button["x1"] <= x <= button["x2"]:
            return button["name"]

    return None


def get_button_type(tool_name):
    for button in buttons:
        if button["name"] == tool_name:
            return button["type"]
    return None


# ----------------------------------------
# Get Drawing Color
# ----------------------------------------

def get_color(tool):

    colors = {
        "RED": (0, 0, 255),
        "BLUE": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "YELLOW": (0, 255, 255),
        "BLACK": (30, 30, 30),
        "ERASER": (0, 0, 0),
    }

    return colors.get(tool, (255, 0, 255))