import cv2
import numpy as np
import os
import shutil

# ----------------------------------------
# OCR (optional dependency)
# ----------------------------------------
# pytesseract needs the actual Tesseract-OCR binary installed on your system,
# not just the pip package. If it's missing, OCR calls will return a helpful
# message instead of crashing the app.
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

if TESSERACT_AVAILABLE:
    # 1) Highest priority: an explicit path set via environment variable,
    #    so you never have to edit this file directly.
    #       Windows (cmd):        set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
    #       Windows (PowerShell): $env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
    #       macOS/Linux:          export TESSERACT_CMD=/usr/local/bin/tesseract
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.exists(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path

    # 2) If tesseract isn't already discoverable on PATH, fall back to
    #    checking the most common default install locations.
    elif shutil.which("tesseract") is None:
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",   # Apple Silicon Homebrew
            "/usr/bin/tesseract",
        ]
        for p in common_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break


# ----------------------------------------
# Shape Recognition
# ----------------------------------------

MIN_STROKE_POINTS = 6          # ignore accidental taps/very short strokes
MIN_STROKE_AREA = 400          # ignore tiny scribbles


def classify_shape(contour):
    """Classify a contour as LINE / TRIANGLE / SQUARE / RECTANGLE / CIRCLE / POLYGON."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "UNKNOWN", None

    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(approx)

    if area < MIN_STROKE_AREA and vertices <= 2:
        return "LINE", (x, y, w, h)

    if vertices == 2:
        return "LINE", (x, y, w, h)
    elif vertices == 3:
        return "TRIANGLE", (x, y, w, h)
    elif vertices == 4:
        aspect_ratio = w / float(h) if h != 0 else 1
        if 0.9 <= aspect_ratio <= 1.1:
            return "SQUARE", (x, y, w, h)
        return "RECTANGLE", (x, y, w, h)
    else:
        # Compare stroke area to the area of its minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        if circle_area > 0 and area / circle_area > 0.65:
            return "CIRCLE", (int(cx), int(cy), int(radius))
        return "POLYGON", (x, y, w, h)


def stroke_points_to_contour(points):
    """Convert a list of (x, y) fingertip points into an OpenCV contour."""
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.convexHull(pts)


def draw_clean_shape(canvas, points, color, thickness):
    """
    Given the raw points of a finished freehand stroke, detect the intended
    shape and draw a clean, perfect version of it onto the canvas.
    Returns the shape name that was drawn (or None if the stroke was too short).
    """
    if len(points) < MIN_STROKE_POINTS:
        return None

    contour = stroke_points_to_contour(points)
    shape, params = classify_shape(contour)

    if shape == "LINE":
        pts = np.array(points)
        start = tuple(pts[0])
        end = tuple(pts[-1])
        cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)

    elif shape in ("SQUARE", "RECTANGLE"):
        x, y, w, h = params
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

    elif shape == "TRIANGLE":
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        cv2.polylines(canvas, [approx], True, color, thickness, cv2.LINE_AA)

    elif shape == "CIRCLE":
        cx, cy, radius = params
        cv2.circle(canvas, (cx, cy), radius, color, thickness, cv2.LINE_AA)

    elif shape == "POLYGON":
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        cv2.polylines(canvas, [approx], True, color, thickness, cv2.LINE_AA)

    else:
        # fallback: just draw the raw stroke as-is
        for i in range(1, len(points)):
            cv2.line(canvas, points[i - 1], points[i], color, thickness, cv2.LINE_AA)

    return shape


# ----------------------------------------
# OCR
# ----------------------------------------

def prepare_canvas_for_ocr(canvas):
    """
    Convert the (black background, colored strokes) drawing canvas into a
    clean black-text-on-white-background image, which Tesseract reads best.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    mask = gray > 20

    ocr_img = np.full(gray.shape, 255, dtype=np.uint8)
    ocr_img[mask] = 0

    # Thicken strokes slightly so thin freehand writing stays connected
    kernel = np.ones((3, 3), np.uint8)
    ocr_img = cv2.erode(ocr_img, kernel, iterations=1)

    # Upscale — handwriting-style strokes are easier for Tesseract to read
    # when the image is larger.
    ocr_img = cv2.resize(ocr_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return ocr_img


def run_ocr(canvas):
    """Run OCR on the drawing canvas and return recognized text (or a status message)."""
    if not TESSERACT_AVAILABLE:
        msg = "pytesseract not installed -> run: pip install pytesseract"
        print("[OCR]", msg)
        return msg

    ocr_img = prepare_canvas_for_ocr(canvas)

    try:
        text = pytesseract.image_to_string(ocr_img, config="--psm 6")
    except pytesseract.pytesseract.TesseractNotFoundError:
        msg = (
            "Tesseract binary not found on PATH. Fix it one of these ways:\n"
            "  1) Find where tesseract.exe/tesseract got installed, then set it via:\n"
            '     Windows (PowerShell): $env:TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"\n'
            "     macOS/Linux:          export TESSERACT_CMD=/usr/local/bin/tesseract\n"
            "     (run that in the SAME terminal before running main.py)\n"
            "  2) Or install Tesseract if you haven't: see requirements.txt for OS-specific steps."
        )
        print("[OCR]", msg)
        return "Tesseract path not set — see terminal for exact fix"
    except Exception as e:
        print("[OCR] error:", repr(e))
        return f"OCR error: {e}"

    text = text.strip()
    print("[OCR] recognized text:", repr(text))
    return text if text else "(no text recognized — try writing bigger/clearer)"