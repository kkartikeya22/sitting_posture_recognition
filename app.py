
import os
import io
import base64
import time
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from scipy.ndimage import gaussian_filter

# Import user's existing code
from config_reader import config_reader
from model import get_testing_model
import util

# ---------------------------
# Model bootstrap (one-time)
# ---------------------------
print("[boot] loading model weights...")
model = get_testing_model()
# Expect weights at ./model/keras/model.h5 relative to where app runs.
# You can change this path with the POSTURE_WEIGHTS environment variable.
weights_path = os.environ.get("POSTURE_WEIGHTS", "./model/keras/model.h5")
model.load_weights(weights_path)
print("[boot] model loaded from", weights_path)

print("[boot] loading config...")
PARAMS, MODEL_PARAMS = config_reader()
print("[boot] config loaded.")

# Colors for keypoints drawing (BGR)
COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
    [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

app = Flask(__name__, static_folder="static", static_url_path="/static")


def _posture_from_position_code(code: int) -> str:
    if code == 1:
        return "Hunchback"
    if code == -1:
        return "Reclined"
    if code == 0:
        return "Straight"
    return "Unknown"


def _calc_angle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if ax == bx:
            return 1.570796  # ~ pi/2
        return np.arctan2(by - ay, bx - ax)
    except Exception:
        return None


def _calc_distance(a, b):
    try:
        x1, y1 = a
        x2, y2 = b
        return float(np.hypot(x2 - x1, y2 - y1))
    except Exception:
        return None


def _check_position(all_peaks):
    # Uses ear and hip to categorize posture
    try:
        f = 0
        if all_peaks[16]:
            a = all_peaks[16][0][0:2]  # Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2]  # Left Ear
        b = all_peaks[11][0][0:2]  # Hip
        angle = _calc_angle(a, b)
        if angle is None:
            return None
        degrees = round(np.degrees(angle))
        if f:
            degrees = 180 - degrees
        if degrees < 70:
            return 1
        elif degrees > 110:
            return -1
        else:
            return 0
    except Exception:
        return None


def _check_hand_fold(all_peaks):
    # Returns a text note
    try:
        if (all_peaks[3][0][0:2]):
            try:
                if (all_peaks[4][0][0:2]):
                    distance = _calc_distance(all_peaks[3][0][0:2], all_peaks[4][0][0:2])
                    armdist = _calc_distance(all_peaks[2][0][0:2], all_peaks[3][0][0:2])
                    if distance is None or armdist is None:
                        return "Arms: Unknown"
                    if (armdist - 100) < distance < (armdist + 100):
                        return "Arms: Not Folding"
                    else:
                        return "Arms: Folding"
            except Exception:
                return "Arms: Folding"
    except Exception:
        try:
            if (all_peaks[7][0][0:2]):
                distance = _calc_distance(all_peaks[6][0][0:2], all_peaks[7][0][0:2])
                armdist = _calc_distance(all_peaks[6][0][0:2], all_peaks[5][0][0:2])
                if distance is None or armdist is None:
                    return "Arms: Unknown"
                if (armdist - 100) < distance < (armdist + 100):
                    return "Arms: Not Folding"
                else:
                    return "Arms: Folding"
        except Exception:
            return "Arms: Unknown"
    return "Arms: Unknown"


def _check_kneeling(all_peaks):
    # Returns a text note
    f = 0
    if all_peaks[16]:
        f = 1
    try:
        if (all_peaks[10][0][0:2] and all_peaks[13][0][0:2]):
            rightankle = all_peaks[10][0][0:2]
            leftankle = all_peaks[13][0][0:2]
            hip = all_peaks[11][0][0:2]
            leftangle = _calc_angle(hip, leftankle)
            rightangle = _calc_angle(hip, rightankle)
            if leftangle is None or rightangle is None:
                return "Legs: Unknown"
            leftdegrees = round(np.degrees(leftangle))
            rightdegrees = round(np.degrees(rightangle))
        if (f == 0):
            leftdegrees = 180 - leftdegrees
            rightdegrees = 180 - rightdegrees
        if (leftdegrees > 60 and rightdegrees > 60):
            return "Legs: Both Kneeling"
        elif (rightdegrees > 60):
            return "Legs: Right Kneeling"
        elif (leftdegrees > 60):
            return "Legs: Left Kneeling"
        else:
            return "Legs: Not Kneeling"
    except IndexError:
        try:
            if (f):
                a = all_peaks[10][0][0:2]
            else:
                a = all_peaks[13][0][0:2]
            b = all_peaks[11][0][0:2]
            angle = _calc_angle(b, a)
            if angle is None:
                return "Legs: Unknown"
            degrees = round(np.degrees(angle))
            if (f == 0):
                degrees = 180 - degrees
            if (degrees > 60):
                return "Legs: Both Kneeling"
            else:
                return "Legs: Not Kneeling"
        except Exception:
            return "Legs: Unknown"
    except Exception:
        return "Legs: Unknown"


def infer_keypoints(image_bgr: np.ndarray):
    """
    Runs the OpenPose-based model and returns:
      - canvas with drawn keypoints (BGR)
      - posture label (string)
      - notes list (arms/legs)
    This function is adapted from posture_realtime.py and posture_image.py
    but designed to operate in-memory for web requests.
    """
    oriImg = image_bgr

    # ✅ FIX: ensure we have a valid list of scales
    scale_search = PARAMS.get('scale_search') or [1.0]
    if not isinstance(scale_search, (list, tuple)) or len(scale_search) == 0:
        scale_search = [1.0]

    multiplier = [x * MODEL_PARAMS['boxsize'] / oriImg.shape[0] for x in scale_search]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    # ✅ iterate properly through all scales
    for m, scale in enumerate(multiplier):
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(
            imageToTest, MODEL_PARAMS['stride'], MODEL_PARAMS['padValue']
        )
        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 0, 1, 2))
        output_blobs = model.predict(input_img)

        heatmap = np.squeeze(output_blobs[1])
        heatmap = cv2.resize(
            heatmap, (0, 0), fx=MODEL_PARAMS['stride'], fy=MODEL_PARAMS['stride'], interpolation=cv2.INTER_CUBIC
        )
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])
        paf = cv2.resize(
            paf, (0, 0), fx=MODEL_PARAMS['stride'], fy=MODEL_PARAMS['stride'], interpolation=cv2.INTER_CUBIC
        )
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg += heatmap / len(multiplier)
        paf_avg += paf / len(multiplier)

    # Peak detection (18 body parts)
    all_peaks = []
    peak_counter = 0
    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        smoothed = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(smoothed.shape)
        map_left[1:, :] = smoothed[:-1, :]
        map_right = np.zeros(smoothed.shape)
        map_right[:-1, :] = smoothed[1:, :]
        map_up = np.zeros(smoothed.shape)
        map_up[:, 1:] = smoothed[:, :-1]
        map_down = np.zeros(smoothed.shape)
        map_down[:, :-1] = smoothed[:, 1:]

        peaks_binary = np.logical_and.reduce((
            smoothed >= map_left,
            smoothed >= map_right,
            smoothed >= map_up,
            smoothed >= map_down,
            smoothed > PARAMS['thre1']
        ))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        ids = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (ids[i],) for i in range(len(ids))]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # Draw keypoints
    canvas = oriImg.copy()
    for i in range(18):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, COLORS[i], thickness=-1)

    # Posture + notes
    pos_code = _check_position(all_peaks)
    posture = _posture_from_position_code(pos_code if pos_code is not None else 999)
    notes = [_check_hand_fold(all_peaks), _check_kneeling(all_peaks)]

    return canvas, posture, [n for n in notes if n]



def _b64_png_from_bgr(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    """
    Accepts a multipart/form-data with 'file' (image).
    Returns JSON: { image_b64: "...", posture: "Straight|Hunchback|Reclined|Unknown", notes: [...] }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    file_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    canvas, posture, notes = infer_keypoints(img)
    b64 = _b64_png_from_bgr(canvas)
    return jsonify({"image_b64": b64, "posture": posture, "notes": notes})


@app.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    """
    Accepts JSON: { frame_b64: "data:image/jpeg;base64,..." }
    Returns JSON: { image_b64: "...", posture: "..." }
    """
    try:
        data = request.get_json(silent=True) or {}
        frame_uri = data.get("frame_b64")
        if not frame_uri or "base64," not in frame_uri:
            return jsonify({"error": "Missing frame_b64"}), 400
        b64 = frame_uri.split("base64,")[1]
        frame_bytes = base64.b64decode(b64)
        arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid frame"}), 400

        canvas, posture, notes = infer_keypoints(frame)
        out_b64 = _b64_png_from_bgr(canvas)
        return jsonify({"image_b64": out_b64, "posture": posture, "notes": notes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Host and port can be overridden via env
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    debug = bool(int(os.environ.get("DEBUG", "0")))
    app.run(host=host, port=port, debug=debug)
