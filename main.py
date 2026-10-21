import os
import shutil
import tempfile
import uuid

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from exercises import EXERCISES, find_reps, smooth

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

VISIBILITY_THRESHOLD = 0.6
FEEDBACK_UNAVAILABLE = "Coach feedback is unavailable right now, but your measured angles above are still accurate."

def get_side_landmarks(landmarks, side: str, names):
    prefix = "RIGHT" if side == "right" else "LEFT"
    return {n: landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_{n.upper()}").value] for n in names}


def pick_best_side(all_landmarks_list, names):
    right_scores, left_scores = [], []
    for landmarks in all_landmarks_list:
        if landmarks is None:
            continue
        right = get_side_landmarks(landmarks, "right", names)
        left = get_side_landmarks(landmarks, "left", names)
        right_scores.append(np.mean([lm.visibility for lm in right.values()]))
        left_scores.append(np.mean([lm.visibility for lm in left.values()]))
    return "right" if np.mean(right_scores) >= np.mean(left_scores) else "left"


def process_video(video_path: str, output_path: str, exercise: str = "squat"):
    """
    Pass 1: read the video once to extract landmarks only (no frames kept in
    memory). Pass 2: re-open the video from disk and stream frame-by-frame
    into the overlay writer, so at most one frame is ever held in RAM at a
    time. The exercise config (see exercises.py) decides which landmarks and
    angles are measured, which angle drives rep detection, and how each rep
    is summarized. Per-frame landmark visibility is tracked so we can report
    how confident the tracking was for each rep.
    """
    cfg = EXERCISES[exercise]
    names = cfg["landmarks"]

    # ---------- Pass 1: landmarks only, no frames buffered ----------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_landmarks = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            frame_landmarks.append(results.pose_landmarks.landmark if results.pose_landmarks else None)
            # frame is discarded here — never appended to a buffer

    cap.release()

    if not any(frame_landmarks):
        raise ValueError("No person detected in the video. Make sure your full body is visible.")

    side = pick_best_side(frame_landmarks, names)

    raw = {}  # metric name -> per-trusted-frame values
    raw_confidence = []
    per_frame_overlay = []

    for landmarks in frame_landmarks:
        if landmarks is None:
            per_frame_overlay.append(None)
            continue
        lms = get_side_landmarks(landmarks, side, names)
        avg_visibility = np.mean([lm.visibility for lm in lms.values()])
        if avg_visibility < VISIBILITY_THRESHOLD:
            per_frame_overlay.append(None)
            continue

        # Pixel coordinates, so angles aren't distorted by the frame's aspect ratio.
        points = {n: (lm.x * width, lm.y * height) for n, lm in lms.items()}
        metrics = cfg["metrics"](points)
        for key, value in metrics.items():
            raw.setdefault(key, []).append(value)
        raw_confidence.append(avg_visibility)
        per_frame_overlay.append(metrics[cfg["overlay_metric"]])

    if len(raw_confidence) < 5:
        raise ValueError(
            "Not enough reliable pose data. Try filming with your full body visible and better lighting."
        )

    series = {key: smooth(values, window=3) for key, values in raw.items()}
    series["confidence"] = smooth(raw_confidence, window=3)

    rep_indices = find_reps(series[cfg["rep_metric"]], fps=fps, standing_thresh=cfg["rep_threshold"])

    if not rep_indices:
        raise ValueError(
            f"Couldn't detect a clear {cfg['name']} rep. Make sure you go through the full range of motion."
        )

    stats = cfg["summarize"](series, rep_indices, side)
    stats["exercise"] = exercise

    # ---------- Pass 2: re-read from disk, stream overlay writes ----------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap2 = cv2.VideoCapture(video_path)
    frame_idx = 0
    connections = mp_pose.POSE_CONNECTIONS

    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break

        landmarks = frame_landmarks[frame_idx] if frame_idx < len(frame_landmarks) else None
        overlay_angle = per_frame_overlay[frame_idx] if frame_idx < len(per_frame_overlay) else None

        if landmarks is not None:
            h, w = frame.shape[:2]
            for start_idx, end_idx in connections:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                if start.visibility > 0.5 and end.visibility > 0.5:
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (46, 125, 84), 2)
            for lm in landmarks:
                if lm.visibility > 0.5:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (23, 61, 51), -1)

        if overlay_angle is not None:
            cv2.putText(frame, f"{cfg['overlay_label']}: {int(overlay_angle)} deg", (24, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (23, 61, 51), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap2.release()
    writer.release()

    return stats


def get_coach_feedback(stats: dict) -> str:
    prompt = EXERCISES[stats["exercise"]]["prompt"](stats)
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        return response.text or FEEDBACK_UNAVAILABLE
    except Exception:
        # The measurements are still valid even if the coach text can't be generated.
        return FEEDBACK_UNAVAILABLE


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...), exercise: str = Form("squat")):
    if exercise not in EXERCISES:
        return JSONResponse(
            {"success": False, "error": f"Unknown exercise '{exercise}'. Choose one of: {', '.join(EXERCISES)}."},
            status_code=400,
        )

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    result_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(RESULTS_DIR, result_filename)

    try:
        stats = process_video(tmp_path, output_path, exercise)
        feedback = get_coach_feedback(stats)
        return JSONResponse({
            "success": True,
            "stats": stats,
            "feedback": feedback,
            "overlay_video_url": f"{str(request.base_url).rstrip('/')}/results/{result_filename}",
        })
    except ValueError as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)
    finally:
        os.remove(tmp_path)


@app.get("/")
def root():
    return {"status": "Form Check Coach backend is running"}
