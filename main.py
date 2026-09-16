import os
import shutil
import tempfile
import uuid

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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

def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def get_side_landmarks(landmarks, side: str):
    prefix = "RIGHT" if side == "right" else "LEFT"
    shoulder = landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_SHOULDER").value]
    hip = landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_HIP").value]
    knee = landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_KNEE").value]
    ankle = landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}_ANKLE").value]
    return shoulder, hip, knee, ankle


def pick_best_side(all_landmarks_list):
    right_scores, left_scores = [], []
    for landmarks in all_landmarks_list:
        if landmarks is None:
            continue
        rs, rh, rk, ra = get_side_landmarks(landmarks, "right")
        ls, lh, lk, la = get_side_landmarks(landmarks, "left")
        right_scores.append(np.mean([rs.visibility, rh.visibility, rk.visibility, ra.visibility]))
        left_scores.append(np.mean([ls.visibility, lh.visibility, lk.visibility, la.visibility]))
    return "right" if np.mean(right_scores) >= np.mean(left_scores) else "left"


def smooth(values, window=3):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return list(np.convolve(values, kernel, mode="valid"))


def find_reps(angles, fps, standing_thresh=140, min_gap_seconds=0.5):
    """
    Rep detection stays driven by knee angle only — hip/torso are read out
    at the same indices. min_gap is expressed in real time (seconds) and
    converted to frames using the video's actual fps, so a 60fps phone video
    and a 24fps video produce the same rep-detection behavior for the same
    physical tempo.
    """
    min_gap = max(1, int(round(min_gap_seconds * fps)))
    reps = []
    i = 0
    n = len(angles)
    while i < n:
        if angles[i] < standing_thresh:
            j = i
            min_idx = i
            while j < n and angles[j] < standing_thresh:
                if angles[j] < angles[min_idx]:
                    min_idx = j
                j += 1
            reps.append(min_idx)
            i = j + min_gap
        else:
            i += 1
    return reps


def process_video(video_path: str, output_path: str):
    """
    Single read pass to extract landmarks, then a second pass to draw the
    overlay. Tracks knee angle (depth/rep detection), hip angle (hip fold),
    and torso lean (forward lean from vertical) together, all keyed to the
    same trustworthy frames. Also tracks per-frame landmark visibility so we
    can report how confident the tracking was for each rep.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_landmarks = []
    frames_buffer = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            frame_landmarks.append(results.pose_landmarks.landmark if results.pose_landmarks else None)
            frames_buffer.append(frame)

    cap.release()

    if not any(frame_landmarks):
        raise ValueError("No person detected in the video. Make sure your full body is visible.")

    side = pick_best_side(frame_landmarks)

    raw_knee, raw_hip, raw_torso, raw_confidence = [], [], [], []
    per_frame_knee = []  

    for landmarks in frame_landmarks:
        if landmarks is None:
            per_frame_knee.append(None)
            continue
        shoulder, hip, knee, ankle = get_side_landmarks(landmarks, side)
        avg_visibility = np.mean([shoulder.visibility, hip.visibility, knee.visibility, ankle.visibility])
        if avg_visibility < VISIBILITY_THRESHOLD:
            per_frame_knee.append(None)
            continue

        knee_angle = calculate_angle([hip.x, hip.y], [knee.x, knee.y], [ankle.x, ankle.y])
        hip_angle = calculate_angle([shoulder.x, shoulder.y], [hip.x, hip.y], [knee.x, knee.y])
        vertical_ref = [hip.x, hip.y - 0.3]
        torso_lean = calculate_angle([shoulder.x, shoulder.y], [hip.x, hip.y], vertical_ref)

        raw_knee.append(knee_angle)
        raw_hip.append(hip_angle)
        raw_torso.append(torso_lean)
        raw_confidence.append(avg_visibility)
        per_frame_knee.append(knee_angle)

    if len(raw_knee) < 5:
        raise ValueError(
            "Not enough reliable pose data. Try filming with your full body visible and better lighting."
        )

    smoothed_knee = smooth(raw_knee, window=3)
    smoothed_hip = smooth(raw_hip, window=3)
    smoothed_torso = smooth(raw_torso, window=3)
    smoothed_confidence = smooth(raw_confidence, window=3)

    max_knee = float(np.max(smoothed_knee))
    rep_indices = find_reps(smoothed_knee, fps=fps)

    if not rep_indices:
        raise ValueError(
            "Couldn't detect a clear squat rep. Make sure you fully stand up between reps."
        )

    rep_depths = [round(float(smoothed_knee[i]), 1) for i in rep_indices]
    rep_hip_angles = [round(float(smoothed_hip[i]), 1) for i in rep_indices]
    rep_torso_leans = [round(float(smoothed_torso[i]), 1) for i in rep_indices]
    rep_confidences = [round(float(smoothed_confidence[i]) * 100, 0) for i in rep_indices]

    deepest_idx = int(np.argmin(rep_depths))
    deepest_angle = rep_depths[deepest_idx]
    avg_depth = float(np.mean(rep_depths))
    avg_hip_angle = float(np.mean(rep_hip_angles))
    max_torso_lean = float(np.max(rep_torso_leans))
    avg_confidence = round(float(np.mean(rep_confidences)), 1)

    for frame, landmarks, knee_angle in zip(frames_buffer, frame_landmarks, per_frame_knee):
        if landmarks is not None:
            h, w = frame.shape[:2]
            connections = mp_pose.POSE_CONNECTIONS
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

        if knee_angle is not None:
            cv2.putText(frame, f"knee: {int(knee_angle)} deg", (24, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (23, 61, 51), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()

    stats = {
        "standing_angle": round(max_knee, 1),
        "deepest_angle": round(deepest_angle, 1),
        "avg_bottom_depth": round(avg_depth, 1),
        "rep_count": len(rep_depths),
        "rep_depths": rep_depths,
        "rep_hip_angles": rep_hip_angles,
        "rep_torso_leans": rep_torso_leans,
        "rep_confidences": rep_confidences,
        "avg_hip_angle": round(avg_hip_angle, 1),
        "max_torso_lean": round(max_torso_lean, 1),
        "avg_confidence": avg_confidence,
        "side_used": side,
    }
    return stats


def get_coach_feedback(stats: dict) -> str:
    rep_list = ", ".join(f"{d}°" for d in stats["rep_depths"])
    hip_list = ", ".join(f"{d}°" for d in stats["rep_hip_angles"])
    torso_list = ", ".join(f"{d}°" for d in stats["rep_torso_leans"])

    prompt = f"""You are a knowledgeable strength coach reviewing squat form data captured from video pose-tracking.

Here is the measured data from the lifter's set:
- Standing knee angle (top of the rep): {stats['standing_angle']} degrees
- Number of reps detected: {stats['rep_count']}
- Knee depth per rep: {rep_list}
- Deepest rep (knee angle): {stats['deepest_angle']} degrees
- Average knee depth across all reps: {stats['avg_bottom_depth']} degrees
- Hip angle per rep (torso-hip-knee fold): {hip_list}
- Average hip angle: {stats['avg_hip_angle']} degrees
- Torso lean from vertical per rep: {torso_list}
- Max torso lean recorded: {stats['max_torso_lean']} degrees
- Average tracking confidence across reps: {stats['avg_confidence']}% (how visible/reliable the pose landmarks were)

For reference: knee angle of 90 degrees roughly means thighs parallel to the ground, lower is deeper.
A smaller hip angle relative to the knee angle (hip folding more than the knee bends) often signals excessive forward lean or a "good morning" squat pattern.
Torso lean near 0 degrees means upright, larger numbers mean leaning further forward over the bar.
If tracking confidence is notably low (below 75%), briefly mention the reading may be less precise, but don't dwell on it.

Give the lifter short, encouraging, and specific coaching feedback based ONLY on these numbers. Comment on:
1. Whether depth looks sufficient, too shallow, or excessively deep
2. Whether hip angle and torso lean suggest good posture or excessive forward lean / hips shooting up first
3. Whether reps were consistent with each other, or varied noticeably
4. One practical tip based on the data
Keep it to 5-6 sentences, casual coach tone, no fluff.
"""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    result_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(RESULTS_DIR, result_filename)

    try:
        stats = process_video(tmp_path, output_path)
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
