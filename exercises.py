"""
Exercise definitions: the math (angles, rep detection, per-rep stats) and the
Gemini prompt for each supported lift. Pure numpy, no MediaPipe/OpenCV, so it
can be tested without a video.

Every metrics function receives `p`, a dict of landmark name -> (x, y) in
PIXEL coordinates. Angles must be computed in pixels, not MediaPipe's
normalized 0-1 coordinates, because normalized x/y are scaled by the frame's
width and height separately and that distorts angles on non-square video.
"""
import numpy as np


def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))


def torso_lean(p):
    """Angle between the hip->shoulder line and straight up."""
    hip = p["hip"]
    return calculate_angle(p["shoulder"], hip, (hip[0], hip[1] - 100))


def smooth(values, window=3):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return list(np.convolve(values, kernel, mode="valid"))


def find_reps(angles, fps, standing_thresh=140, min_gap_seconds=0.5):
    """
    Rep detection is driven by one angle curve: each dip below
    standing_thresh is a rep, located at the dip's lowest point. min_gap is
    expressed in real time (seconds) and converted to frames using the
    video's actual fps, so the same physical tempo gives the same result on
    24fps and 60fps video.
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


# ---------------------------------------------------------------- metrics

def squat_metrics(p):
    return {
        "knee": calculate_angle(p["hip"], p["knee"], p["ankle"]),
        "hip": calculate_angle(p["shoulder"], p["hip"], p["knee"]),
        "torso": torso_lean(p),
    }


def deadlift_metrics(p):
    return {
        "hip": calculate_angle(p["shoulder"], p["hip"], p["knee"]),
        "knee": calculate_angle(p["hip"], p["knee"], p["ankle"]),
        "torso": torso_lean(p),
    }


def pushup_metrics(p):
    sx, sy = p["shoulder"]
    ax, ay = p["ankle"]
    hx, hy = p["hip"]
    dx, dy = ax - sx, ay - sy
    length = np.hypot(dx, dy)
    # Signed distance of the hip from the shoulder->ankle line, as a % of
    # that line's length. Positive = hip below the line (sag), negative =
    # above it (pike). sign(dx) makes "below" mean the same thing whichever
    # way the lifter is facing.
    cross = dx * (hy - sy) - dy * (hx - sx)
    direction = 1.0 if dx >= 0 else -1.0
    hip_dev = (cross / length**2 * direction * 100) if length else 0.0
    return {
        "elbow": calculate_angle(p["shoulder"], p["elbow"], p["wrist"]),
        "body_line": calculate_angle(p["shoulder"], p["hip"], p["ankle"]),
        "hip_dev": float(hip_dev),
    }


# ---------------------------------------------------------------- summaries

def _at(series, key, reps, ndigits=1):
    return [round(float(series[key][i]), ndigits) for i in reps]


def _common_stats(series, reps, side):
    rep_conf = [round(float(series["confidence"][i]) * 100, 0) for i in reps]
    return {
        "rep_count": len(reps),
        "rep_confidences": rep_conf,
        "avg_confidence": round(float(np.mean(rep_conf)), 1),
        "side_used": side,
    }


def summarize_squat(series, reps, side):
    knee = _at(series, "knee", reps)
    hip = _at(series, "hip", reps)
    torso = _at(series, "torso", reps)
    stats = {
        "standing_angle": round(float(np.max(series["knee"])), 1),
        "deepest_angle": round(float(np.min(knee)), 1),
        "avg_bottom_depth": round(float(np.mean(knee)), 1),
        "rep_depths": knee,
        "rep_hip_angles": hip,
        "rep_torso_leans": torso,
        "avg_hip_angle": round(float(np.mean(hip)), 1),
        "max_torso_lean": round(float(np.max(torso)), 1),
    }
    stats.update(_common_stats(series, reps, side))
    return stats


def summarize_deadlift(series, reps, side):
    hip = series["hip"]
    lockouts = []
    for k, i in enumerate(reps):
        # Lockout of rep k = highest hip angle between its bottom and the
        # midpoint to the next bottom (or the end of the video).
        end = (i + reps[k + 1]) // 2 if k + 1 < len(reps) else len(hip)
        lockouts.append(round(float(np.max(hip[i:max(end, i + 1)])), 1))
    bottom_hip = _at(series, "hip", reps)
    stats = {
        "rep_lockout_hips": lockouts,
        "avg_lockout_hip": round(float(np.mean(lockouts)), 1),
        "min_lockout_hip": round(float(np.min(lockouts)), 1),
        "rep_bottom_hips": bottom_hip,
        "rep_bottom_knees": _at(series, "knee", reps),
        "rep_bottom_torso": _at(series, "torso", reps),
        "avg_bottom_hip": round(float(np.mean(bottom_hip)), 1),
        "avg_bottom_knee": round(float(np.mean(_at(series, "knee", reps))), 1),
        "avg_bottom_torso": round(float(np.mean(_at(series, "torso", reps))), 1),
    }
    stats.update(_common_stats(series, reps, side))
    return stats


def summarize_pushup(series, reps, side):
    elbow = _at(series, "elbow", reps)
    line = _at(series, "body_line", reps)
    dev = _at(series, "hip_dev", reps)
    worst_dev = max(dev, key=abs)
    stats = {
        "standing_angle": round(float(np.max(series["elbow"])), 1),
        "deepest_angle": round(float(np.min(elbow)), 1),
        "avg_bottom_depth": round(float(np.mean(elbow)), 1),
        "rep_depths": elbow,
        "rep_body_lines": line,
        "rep_hip_devs": dev,
        "avg_body_line": round(float(np.mean(line)), 1),
        "worst_hip_dev": round(float(worst_dev), 1),
    }
    stats.update(_common_stats(series, reps, side))
    return stats


# ---------------------------------------------------------------- prompts

def _fmt(values, unit):
    return ", ".join(f"{v}{unit}" for v in values)


def _confidence_note(stats):
    return (
        f"- Average tracking confidence across reps: {stats['avg_confidence']}% "
        "(how visible/reliable the pose landmarks were)\n"
        "If tracking confidence is notably low (below 75%), briefly mention the "
        "reading may be less precise, but don't dwell on it."
    )


def squat_prompt(stats):
    return f"""You are a knowledgeable strength coach reviewing squat form data captured from video pose-tracking.

Here is the measured data from the lifter's set:
- Standing knee angle (top of the rep): {stats['standing_angle']} degrees
- Number of reps detected: {stats['rep_count']}
- Knee depth per rep: {_fmt(stats['rep_depths'], '°')}
- Deepest rep (knee angle): {stats['deepest_angle']} degrees
- Average knee depth across all reps: {stats['avg_bottom_depth']} degrees
- Hip angle per rep (torso-hip-knee fold): {_fmt(stats['rep_hip_angles'], '°')}
- Average hip angle: {stats['avg_hip_angle']} degrees
- Torso lean from vertical per rep: {_fmt(stats['rep_torso_leans'], '°')}
- Max torso lean recorded: {stats['max_torso_lean']} degrees
{_confidence_note(stats)}

For reference: knee angle of 90 degrees roughly means thighs parallel to the ground, lower is deeper.
A smaller hip angle relative to the knee angle (hip folding more than the knee bends) often signals excessive forward lean or a "good morning" squat pattern.
Torso lean near 0 degrees means upright, larger numbers mean leaning further forward over the bar.

Give the lifter short, encouraging, and specific coaching feedback based ONLY on these numbers. Comment on:
1. Whether depth looks sufficient, too shallow, or excessively deep
2. Whether hip angle and torso lean suggest good posture or excessive forward lean / hips shooting up first
3. Whether reps were consistent with each other, or varied noticeably
4. One practical tip based on the data
Keep it to 5-6 sentences, casual coach tone, no fluff.
"""


def deadlift_prompt(stats):
    return f"""You are a knowledgeable strength coach reviewing deadlift form data captured from side-view video pose-tracking.

Here is the measured data from the lifter's set:
- Number of reps detected: {stats['rep_count']}
- Hip angle (shoulder-hip-knee) at lockout per rep: {_fmt(stats['rep_lockout_hips'], '°')}
- Average lockout hip angle: {stats['avg_lockout_hip']} degrees (lowest rep: {stats['min_lockout_hip']} degrees)
- Hip angle at the bottom/setup per rep: {_fmt(stats['rep_bottom_hips'], '°')}
- Knee angle at the bottom/setup per rep: {_fmt(stats['rep_bottom_knees'], '°')}
- Torso lean from vertical at the bottom/setup per rep: {_fmt(stats['rep_bottom_torso'], '°')}
{_confidence_note(stats)}

For reference: a hip angle near 180 degrees at lockout means the hips are fully extended; clearly below roughly 165 suggests an incomplete lockout.
Bottom-position angles depend on stance (conventional vs sumo) and limb lengths, so judge them mainly by how consistent they are from rep to rep rather than against a fixed number.
The tracking only measures the shoulder-hip line, so you cannot see spinal rounding. Do not claim the back was rounded or neutral; say that spinal position can't be judged from this data if it's relevant.

Give the lifter short, encouraging, and specific coaching feedback based ONLY on these numbers. Comment on:
1. Whether the lockout looks complete on every rep
2. Whether the setup position (hip, knee, torso angle) was consistent between reps
3. Whether the setup looks more like a squat-style start (very bent knees, upright torso) or a hip-hinge-style start, if the numbers show it
4. One practical tip based on the data
Keep it to 5-6 sentences, casual coach tone, no fluff.
"""


def pushup_prompt(stats):
    return f"""You are a knowledgeable strength coach reviewing push-up form data captured from side-view video pose-tracking.

Here is the measured data from the lifter's set:
- Number of reps detected: {stats['rep_count']}
- Elbow angle at the top (arms extended): {stats['standing_angle']} degrees
- Elbow angle at the bottom per rep: {_fmt(stats['rep_depths'], '°')}
- Deepest rep (elbow angle): {stats['deepest_angle']} degrees
- Average bottom elbow angle: {stats['avg_bottom_depth']} degrees
- Body line angle (shoulder-hip-ankle) at the bottom per rep: {_fmt(stats['rep_body_lines'], '°')}
- Hip position relative to the shoulder-ankle line at the bottom per rep, as % of body length (positive = hips sagging below the line, negative = hips piking above it): {_fmt(stats['rep_hip_devs'], '%')}
- Largest hip deviation: {stats['worst_hip_dev']}%
{_confidence_note(stats)}

For reference: an elbow angle around 90 degrees or less at the bottom is a full-depth push-up; 110+ is a partial range. A top elbow angle near 180 means a full lockout. A body line angle near 180 degrees means a straight plank; sag or pike pulls it lower. Hip deviation within about 5% of body length is essentially straight.

Give the lifter short, encouraging, and specific coaching feedback based ONLY on these numbers. Comment on:
1. Whether depth looks full or partial, and whether the arms fully lock out at the top
2. Whether the body stays in a straight line, or whether hips sag or pike
3. Whether reps were consistent with each other, or varied noticeably
4. One practical tip based on the data
Keep it to 5-6 sentences, casual coach tone, no fluff.
"""


# ---------------------------------------------------------------- registry

EXERCISES = {
    "squat": {
        "name": "squat",
        "landmarks": ["shoulder", "hip", "knee", "ankle"],
        "metrics": squat_metrics,
        "rep_metric": "knee",
        "rep_threshold": 140,
        "overlay_metric": "knee",
        "overlay_label": "knee",
        "summarize": summarize_squat,
        "prompt": squat_prompt,
    },
    "deadlift": {
        "name": "deadlift",
        "landmarks": ["shoulder", "hip", "knee", "ankle"],
        "metrics": deadlift_metrics,
        "rep_metric": "hip",
        "rep_threshold": 150,
        "overlay_metric": "hip",
        "overlay_label": "hip",
        "summarize": summarize_deadlift,
        "prompt": deadlift_prompt,
    },
    "pushup": {
        "name": "push-up",
        "landmarks": ["shoulder", "elbow", "wrist", "hip", "ankle"],
        "metrics": pushup_metrics,
        "rep_metric": "elbow",
        "rep_threshold": 150,
        "overlay_metric": "elbow",
        "overlay_label": "elbow",
        "summarize": summarize_pushup,
        "prompt": pushup_prompt,
    },
}
