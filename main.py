from fastapi import FastAPI, UploadFile, File
import shutil
import os
import cv2
from ultralytics import YOLO
import threading
import uuid

app = FastAPI()

# 📁 folders
UPLOAD_DIR = "videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🧠 YOLO model load
model = YOLO("yolov8n.pt")

# 🧠 job storage
jobs = {}


# =========================
# 📥 UPLOAD VIDEO API
# =========================
@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "uploaded successfully",
        "filename": file.filename,
        "path": file_path
    }


# =========================
# 🎥 VIDEO ANALYSIS
# =========================
def analyze_video_yolo(video_path, job_id, sport):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_index = 0

    highlights = []
    active_event = None

    frame_skip = int(fps * 2)
    last_event_type = None
    prev_frame = None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        total_frames = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        # 📊 progress update
        progress = int((frame_index / total_frames) * 100)
        jobs[job_id]["progress"] = progress

        # ⏩ skip frames
        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        # 🧠 motion detection
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            if diff.mean() < 7:
                frame_index += 1
                continue

        prev_frame = frame

        # 🧠 YOLO detection
        results = model(frame, imgsz=480, conf=0.3, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]

                event_type = None
                
                # 🔥 sport-based logic
                if sport == "Football":
                    if name == "sports ball":
                        event_type = "Goal"
                    elif name == "person":
                        event_type = "Player Movement"

                elif sport == "Cricket":
                    if name == "sports ball":
                        event_type = "Shot"
                    elif name == "person":
                        event_type = "Batting"

                elif sport == "Basketball":
                    if name == "sports ball":
                        event_type = "Basket Shot"
                    elif name == "person":
                        event_type = "Player Move"

                elif sport == "Volleyball":
                    if name == "sports ball":
                        event_type = "Spike"
                    elif name == "person":
                        event_type = "Jump"

                time_sec = frame_index / fps

                if event_type:
                    if active_event is None:
                        active_event = {
                            "tag": event_type,
                            "start": time_sec,
                            "end": time_sec
                        }
                    else:
                        active_event["end"] = time_sec

                    last_event_type = event_type

        # 🧠 close event
        if active_event:
            if frame_index / fps - active_event["end"] > 5:
                highlights.append(active_event)
                active_event = None

        frame_index += 1

    cap.release()

    if active_event:
        highlights.append(active_event)

    return highlights


# =========================
# 🚀 START ANALYSIS (BACKGROUND)
# =========================
@app.post("/start-analysis/")
def start_analysis(filename: str, sport: str):
    video_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(video_path):
        return {"error": "file not found"}

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "highlights": []
    }

    def run_analysis():
        highlights = analyze_video_yolo(video_path, job_id, sport)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["highlights"] = highlights

    threading.Thread(target=run_analysis).start()

    return {"job_id": job_id}


# =========================
# 📊 JOB STATUS API
# =========================
@app.get("/job-status/")
def job_status(job_id: str):
    return jobs.get(job_id, {"error": "job not found"})


# =========================
# 🔍 SMART SEARCH
# =========================
def find_event(highlights, query):
    query = query.lower()

    number_map = {
        "1st": "1",
        "2nd": "2",
        "3rd": "3"
    }

    for k, v in number_map.items():
        query = query.replace(k, v)

    for h in highlights:
        if query in h["tag"].lower():
            return h

    return None


@app.get("/search-event/")
def search_event(job_id: str, query: str):
    job = jobs.get(job_id)

    if not job or job["status"] != "completed":
        return {"error": "not ready"}

    result = find_event(job["highlights"], query)

    return result or {"message": "not found"}


# =========================
# 🎥 CLIP GENERATION
# =========================
def generate_clip(video_path, start, end, output):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (640, 360))

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_index / fps

        if start <= time_sec <= end:
            frame = cv2.resize(frame, (640, 360))
            out.write(frame)

        frame_index += 1

    cap.release()
    out.release()


@app.get("/generate-clip/")
def create_clip(filename: str, start: float, end: float):
    video_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(video_path):
        return {"error": "file not found"}

    output_path = f"videos/clip_{int(start)}_{int(end)}.mp4"

    generate_clip(video_path, start, end, output_path)

    return {"clip": output_path}