import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import random
import threading
import time
from collections import deque

# Settings
CAMERA_ID = 1
MODEL_PATH = "hand_landmarker.task"
FACE_DETECTOR_PATH = "blaze_face_short_range.tflite"
FRAME_WIDTH, FRAME_HEIGHT = 1440, 1080
SMOOTHING = 0.6
SKIP_FRAMES = 2

# Physics and cube settings
GRAVITY = 0.0  
FRICTION = 0.99 
BOUNCE = -0.8
MIN_CUBE_SCALE, MAX_CUBE_SCALE = 0.5, 1.3 
CUBE_HALF_SIZE = 1.2
CUBE_COLOR = np.array([250, 25, 62])
NEON_COLOR = (163, 106, 8)

# 2nd mode colors
PALM_WEB_COLOR = (140, 60, 110)
CROSS_HAND_COLOR = (180, 100, 100)

class GlobalState:
    def __init__(self):
        self.frame = None
        self.hand_result = None
        self.face_result = None
        self.running = True
        self.frame_count = 0
        self.lock = threading.Lock()

state = GlobalState()

# Object state
cube_pos = np.array([FRAME_WIDTH/2, FRAME_HEIGHT/2], dtype=float)
cube_vel = np.array([0.0, 0.0], dtype=float)
cube_depth = 5.0
cube_scale = 1.0
grabbed_hand_id = None
rotation_x, rotation_z = 0.0, 0.0
cube_material = "solid" 
hand_pos_history = deque(maxlen=5) 

# Flags
SKELETON_VISIBLE = True
FACE_BLUR_ENABLED = False
WEB_HANDS_VISIBLE = True
WEB_VISIBLE = True
LABEL_VISIBLE = True

FINGER_JOINTS = [4, 3, 8, 7, 12, 11, 16, 15, 20, 19]
FINGER_TIPS = [4, 8, 12, 16, 20]
PALM_BASE_INDICES = [0, 1, 2, 5, 9, 13, 17]
HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16)]

vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * CUBE_HALF_SIZE
faces_indices = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3], [1, 5, 6, 2], [0, 1, 5, 4], [3, 2, 6, 7]]
LIGHT_DIR = np.array([0.5, 0.5, -1.0])
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)

def processing_worker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH, delegate=python.BaseOptions.Delegate.CPU)
    hand_analyzer = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_tracking_confidence=0.4
    ))
    face_analyzer = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_DETECTOR_PATH), running_mode=vision.RunningMode.VIDEO
    ))
    while state.running:
        with state.lock:
            if state.frame is None or state.frame_count % SKIP_FRAMES != 0:
                state.frame_count += 1
                time.sleep(0.01)
                continue
            local_frame = state.frame.copy()
            curr_ts = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB))
        h_res = hand_analyzer.detect_for_video(mp_image, curr_ts)
        f_res = face_analyzer.detect_for_video(mp_image, curr_ts)
        with state.lock:
            state.hand_result = h_res
            state.face_result = f_res
            state.frame_count += 1
    hand_analyzer.close()
    face_analyzer.close()

def draw_bloom_poly(img, pts, color, thickness=2):
    mask = np.zeros_like(img)
    cv2.polylines(mask, [pts], True, color, thickness, cv2.LINE_AA)
    blur = cv2.GaussianBlur(mask, (15, 15), 0)
    return cv2.addWeighted(img, 1.0, blur, 2.5, 0)

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
threading.Thread(target=processing_worker, daemon=True).start()

selected_mode = int(input("Mode (1-Cube, 2-Web): ") or 1)
buttons = {"solid": (50, 50, 220, 110), "transparent": (50, 130, 220, 190), "neon": (50, 210, 220, 270)}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    with state.lock:
        state.frame = frame
        hands = state.hand_result
        faces = state.face_result

    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # UI
    if selected_mode == 1:
        for mat_name, (x1, y1, x2, y2) in buttons.items():
            b_color = (0, 180, 0) if cube_material == mat_name else (0, 0, 150)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), b_color, -1, cv2.LINE_AA)
            cv2.putText(canvas, mat_name.upper(), (x1+15, y1+40), 0, 0.6, (255, 255, 255), 2)

    # Face blur
    if FACE_BLUR_ENABLED and faces:
        for face in faces.detections:
            b = face.bounding_box
            sx, sy = max(0, int(b.origin_x)), max(0, int(b.origin_y))
            ex, ey = min(w, int(b.origin_x+b.width)), min(h, int(b.origin_y+b.height))
            canvas[sy:ey, sx:ex] = cv2.GaussianBlur(canvas[sy:ey, sx:ex], (51, 51), 30)

    # Cube physics and rendering
    if selected_mode == 1:
        if grabbed_hand_id is None:
            cube_pos += cube_vel
            cube_vel *= FRICTION
            if cube_pos[1] < 50 or cube_pos[1] > h - 50: cube_vel[1] *= BOUNCE; cube_pos[1] = np.clip(cube_pos[1], 50, h-50)
            if cube_pos[0] < 50 or cube_pos[0] > w - 50: cube_vel[0] *= BOUNCE; cube_pos[0] = np.clip(cube_pos[0], 50, w-50)

        cv2.circle(canvas, (int(cube_pos[0]), int(cube_pos[1])), 70, (0, 0, 255) if grabbed_hand_id is not None else (0, 255, 0), 2)

        rx = np.array([[1,0,0], [0,math.cos(rotation_x),-math.sin(rotation_x)], [0,math.sin(rotation_x),math.cos(rotation_x)]])
        rz = np.array([[math.cos(rotation_z),-math.sin(rotation_z),0], [math.sin(rotation_z),math.cos(rotation_z),0], [0,0,1]])
        rot_mat = rz @ rx
        t_3d = [rot_mat @ (v * cube_scale) for v in vertices]
        poly_data = []
        for f_idx in faces_indices:
            v_poly = np.array([t_3d[i] for i in f_idx])
            avg_z = np.mean(v_poly[:, 2]) + cube_depth
            pts = np.array([(int(p[0]*(600/(p[2]+cube_depth)) + cube_pos[0]), int(p[1]*(600/(p[2]+cube_depth)) + cube_pos[1])) for p in v_poly], np.int32)
            n = np.cross(v_poly[1]-v_poly[0], v_poly[2]-v_poly[0])
            if np.linalg.norm(n) > 1e-6:
                intensity = max(0.2, np.dot(n/np.linalg.norm(n), LIGHT_DIR))
                poly_data.append((avg_z, pts, (CUBE_COLOR * intensity).astype(int)))
        poly_data.sort(key=lambda x: x[0], reverse=True)

        for _, pts, color in poly_data:
            if cube_material == "solid": 
                cv2.fillPoly(canvas, [pts], (int(color[0]), int(color[1]), int(color[2])), cv2.LINE_AA)
            elif cube_material == "transparent":
                ov = canvas.copy()
                cv2.fillPoly(ov, [pts], (int(color[0]), int(color[1]), int(color[2])), cv2.LINE_AA)
                cv2.addWeighted(ov, 0.4, canvas, 0.6, 0, canvas)
            elif cube_material == "neon":
                glow_mask = np.zeros_like(canvas)
                cv2.polylines(glow_mask, [pts], True, NEON_COLOR, 8, cv2.LINE_AA)
                glow_mask = cv2.GaussianBlur(glow_mask, (15, 15), 0)
                canvas = cv2.addWeighted(canvas, 1.0, glow_mask, 1.5, 0)
                
                cv2.polylines(canvas, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

            if cube_material != "neon":
                cv2.polylines(canvas, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

    # Hand landmarks
    all_tips = []
    if hands and hands.hand_landmarks:
        for h_idx, landmarks in enumerate(hands.hand_landmarks):
            h_type = hands.handedness[h_idx][0].category_name
            tips = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in FINGER_TIPS]
            all_tips.append(tips)

            if SKELETON_VISIBLE:
                for s, e in HAND_CONNECTIONS:
                    cv2.line(canvas, (int(landmarks[s].x*w), int(landmarks[s].y*h)), (int(landmarks[e].x*w), int(landmarks[e].y*h)), (0, 255, 0), 1)
                for lm in landmarks: cv2.circle(canvas, (int(lm.x*w), int(lm.y*h)), 3, (0, 0, 255), -1)
            
            if LABEL_VISIBLE:
                cv2.putText(canvas, h_type, (int(landmarks[0].x*w), int(landmarks[0].y*h)-20), 0, 0.6, (255,255,255), 2)

            idx_smooth = np.array([landmarks[8].x * w, landmarks[8].y * h])
            pinch = math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y)

            # Interaction with cube
            if selected_mode == 1:
                for name, (x1, y1, x2, y2) in buttons.items():
                    if x1 < idx_smooth[0] < x2 and y1 < idx_smooth[1] < y2: cube_material = name

                if grabbed_hand_id is None:
                    if pinch < 0.07 and math.hypot(idx_smooth[0]-cube_pos[0], idx_smooth[1]-cube_pos[1]) < 120:
                        grabbed_hand_id = h_idx
                
                if grabbed_hand_id == h_idx:
                    if pinch < 0.07: cube_vel = (idx_smooth - cube_pos) * 0.5; cube_pos[:] = idx_smooth
                    else: grabbed_hand_id = None
                elif grabbed_hand_id is not None:
                    # Rotation
                    v1 = np.array([landmarks[5].x-landmarks[0].x, landmarks[5].y-landmarks[0].y, landmarks[5].z-landmarks[0].z])
                    v2 = np.array([landmarks[17].x-landmarks[0].x, landmarks[17].y-landmarks[0].y, landmarks[17].z-landmarks[0].z])
                    norm = np.cross(v1, v2)
                    if np.linalg.norm(norm) > 1e-6:
                        norm /= np.linalg.norm(norm)
                        rotation_x = rotation_x * SMOOTHING + (math.asin(-norm[1])*1.5) * (1-SMOOTHING)
                        rotation_z = rotation_z * SMOOTHING + (math.atan2(norm[0], norm[2])*1.5) * (1-SMOOTHING)

                if grabbed_hand_id is None:
                    cube_scale = cube_scale * SMOOTHING + np.interp(pinch, [0.05, 0.25], [MIN_CUBE_SCALE, MAX_CUBE_SCALE]) * (1-SMOOTHING)

            # Web hands
            if selected_mode == 2 and WEB_HANDS_VISIBLE:
                cx, cy = int(sum(landmarks[i].x for i in PALM_BASE_INDICES)/7 * w), int((sum(landmarks[i].y for i in PALM_BASE_INDICES)/7 + 0.1) * h)
                poly = [[int(cx + math.cos(t)*random.uniform(10, 45)), int(cy + math.sin(t)*random.uniform(10, 45))] for t in np.linspace(0, 2*math.pi, 10)]
                cv2.polylines(canvas, [np.array(poly, np.int32)], True, PALM_WEB_COLOR, 1)
                for px, py in poly:
                    for j in FINGER_JOINTS:
                        if random.random() > 0.3: cv2.line(canvas, (px, py), (int(landmarks[j].x*w), int(landmarks[j].y*h)), PALM_WEB_COLOR, 1)

    # Main web
    if selected_mode == 2 and WEB_VISIBLE and len(all_tips) == 2:
        for _ in range(80):
            if random.random() > 0.4:
                p1, p2 = random.choice(all_tips[0]), random.choice(all_tips[1])
                cv2.line(canvas, p1, p2, CROSS_HAND_COLOR, 1)

    cv2.imshow("Hand tracker", canvas)
    key = cv2.waitKey(1)
    if key == ord('q'): state.running = False; break
    elif key == ord('s'): SKELETON_VISIBLE = not SKELETON_VISIBLE
    elif key == ord('b'): FACE_BLUR_ENABLED = not FACE_BLUR_ENABLED
    elif key == ord('w'): WEB_VISIBLE = not WEB_VISIBLE
    elif key == ord('h'): WEB_HANDS_VISIBLE = not WEB_HANDS_VISIBLE
    elif key == ord('l'): LABEL_VISIBLE = not LABEL_VISIBLE

cap.release()
cv2.destroyAllWindows()