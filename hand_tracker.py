import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import random
import threading
import time

#Video settings
CAMERA_ID = 1
MODEL_PATH = "hand_landmarker.task"
FACE_DETECTOR_PATH = "blaze_face_short_range.tflite"

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SMOOTHING = 0.65
SKIP_FRAMES = 2


class GlobalState:
    def __init__(self):
        self.frame = None
        self.hand_result = None
        self.face_result = None
        self.running = True
        self.frame_count = 0
        self.lock = threading.Lock()

state = GlobalState()

#Cube, skeleton, web, blur settings
cube_pos = np.array([FRAME_WIDTH/2, FRAME_HEIGHT/2], dtype=float)
grabbed_hand_id = None
rotation_x, rotation_z = 0.0, 0.0
CUBE_HALF_SIZE = 1.2
CUBE_COLOR = (0, 165, 255)

PALM_WEB_COLOR = (140, 60, 110)
CROSS_HAND_COLOR = (180, 100, 100)
WEB_THICKNESS = 1
FLICKER_THRESHOLD = 0.25
SKELETON_VISIBLE = False
FACE_BLUR_ENABLED = False
WEB_HANDS_VISIBLE = True
WEB_VISIBLE = True
LABEL_VISIBLE = True

FINGER_JOINTS = [4, 3, 8, 7, 12, 11, 16, 15, 20, 19]
FINGER_TIPS = [4, 8, 12, 16, 20]
PALM_BASE_INDICES = [0, 1, 2, 5, 9, 13, 17]
HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16)]

vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * CUBE_HALF_SIZE
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

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


cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

selected_mode = int(input("Mode (1-Cube, 2-Web): ") or 1)

threading.Thread(target=processing_worker, daemon=True).start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    #frame = cv2.flip(frame, 1)
    
    with state.lock:
        state.frame = frame
        hands = state.hand_result
        faces = state.face_result

    canvas = frame.copy()
    h, w = canvas.shape[:2]

    #Face blur
    if FACE_BLUR_ENABLED and faces:
        for face in faces.detections:
            b = face.bounding_box
            sx, sy, ex, ey = max(0, int(b.origin_x)), max(0, int(b.origin_y)), min(w, int(b.origin_x+b.width)), min(h, int(b.origin_y+b.height))
            canvas[sy:ey, sx:ex] = cv2.GaussianBlur(canvas[sy:ey, sx:ex], (51, 51), 30)

    #Cube moving
    if selected_mode == 1:
        rx = np.array([[1,0,0], [0,math.cos(rotation_x),-math.sin(rotation_x)], [0,math.sin(rotation_x),math.cos(rotation_x)]])
        rz = np.array([[math.cos(rotation_z),-math.sin(rotation_z),0], [math.sin(rotation_z),math.cos(rotation_z),0], [0,0,1]])
        proj = []
        for v in vertices:
            p = rz @ rx @ v
            f = 350 / (p[2] + 5.0)
            proj.append((int(p[0]*f + cube_pos[0]), int(p[1]*f + cube_pos[1])))
        
        zone_color = (0, 80, 255) if grabbed_hand_id is not None else (0, 220, 100)
        cv2.circle(canvas, (int(cube_pos[0]), int(cube_pos[1])), 60, zone_color, 2)
        for s, e in edges: cv2.line(canvas, proj[s], proj[e], CUBE_COLOR, 3)

    #Hands detection, skeleton, web
    all_tips = []
    if hands and hands.hand_landmarks:
        num_hands_in_frame = len(hands.hand_landmarks)

        for h_idx, landmarks in enumerate(hands.hand_landmarks):
            h_type = hands.handedness[h_idx][0].category_name
            tips = [(int(lm.x * w), int(lm.y * h)) for lm in [landmarks[i] for i in FINGER_TIPS]]
            all_tips.append(tips)

            #Left/Right label pos
            if LABEL_VISIBLE:
                cv2.putText(canvas, h_type, (int(landmarks[0].x*w), int(landmarks[0].y*h)-20), 0, 0.6, (255,255,255), 2)

            if SKELETON_VISIBLE:
                for s, e in HAND_CONNECTIONS:
                    cv2.line(canvas, (int(landmarks[s].x*w), int(landmarks[s].y*h)), (int(landmarks[e].x*w), int(landmarks[e].y*h)), (180,180,180), 1)
                for lm in landmarks: cv2.circle(canvas, (int(lm.x*w), int(lm.y*h)), 3, (0,0,255), -1)

            #Interaction
            if selected_mode == 1:
                idx = np.array([landmarks[8].x * w, landmarks[8].y * h])
                pinch = math.hypot(landmarks[4].x-landmarks[8].x, landmarks[4].y-landmarks[8].y)
                if pinch < 0.07:
                    if grabbed_hand_id is None and math.hypot(idx[0]-cube_pos[0], idx[1]-cube_pos[1]) < 60: grabbed_hand_id = h_idx
                    if grabbed_hand_id == h_idx: cube_pos[:] = cube_pos * SMOOTHING + idx * (1 - SMOOTHING)
                elif grabbed_hand_id == h_idx: grabbed_hand_id = None
                
                if len(hands.hand_landmarks) == 2 and grabbed_hand_id is not None and h_idx != grabbed_hand_id:
                    v1 = np.array([landmarks[5].x-landmarks[0].x, landmarks[5].y-landmarks[0].y, landmarks[5].z-landmarks[0].z])
                    v2 = np.array([landmarks[17].x-landmarks[0].x, landmarks[17].y-landmarks[0].y, landmarks[17].z-landmarks[0].z])
                    norm = np.cross(v1, v2)
                    if np.linalg.norm(norm) > 1e-6:
                        norm /= np.linalg.norm(norm)
                        rotation_x = rotation_x * SMOOTHING + (math.asin(-norm[1])*1.8) * (1-SMOOTHING)
                        rotation_z = rotation_z * SMOOTHING + (math.atan2(norm[0], norm[2])*1.6) * (1-SMOOTHING)
            if WEB_HANDS_VISIBLE:
                if selected_mode == 2:
                    cx, cy = int(sum(landmarks[i].x for i in PALM_BASE_INDICES)/7 * w), int((sum(landmarks[i].y for i in PALM_BASE_INDICES)/7 + 0.1) * h)
                    poly = [[int(cx + math.cos(t)*random.uniform(10, 45)), int(cy + math.sin(t)*random.uniform(10, 45))] for t in np.linspace(0, 2*math.pi, 10)]
                    cv2.polylines(canvas, [np.array(poly, np.int32)], True, PALM_WEB_COLOR, 1)
                    for px, py in poly:
                        for j in FINGER_JOINTS:
                            if random.random() > 0.3: cv2.line(canvas, (px, py), (int(landmarks[j].x*w), int(landmarks[j].y*h)), PALM_WEB_COLOR, 1)
    if WEB_VISIBLE:
        if selected_mode == 2 and len(all_tips) == 2:
            for _ in range(40):
                if random.random() > 0.4:
                    p1, p2 = random.choice(all_tips[0]), random.choice(all_tips[1])
                    cv2.line(canvas, (p1[0]+random.randint(-5,5), p1[1]+random.randint(-5,5)), (p2[0]+random.randint(-5,5), p2[1]+random.randint(-5,5)), CROSS_HAND_COLOR, 1)

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