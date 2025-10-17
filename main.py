import cv2
import mediapipe as mp
import numpy as np
import os
import time
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True)
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1)
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,
        enable_segmentation=False)
    
    camera_id = int(os.getenv('CAMERA_ID', '0'))
    cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
    
    target_fps = 20
    frame_time = 1.0 / target_fps
    last_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # 虹彩の中心から視線方向を計算
            left_iris = face_landmarks.landmark[468]  # 左虹彩中心
            right_iris = face_landmarks.landmark[473]  # 右虹彩中心
            left_eye_center = face_landmarks.landmark[33]
            right_eye_center = face_landmarks.landmark[263]
            
            gaze_x = ((left_iris.x - left_eye_center.x) + (right_iris.x - right_eye_center.x)) / 2
            gaze_y = ((left_iris.y - left_eye_center.y) + (right_iris.y - right_eye_center.y)) / 2
            
            # 鼻、顎、左目、右目、左耳、右耳の6点を取得
            points_3d = np.array([
                [face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h, face_landmarks.landmark[1].z * w],  # 鼻
                [face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h, face_landmarks.landmark[152].z * w],  # 顎
                [face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h, face_landmarks.landmark[33].z * w],  # 左目
                [face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h, face_landmarks.landmark[263].z * w],  # 右目
                [face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h, face_landmarks.landmark[61].z * w],  # 左口
                [face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h, face_landmarks.landmark[291].z * w],  # 右口
            ], dtype=np.float64)
            
            points_2d = np.ascontiguousarray(points_3d[:, :2])
            
            # カメラ行列
            focal_length = w
            camera_matrix = np.array([[focal_length, 0, w / 2],
                                     [0, focal_length, h / 2],
                                     [0, 0, 1]], dtype=np.float64)
            
            # 3Dモデル座標
            model_points = np.array([
                [0.0, 0.0, 0.0],
                [0.0, -330.0, -65.0],
                [-225.0, 170.0, -135.0],
                [225.0, 170.0, -135.0],
                [-150.0, -150.0, -125.0],
                [150.0, -150.0, -125.0]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, points_2d, camera_matrix, dist_coeffs)
            
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = cv2.RQDecomp3x3(rotation_matrix)[0]
                pitch, yaw, roll = angles[0], angles[1], angles[2]
                x, y, z = translation_vector[0][0], translation_vector[1][0], translation_vector[2][0]
                
                cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"X: {x:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Y: {y:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Z: {z:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Gaze X: {gaze_x:.3f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(frame, f"Gaze Y: {gaze_y:.3f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        if pose_results.pose_landmarks:
            # 顔以外のポーズランドマークのみ描画
            pose_connections = frozenset([(c[0], c[1]) for c in mp_pose.POSE_CONNECTIONS 
                                         if c[0] >= 11 and c[1] >= 11])
            
            # 顔のランドマークポイントを非表示にするためのカスタムスタイル
            landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            
            # ランドマークを個別に描画（11番以降のみ）
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                if idx >= 11:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # 接続線を描画
            for connection in pose_connections:
                start_idx, end_idx = connection
                start = pose_results.pose_landmarks.landmark[start_idx]
                end = pose_results.pose_landmarks.landmark[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
        
        cv2.imshow('Head Pose', frame)
        
        # FPS制限
        elapsed = time.time() - last_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)
        last_time = time.time()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
