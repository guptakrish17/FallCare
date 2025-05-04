import cv2
import numpy as np
import time
import mediapipe as mp
import requests
import threading

# MediaPipe initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, data=data)
        print(f"Telegram Response: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram Error: {e}")
        return False

class FallDetector:
    def __init__(self, telegram_token, telegram_chat_id):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.last_alert_time = 0
        self.alert_cooldown = 60  # in seconds
        
        # Rebalanced thresholds
        self.vertical_ratio_threshold = 0.85  # Adjusted to prevent standing misclassification
        self.motion_threshold = 0.0001         # Slightly more tolerant
        self.velocity_threshold = 0.00002      # Moderate sensitivity
        self.fall_frames_threshold = 3
        self.recovery_frames_threshold = 3

        self.fall_counter = 0
        self.recovery_counter = 0
        self.previous_landmarks = None
        self.fall_status = "NORMAL"

    def calculate_vertical_ratio(self, landmarks):
        x_coords = [l.x for l in landmarks]
        y_coords = [l.y for l in landmarks]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return height / width if width > 0 else 1.0

    def calculate_motion(self, current, previous):
        if not current or not previous:
            return 0.0
        total_motion = sum(
            np.sqrt((c.x - p.x)**2 + (c.y - p.y)**2)
            for c, p in zip(current, previous)
        )
        return total_motion / len(current)

    def calculate_vertical_velocity(self, current, previous):
        torso_indices = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP
        ]
        curr_y = np.mean([current[i].y for i in torso_indices])
        prev_y = np.mean([previous[i].y for i in torso_indices])
        return curr_y - prev_y

    def detect_fall(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        output = frame.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            vertical_ratio = self.calculate_vertical_ratio(landmarks)
            motion = self.calculate_motion(landmarks, self.previous_landmarks) if self.previous_landmarks else 0.0
            velocity = self.calculate_vertical_velocity(landmarks, self.previous_landmarks) if self.previous_landmarks else 0.0
            self.previous_landmarks = landmarks

            print(f"Vertical Ratio: {vertical_ratio:.3f}, Motion: {motion:.6f}, Velocity: {velocity:.6f}")

            if self.fall_status == "NORMAL":
                if vertical_ratio < self.vertical_ratio_threshold and motion > self.motion_threshold and velocity > self.velocity_threshold:
                    self.fall_counter += 1
                    print(f"Potential fall: {self.fall_counter}/{self.fall_frames_threshold}")
                    if self.fall_counter >= self.fall_frames_threshold:
                        self.fall_status = "FALL_DETECTED"
                        self.fall_counter = 0
                        self.send_alert()
                else:
                    self.fall_counter = 0

            elif self.fall_status == "FALL_DETECTED":
                if vertical_ratio > self.vertical_ratio_threshold and motion < self.motion_threshold:
                    self.recovery_counter += 1
                    if self.recovery_counter >= self.recovery_frames_threshold:
                        self.fall_status = "NORMAL"
                        self.recovery_counter = 0
                        print("Recovered from fall.")
                else:
                    self.recovery_counter = 0

            self.draw_status(output, vertical_ratio, motion, velocity)
        else:
            cv2.putText(output, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return output

    def draw_status(self, frame, ratio, motion, velocity):
        color = (0, 0, 255) if self.fall_status == "FALL_DETECTED" else (0, 255, 0)
        text = "Status: FALL DETECTED!" if self.fall_status == "FALL_DETECTED" else "Status: Normal"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Ratio: {ratio:.2f} | Motion: {motion:.6f} | Velocity: {velocity:.6f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def send_alert(self):
        if time.time() - self.last_alert_time < self.alert_cooldown:
            print("Alert skipped due to cooldown.")
            return
        self.last_alert_time = time.time()
        message = f"⚠️ EMERGENCY: Fall detected at {time.strftime('%Y-%m-%d %H:%M:%S')}."
        threading.Thread(target=self.send_telegram_alert, args=(message,)).start()

    def send_telegram_alert(self, message):
        print("Sending alert...")
        if self.telegram_token and self.telegram_chat_id:
            result = send_telegram_message(self.telegram_token, self.telegram_chat_id, message)
            print("Telegram send success!" if result else "Telegram send failed.")
        else:
            print("Missing Telegram token/chat_id!")

# --------------- MAIN LOOP ---------------
if __name__ == "__main__":
    TELEGRAM_TOKEN = "8195982369:AAH5c2Peh5_jaKtYBx13IVJdWOGJcdk29-U"
    TELEGRAM_CHAT_ID = "your_chat_id_here"

    cap = cv2.VideoCapture(0)
    detector = FallDetector(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

    print("Press ESC or Q to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detector.detect_fall(frame)
        cv2.imshow("Fall Detection", output_frame)

        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):  # ESC or 'q'
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
