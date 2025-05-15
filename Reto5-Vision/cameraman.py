import numpy as np
import cv2

class CameraMotionEstimator:
    def __init__(self):
        self.positions = [(0.0, 0.0)]
        self.angles = [0.0]  # en grados
        self.prev_kp = None
        self.prev_desc = None
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_keypoints(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(gray, None)

    def update(self, current_frame):
        kp, desc = self.extract_keypoints(current_frame)
        if self.prev_kp is None or self.prev_desc is None:
            self.prev_kp = kp
            self.prev_desc = desc
            return

        matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > 10:
            pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts_curr = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, _ = cv2.estimateAffinePartial2D(pts_prev, pts_curr)

            if M is not None:
                dx = M[0, 2]
                dy = M[1, 2]
                dtheta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

                last_x, last_y = self.positions[-1]
                self.positions.append((last_x + dx, last_y + dy))
                self.angles.append(self.angles[-1] + dtheta)

        self.prev_kp = kp
        self.prev_desc = desc

    def get_trajectory(self):
        return np.array(self.positions), np.array(self.angles)
