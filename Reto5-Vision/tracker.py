import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class Pollito:
    def __init__(self, initial_pos, tiempo):
        self.trajectory = [initial_pos]
        self.timestamps = [tiempo]
        self.lost_frames = 0
        self.active = True
        self.last_velocity = np.array([0.0, 0.0])
        self.predicted_pos = initial_pos

    def update(self, pos, tiempo):
        if len(self.trajectory) > 0:
            self.last_velocity = pos - self.trajectory[-1]
        self.trajectory.append(pos)
        self.timestamps.append(tiempo)
        self.lost_frames = 0
        self.active = True
        self.predicted_pos = pos + self.last_velocity

    def predict_next(self):
        return self.trajectory[-1] + self.last_velocity

    def mark_lost(self):
        self.lost_frames += 1
        if self.lost_frames > 60:
            self.active = False
        else:
            self.predicted_pos += self.last_velocity

    def get_direction(self):
        if len(self.trajectory) >= 2:
            return self.trajectory[-1] - self.trajectory[-2]
        else:
            return np.array([0.0, 0.0])

class MultiPollitoTrackerHybrid:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.max_lost = 60

    def update(self, detections, tiempo):
        active_ids = [tid for tid, t in self.trackers.items() if t.active]
        preds = [self.trackers[tid].predict_next() for tid in active_ids]

        if len(detections) == 0:
            for tid in active_ids:
                self.trackers[tid].mark_lost()
            return

        if len(preds) == 0:
            for d in detections:
                if not self.try_reassign(d, tiempo):
                    self.trackers[self.next_id] = Pollito(d, tiempo)
                    self.next_id += 1
            return

        cost_matrix = np.zeros((len(preds), len(detections)))
        for i, p in enumerate(preds):
            for j, d in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(p - d)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 100:  # umbral amplio para errores de YOLO
                tid = active_ids[i]
                self.trackers[tid].update(detections[j], tiempo)
                assigned.add(j)
            else:
                self.trackers[active_ids[i]].mark_lost()

        for j, d in enumerate(detections):
            if j not in assigned:
                if not self.try_reassign(d, tiempo):
                    self.trackers[self.next_id] = Pollito(d, tiempo)
                    self.next_id += 1

    def try_reassign(self, detection, tiempo):
        for tid, t in self.trackers.items():
            if not t.active and t.lost_frames < self.max_lost:
                last_pos = t.trajectory[-1]
                predicted = t.predict_next()
                if np.linalg.norm(predicted - detection) < 50:
                    t.update(detection, tiempo)
                    print(f"♻️ Pollito {tid} reactivado con predicción visual")
                    return True
        return False

    def get_trajectories(self):
        result = {}
        for tid, t in self.trackers.items():
            if len(t.trajectory) > 3:
                result[f"pollito_{tid}"] = {
                    "x": [p[0] for p in t.trajectory],
                    "y": [p[1] for p in t.trajectory],
                    "t": t.timestamps
                }
        return result
