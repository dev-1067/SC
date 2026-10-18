"""
Smart Traffic Signal Control System (STCS)
Person 2 Deliverable: Vehicle Counting & Queue Estimator (vision_detector.py)
Covers Syllabus Module 1 (Imprecision tolerance & noisy sensing):
- Computer Vision Detection via OpenCV Background Subtraction (MOG2)
- Multi-lane Region of Interest (ROI) Masking
- Vehicle Centroid Tracking & Velocity Vector Estimation
- Lane Queue Extraction: Stationary Vehicles (Queue Length Q) vs. Approaching Vehicles (Arrival Rate V)
- Built-in Synthetic Camera Feed Generator for Standalone & Live Testing
"""

import math
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2


# ==============================================================================
# 1. VEHICLE CENTROID TRACKER
# ==============================================================================

class TrackedVehicle:
    """Represents a detected vehicle tracked across consecutive video frames."""
    def __init__(self, vehicle_id: int, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], lane_id: str):
        self.vehicle_id = vehicle_id
        self.centroid = centroid  # (x, y)
        self.bbox = bbox          # (x, y, w, h)
        self.lane_id = lane_id
        self.trajectory: List[Tuple[int, int]] = [centroid]
        self.speeds: List[float] = []
        self.stationary_frames = 0
        self.disappeared_frames = 0
        self.is_stationary = False

    def update(self, new_centroid: Tuple[int, int], new_bbox: Tuple[int, int, int, int], fps: float = 30.0):
        """Updates vehicle position, computes instantaneous velocity, and classifies queue state."""
        dx = new_centroid[0] - self.centroid[0]
        dy = new_centroid[1] - self.centroid[1]
        dist_px = math.hypot(dx, dy)
        speed_px_per_sec = dist_px * fps

        self.speeds.append(speed_px_per_sec)
        if len(self.speeds) > 10:
            self.speeds.pop(0)

        # Smooth velocity
        avg_speed = np.mean(self.speeds) if self.speeds else 0.0

        # Stationary threshold (e.g. less than 4 px/sec implies queue stop)
        if avg_speed < 4.0:
            self.stationary_frames += 1
            if self.stationary_frames >= 4:
                self.is_stationary = True
        else:
            self.stationary_frames = 0
            self.is_stationary = False

        self.centroid = new_centroid
        self.bbox = new_bbox
        self.trajectory.append(new_centroid)
        if len(self.trajectory) > 20:
            self.trajectory.pop(0)
        self.disappeared_frames = 0


# ==============================================================================
# 2. OPENCV VISION DETECTOR & QUEUE EXTRACTOR
# ==============================================================================

class VisionQueueDetector:
    """
    OpenCV-based Traffic Sensing & Queue Estimation Engine.
    Processes video feeds, performs background subtraction, segments lanes,
    and returns crisp lane counts (Stationary Queues and Arrival Rates).
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        history: int = 500,
        var_threshold: float = 16.0,
        min_contour_area: float = 50.0
    ):
        self.width = frame_width
        self.height = frame_height
        self.min_area = min_contour_area

        # Gaussian Mixture-based Background/Foreground Segmentation
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )

        # Morphological kernels for noise reduction and blob consolidation
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Lane Regions of Interest (ROIs): [x_min, y_min, x_max, y_max]
        # Configured for a standard 4-way intersection layout
        cx, cy = frame_width // 2, frame_height // 2
        lane_w = 60
        self.lane_rois = {
            "North": (cx - lane_w, 0, cx, cy - 35),
            "South": (cx, cy + 35, cx + lane_w, frame_height),
            "West":  (0, cy - lane_w, cx - 35, cy),
            "East":  (cx + 35, cy, frame_width, cy + lane_w)
        }

        self.tracked_vehicles: Dict[int, TrackedVehicle] = {}
        self.next_vehicle_id = 1
        self.max_disappeared = 15
        self.lane_crossings: Dict[str, List[float]] = {k: [] for k in self.lane_rois}
        self.last_timestamp = time.time()

    def _get_lane_for_point(self, pt: Tuple[int, int]) -> Optional[str]:
        """Maps an (x, y) point to its designated approach lane."""
        x, y = pt
        for lane, (x1, y1, x2, y2) in self.lane_rois.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return lane
        return None

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Executes vision detection pipeline:
        1. Background Subtraction
        2. Morphological Filtering (Closing & Dilating)
        3. Contour Extraction & Filtering
        4. Centroid Tracking
        5. Queue & Arrival Rate Computation
        """
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))

        current_time = time.time()
        dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        inst_fps = 1.0 / max(1e-4, dt) if dt > 0 else 30.0
        self.current_fps = 0.85 * getattr(self, "current_fps", min(inst_fps, 60.0)) + 0.15 * min(inst_fps, 60.0)

        # Step 1: Foreground mask
        fg_mask = self.bg_subtractor.apply(frame)

        # Step 2: Morphological filtering to bridge car outlines
        fg_closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)
        fg_cleaned = cv2.dilate(fg_closed, self.kernel_dilate, iterations=1)

        # Step 3: Find vehicle contours
        contours, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Aspect ratio filter for vehicle shapes
                aspect = float(w) / max(1, h)
                if 0.25 <= aspect <= 4.0 and (w >= 10 and h >= 10):
                    centroid = (int(x + w / 2), int(y + h / 2))
                    lane = self._get_lane_for_point(centroid)
                    if lane is not None:
                        current_detections.append((centroid, (x, y, w, h), lane))

        # Step 4: Tracking association (nearest neighbor match)
        updated_ids = set()
        for centroid, bbox, lane in current_detections:
            matched_id = None
            min_dist = 60.0  # max association distance in pixels

            for v_id, vehicle in self.tracked_vehicles.items():
                if v_id not in updated_ids and vehicle.lane_id == lane:
                    dist = math.hypot(centroid[0] - vehicle.centroid[0], centroid[1] - vehicle.centroid[1])
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = v_id

            if matched_id is not None:
                self.tracked_vehicles[matched_id].update(centroid, bbox, fps=self.current_fps)
                updated_ids.add(matched_id)
            else:
                # New vehicle entering lane
                new_veh = TrackedVehicle(self.next_vehicle_id, centroid, bbox, lane)
                self.tracked_vehicles[self.next_vehicle_id] = new_veh
                updated_ids.add(self.next_vehicle_id)
                self.next_vehicle_id += 1
                self.lane_crossings[lane].append(current_time)

        # Remove disappeared vehicles (allowing high tolerance for stationary queued cars)
        to_delete = []
        for v_id, vehicle in self.tracked_vehicles.items():
            if v_id not in updated_ids:
                vehicle.disappeared_frames += 1
                # If motion contours cease inside the lane approach, vehicle has stopped (queued at signal)
                vehicle.stationary_frames += 1
                if vehicle.stationary_frames >= 2:
                    vehicle.is_stationary = True
                    vehicle.speeds.append(0.0)
                    if len(vehicle.speeds) > 10:
                        vehicle.speeds.pop(0)

                # Stationary vehicles waiting at red light retained for up to 60s (1800 frames)
                allowance = self.max_disappeared * 60 if vehicle.is_stationary else self.max_disappeared
                if vehicle.disappeared_frames > allowance:
                    to_delete.append(v_id)
        for v_id in to_delete:
            del self.tracked_vehicles[v_id]

        # Step 5: Aggregate Lane Metrics
        lane_summary: Dict[str, Dict[str, Any]] = {}
        for lane in self.lane_rois:
            vehicles_in_lane = [v for v in self.tracked_vehicles.values() if v.lane_id == lane]
            stationary = [v for v in vehicles_in_lane if v.is_stationary]
            moving = [v for v in vehicles_in_lane if not v.is_stationary]

            # Calculate arrival rate (crossings in past 60s)
            recent_crossings = [t for t in self.lane_crossings[lane] if current_time - t <= 60.0]
            self.lane_crossings[lane] = recent_crossings
            arrival_rate_veh_min = float(len(recent_crossings))

            moving_speeds = [float(np.mean(v.speeds)) for v in moving if v.speeds and len(v.speeds) > 0]
            avg_spd = float(np.mean(moving_speeds)) if len(moving_speeds) > 0 else 0.0

            # Estimate waiting time for stationary vehicles
            est_wait_time = float(min(180.0, len(stationary) * 6.5))

            lane_summary[lane] = {
                "queue_length": len(stationary),
                "moving_count": len(moving),
                "total_count": len(vehicles_in_lane),
                "arrival_rate": round(arrival_rate_veh_min, 2),
                "wait_time": round(est_wait_time, 1),
                "average_speed_px_s": round(avg_spd, 1)
            }

        # Format paired phases (NS and EW) for Person 1 / Person 3 integration
        phases_payload = {
            0: {
                "queue_length": lane_summary["North"]["queue_length"] + lane_summary["South"]["queue_length"],
                "wait_time": max(lane_summary["North"]["wait_time"], lane_summary["South"]["wait_time"]),
                "arrival_rate": lane_summary["North"]["arrival_rate"] + lane_summary["South"]["arrival_rate"],
                "name": "North-South"
            },
            1: {
                "queue_length": lane_summary["East"]["queue_length"] + lane_summary["West"]["queue_length"],
                "wait_time": max(lane_summary["East"]["wait_time"], lane_summary["West"]["wait_time"]),
                "arrival_rate": lane_summary["East"]["arrival_rate"] + lane_summary["West"]["arrival_rate"],
                "name": "East-West"
            }
        }

        return {
            "lane_metrics": lane_summary,
            "phases_payload": phases_payload,
            "tracked_count": len(self.tracked_vehicles),
            "fps": round(self.current_fps, 1),
            "timestamp": round(current_time, 3)
        }

    def render_annotated_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draws lane bounding boxes, vehicle detections, and metrics HUD onto frame."""
        annotated = frame.copy()

        # Draw Lane ROIs
        for lane, (x1, y1, x2, y2) in self.lane_rois.items():
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 80, 80), 1)
            cv2.putText(annotated, lane, (x1 + 5, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw Vehicles
        for v in self.tracked_vehicles.values():
            x, y, w, h = v.bbox
            color = (0, 0, 255) if v.is_stationary else (0, 255, 0)
            status_text = "QUEUED" if v.is_stationary else "MOVING"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.circle(annotated, v.centroid, 3, color, -1)
            cv2.putText(annotated, f"ID:{v.vehicle_id} {status_text}", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        return annotated


# ==============================================================================
# 3. SYNTHETIC JUNCTION CAMERA FEED GENERATOR
# ==============================================================================

class SyntheticTrafficCamera:
    """
    Renders realistic synthetic 4-way crossroad video frames with asphalt,
    lane markings, and animated moving/stopping cars to test vision detection live.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.cx = width // 2
        self.cy = height // 2
        self.cars = []
        self._init_cars()

    def _init_cars(self):
        """Seed initial traffic."""
        # NS cars (moving south)
        for i in range(4):
            self.cars.append({
                "x": self.cx - 30, "y": 30 + i * 50, "vx": 0.0, "vy": 2.0,
                "w": 24, "h": 36, "color": (180, 40, 40), "stopped": False, "lane": "North"
            })
        # EW cars (moving east)
        for i in range(5):
            self.cars.append({
                "x": 40 + i * 55, "y": self.cy - 30, "vx": 2.2, "vy": 0.0,
                "w": 36, "h": 24, "color": (40, 180, 40), "stopped": False, "lane": "West"
            })

    def update_physics(self, signal_ns: str = "GREEN", signal_ew: str = "RED"):
        """Advances synthetic car positions respecting traffic lights and gap spacing."""
        stop_y_north = self.cy - 50
        stop_x_west = self.cx - 50

        # Sort North cars by y descending (lead car closest to stop line first)
        north_cars = [c for c in self.cars if c["lane"] == "North"]
        north_cars.sort(key=lambda c: c["y"], reverse=True)

        for i, car in enumerate(north_cars):
            dist_to_stop = stop_y_north - (car["y"] + car["h"])
            should_stop = False

            if signal_ns in ("RED", "YELLOW") and 0 <= dist_to_stop <= 45:
                should_stop = True

            # Maintain buffer behind stopped lead car
            if i > 0:
                lead = north_cars[i - 1]
                gap_to_lead = lead["y"] - (car["y"] + car["h"])
                if lead["stopped"] and 0 <= gap_to_lead <= 18:
                    should_stop = True
                elif gap_to_lead < 12:
                    should_stop = True

            car["stopped"] = should_stop
            if not car["stopped"]:
                car["y"] += car["vy"]
            if car["y"] > self.height:
                car["y"] = -40

        # Sort West cars by x descending (lead car closest to stop line first)
        west_cars = [c for c in self.cars if c["lane"] == "West"]
        west_cars.sort(key=lambda c: c["x"], reverse=True)

        for i, car in enumerate(west_cars):
            dist_to_stop = stop_x_west - (car["x"] + car["w"])
            should_stop = False

            if signal_ew in ("RED", "YELLOW") and 0 <= dist_to_stop <= 45:
                should_stop = True

            # Maintain buffer behind stopped lead car
            if i > 0:
                lead = west_cars[i - 1]
                gap_to_lead = lead["x"] - (car["x"] + car["w"])
                if lead["stopped"] and 0 <= gap_to_lead <= 18:
                    should_stop = True
                elif gap_to_lead < 12:
                    should_stop = True

            car["stopped"] = should_stop
            if not car["stopped"]:
                car["x"] += car["vx"]
            if car["x"] > self.width:
                car["x"] = -40

    def generate_frame(self, signal_ns: str = "GREEN", signal_ew: str = "RED") -> np.ndarray:
        """Renders complete 640x480 RGB frame with road and vehicles."""
        self.update_physics(signal_ns, signal_ew)

        # 1. Background Grass
        frame = np.full((self.height, self.width, 3), (35, 75, 40), dtype=np.uint8)

        # 2. Road Asphalt (Dark Gray)
        road_w = 120
        # North-South Road
        cv2.rectangle(frame, (self.cx - road_w // 2, 0), (self.cx + road_w // 2, self.height), (45, 45, 45), -1)
        # East-West Road
        cv2.rectangle(frame, (0, self.cy - road_w // 2), (self.width, self.cy + road_w // 2), (45, 45, 45), -1)

        # 3. Yellow Center Divider Lines
        # NS dashed
        for y in range(0, self.height, 30):
            if not (self.cy - road_w // 2 < y < self.cy + road_w // 2):
                cv2.line(frame, (self.cx, y), (self.cx, y + 15), (0, 215, 255), 2)
        # EW dashed
        for x in range(0, self.width, 30):
            if not (self.cx - road_w // 2 < x < self.cx + road_w // 2):
                cv2.line(frame, (x, self.cy), (x + 15, self.cy), (0, 215, 255), 2)

        # 4. White Stop Lines
        cv2.line(frame, (self.cx - road_w // 2, self.cy - road_w // 2), (self.cx, self.cy - road_w // 2), (255, 255, 255), 3)
        cv2.line(frame, (self.cx - road_w // 2, self.cy - road_w // 2), (self.cx - road_w // 2, self.cy), (255, 255, 255), 3)

        # 5. Draw Vehicles
        for car in self.cars:
            x, y, w, h = int(car["x"]), int(car["y"]), int(car["w"]), int(car["h"])
            # Car body
            cv2.rectangle(frame, (x, y), (x + w, y + h), car["color"], -1)
            # Windshield
            cv2.rectangle(frame, (x + 2, y + 2), (x + w - 2, y + h - 2), (220, 220, 230), 1)
            # Brake lights if stopped
            if car["stopped"]:
                cv2.circle(frame, (x + 3, y + h - 2), 2, (0, 0, 255), -1)
                cv2.circle(frame, (x + w - 3, y + h - 2), 2, (0, 0, 255), -1)

        # 6. Add subtle simulated camera sensor noise
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame


if __name__ == "__main__":
    detector = VisionQueueDetector()
    camera = SyntheticTrafficCamera()

    print("Running OpenCV Vision Queue Extractor on Synthetic Camera Feed...")
    # Warm up background subtractor with 30 frames
    for _ in range(30):
        raw_frame = camera.generate_frame("GREEN", "RED")
        metrics = detector.process_frame(raw_frame)

    # Process sample active frame
    frame = camera.generate_frame("RED", "GREEN")
    metrics = detector.process_frame(frame)
    annotated = detector.render_annotated_frame(frame)

    print("\nVision Detection Output:")
    print(f"  Tracked Vehicles: {metrics['tracked_count']}")
    print(f"  Processing FPS:   {metrics['fps']}")
    print("\nLane Metrics Extracted by Computer Vision:")
    for lane, data in metrics["lane_metrics"].items():
        print(f"  [{lane:<5}] Queue: {data['queue_length']:2d} cars | Moving: {data['moving_count']:2d} | "
              f"Arrival Rate: {data['arrival_rate']} veh/min | Avg Speed: {data['average_speed_px_s']} px/s")

    print("\nPhases Payload Formatted for Person 1 (FLC):")
    print(f"  Phase 0 (North-South): Queue={metrics['phases_payload'][0]['queue_length']}, Wait={metrics['phases_payload'][0]['wait_time']}s")
    print(f"  Phase 1 (East-West):   Queue={metrics['phases_payload'][1]['queue_length']}, Wait={metrics['phases_payload'][1]['wait_time']}s")
