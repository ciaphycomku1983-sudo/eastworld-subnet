# The MIT License (MIT)
# Copyright © 2025 Eastworld AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import collections
import json
import math
import os
import pickle
import traceback

import bittensor as bt
import gtsam
import numpy as np
from gtsam import symbol, symbolChr, symbolIndex

from eastworld.miner.slam.grid import OccupancyGridMap

SENSOR_MAX_RANGE = 50.0


class ISAM2:
    direction_to_angle = {
        "north": np.pi / 2,  # 90 degrees - up
        "northeast": np.pi / 4,  # 45 degrees - upper right
        "east": 0,  # 0 degrees - right
        "southeast": -np.pi / 4,  # -45 degrees - lower right
        "south": -np.pi / 2,  # -90 degrees - down
        "southwest": -3 * np.pi / 4,  # -135 degrees - lower left
        "west": np.pi,  # 180 degrees or -180 degrees - left
        "northwest": 3 * np.pi / 4,  # 135 degrees - upper left
    }

    def __init__(
        self,
        load_data: bool = False,
        data_dir: str = "slam_data",
        save_interval: int = 5,
    ):
        # Create data save directory
        self.data_dir = data_dir
        self.save_interval = save_interval
        os.makedirs(self.data_dir, exist_ok=True)

        try:
            self.isam_params = gtsam.ISAM2Params()
            self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.01, 0.01, 0.01])
            self.between_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
                [0.1, 0.1, 0.005]
            )
            self.bearing_range_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.05])

            noise = gtsam.noiseModel.Diagonal.Sigmas([0.3, 0.3, 0.005])
            huber = gtsam.noiseModel.mEstimator.Huber(1.345)
            self.loop_noise = gtsam.noiseModel.Robust.Create(huber, noise)

            self.max_graph_size = 100_000 * 9
            self.max_segment_poses = self.max_graph_size // 10

            if load_data:
                self.load(self.data_dir)
            else:
                self._reset_isam()
        except Exception as e:
            bt.logging.error(f"GTSAM initialization error: {e}")
            traceback.print_exc()
            raise e

    def _reset_isam(self):
        self.grid_map = OccupancyGridMap(width=1000, height=1000, resolution=2)

        self.segments: collections.deque[
            tuple[gtsam.NonlinearFactorGraph, gtsam.Values]
        ] = collections.deque(maxlen=10)
        self.segments.append((gtsam.NonlinearFactorGraph(), gtsam.Values()))

        self.isam = gtsam.ISAM2(self.isam_params)
        self.pose_index = 0
        self.landmark_index = 0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_pose = gtsam.Pose2(
            self.current_x, self.current_y, self.current_theta
        )

        current_key = symbol("x", self.pose_index)
        values = gtsam.Values()
        graph = gtsam.NonlinearFactorGraph()
        values.insert(current_key, self.current_pose)
        self.segments[-1][1].insert(current_key, self.current_pose)
        factor = gtsam.PriorFactorPose2(
            current_key, self.current_pose, self.prior_noise
        )
        graph.add(factor)
        self.segments[-1][0].add(factor)
        self.isam.update(graph, values)

        self.sensor_data: dict[int, dict] = {}
        self.key_frames: dict[tuple, tuple[int, gtsam.Pose2]] = {}
        self.last_key_frame = self.pose_index
        self.last_loop_frame = self.pose_index

        self._rebuild_grid_map()

    def _new_segment(self):
        if len(self.segments) >= self.segments.maxlen:
            self.segments.popleft()
        self.segments.append((gtsam.NonlinearFactorGraph(), gtsam.Values()))

    def _reset_graph(self):
        bt.logging.warning(f"Reset GTSAM graph structure: {self.current_pose}")

        self.isam = gtsam.ISAM2(self.isam_params)
        all_values = gtsam.Values()
        while len(self.segments) > self.segments.maxlen // 2:
            discard_graph, discard_values = self.segments.popleft()
            all_values.insert(discard_values)

        prior_set = False
        for graph, values in self.segments:
            result = self.isam.calculateEstimate()
            for i in range(graph.size()):
                factor = graph.at(i)
                for key in factor.keys():
                    if not values.exists(key) and (result and not result.exists(key)):
                        bt.logging.info(
                            f"Adding missing pose: {i} {chr(symbolChr(key))} {symbolIndex(key)}"
                        )
                        pose = all_values.atPose2(key)
                        values.insert(key, pose)
                        if not prior_set and chr(symbolChr(key)) == "x":
                            bt.logging.info(
                                f"Adding prior factor: {i} {chr(symbolChr(key))} {symbolIndex(key)}"
                            )
                            factor = gtsam.PriorFactorPose2(key, pose, self.prior_noise)
                            graph.add(factor)
                            prior_set = True

            self.isam.update(graph, values)
            bt.logging.info(f"Adding graph structure: {graph.size()} {values.size()}")

        self.segments.append((gtsam.NonlinearFactorGraph(), gtsam.Values()))

    def _check_key_frame(
        self, lidar_data: dict[str, float]
    ) -> tuple[int | None, gtsam.Pose2 | None]:
        # Sensor data precision is 1 meter, used as signature directly
        signature = tuple(
            lidar_data.get(d, SENSOR_MAX_RANGE + 1)
            for d in self.direction_to_angle.keys()
        )

        match_frame_id, match_frame_pose = self.key_frames.get(signature, (None, None))
        if match_frame_id and match_frame_pose:
            # Calculate distance between current pose and key frame pose
            dx = self.current_pose.x() - match_frame_pose.x()
            dy = self.current_pose.y() - match_frame_pose.y()
            distance = math.sqrt(dx**2 + dy**2)
            bt.logging.info(
                f"Matched key frame: {self.pose_index} > {match_frame_id} {distance}"
            )
            if distance < 100:
                # Key frame pose may have been expired and cleared
                if self.isam.calculateBestEstimate().exists(
                    symbol("x", match_frame_id)
                ):
                    return match_frame_id, match_frame_pose
                else:
                    bt.logging.debug(
                        f"Key frame pose has been cleared {match_frame_id}"
                    )
                    self.key_frames.pop(signature)
            else:
                return None, None

        if abs(self.pose_index - self.last_key_frame) < 10:
            return None, None

        # Calculate signature gradient, ignore frames with unclear features
        shifted = np.roll(signature, -1)
        gradient = np.abs(shifted - signature)
        gradient_score = np.sum(gradient)
        bt.logging.info(f"Signature gradient: {gradient_score} {signature}")
        if gradient_score < 50:
            return None, None

        self.key_frames[tuple(signature)] = (self.pose_index, self.current_pose)
        self.last_key_frame = self.pose_index
        return None, None

    def get_current_pose(self) -> tuple[float, float, float]:
        return self.current_x, self.current_y, self.current_theta

    def run_iteration(
        self, lidar_data: dict[str, float], odometry: float, odometry_direction: str
    ):
        try:
            if odometry <= 0:
                return

            # Update pose based on odometry
            angle_rad = self.direction_to_angle[odometry_direction]
            # Calculate displacement increment
            dx = odometry * math.cos(angle_rad)
            dy = odometry * math.sin(angle_rad)

            self.current_x += dx
            self.current_y += dy

            try:
                values = gtsam.Values()
                graph = gtsam.NonlinearFactorGraph()

                # Create incremental pose
                delta_pose = gtsam.Pose2(
                    dx, dy, 0
                )  # Assume only translation, no rotation

                # Update GTSAM pose
                self.current_pose = self.current_pose.compose(delta_pose)

                # Add new pose node
                prev_key = symbol("x", self.pose_index)
                current_key = symbol("x", self.pose_index + 1)
                self.pose_index += 1
                # Add between-pose factor
                between_factor = gtsam.BetweenFactorPose2(
                    prev_key, current_key, delta_pose, self.between_pose_noise
                )
                values.insert(current_key, self.current_pose)
                self.segments[-1][1].insert(current_key, self.current_pose)
                graph.add(between_factor)
                self.segments[-1][0].add(between_factor)

                # Process lidar data, add bearing and range factors
                # Keep a local copy and used for map rebuilding
                self.sensor_data[self.pose_index] = dict(lidar_data)
                for direction, distance in lidar_data.items():
                    try:
                        # Calculate landmark position in global coordinate system based on bearing and distance
                        bearing_angle = self.direction_to_angle[direction]
                        landmark_x = self.current_x + distance * math.cos(bearing_angle)
                        landmark_y = self.current_y + distance * math.sin(bearing_angle)

                        landmark_key = symbol("l", self.landmark_index)
                        point2 = gtsam.Point2(landmark_x, landmark_y)
                        values.insert(landmark_key, point2)
                        self.segments[-1][1].insert(landmark_key, point2)
                        self.landmark_index += 1

                        # Add bearing and range factor
                        bearing = gtsam.Rot2(bearing_angle)
                        bearing_range_factor = gtsam.BearingRangeFactor2D(
                            current_key,
                            landmark_key,
                            bearing,
                            distance,
                            self.bearing_range_noise,
                        )
                        graph.add(bearing_range_factor)
                        self.segments[-1][0].add(bearing_range_factor)
                    except Exception as e:
                        bt.logging.error(
                            f"Failed to process lidar measurement (direction: {direction}): {e}"
                        )

                close_loop = False
                key_frame_id, key_frame_pose = self._check_key_frame(lidar_data)
                if key_frame_id and key_frame_pose:
                    if self.pose_index - self.last_loop_frame > 10:
                        bt.logging.info(
                            f"Loop closure detected: {key_frame_id} {key_frame_pose}"
                        )
                        # Add loop closure factor
                        key_frame_symbol = symbol("x", key_frame_id)
                        if self.isam.calculateBestEstimate().exists(key_frame_symbol):
                            loop_factor = gtsam.BetweenFactorPose2(
                                current_key,
                                symbol("x", key_frame_id),
                                gtsam.Pose2(0.01, 0.01, 0),
                                self.loop_noise,
                            )
                            graph.add(loop_factor)
                            self.segments[-1][0].add(loop_factor)
                            self.last_loop_frame = self.pose_index
                            close_loop = True
                        else:
                            bt.logging.warning(
                                "Loop closure pose does not exist in optimization result"
                            )

                try:
                    self.isam.update(graph, values)
                    result: gtsam.Values = self.isam.calculateEstimate()

                    if result.exists(current_key):
                        self.current_pose = result.atPose2(current_key)
                        self.current_x = self.current_pose.x()
                        self.current_y = self.current_pose.y()
                        self.current_theta = self.current_pose.theta()
                        bt.logging.info(f"Current pose: {self.current_pose}")
                    else:
                        bt.logging.warning(
                            "Current pose does not exist in optimization result"
                        )
                except Exception as e:
                    bt.logging.error(f"Optimization failed: {e}")
                    traceback.print_exc()
                    self._reset_isam()

                if close_loop:
                    bt.logging.info(
                        ">>> Rebuilding map after loop closure optimization"
                    )
                    self._rebuild_grid_map()
                    self._reanchor_grid_map_nodes()

                if result.size() > self.max_graph_size:
                    bt.logging.warning(
                        f"GTSAM graph structure too large {result.size()}, simplifying"
                    )
                    self._reset_graph()

                if self.segments[-1][0].size() > self.max_segment_poses:
                    bt.logging.warning("Creating new window for global pose data")
                    self._new_segment()
            except Exception as e:
                bt.logging.error(f"GTSAM processing failed: {e}")
                traceback.print_exc()

            # Update grid map
            self._update_grid_map(self.current_pose, lidar_data)

            # Save map and trajectory periodically
            if self.pose_index and self.pose_index % self.save_interval == 0:
                self.save()
        except Exception as e:
            bt.logging.error(f"SLAM iteration error: {e}")
            traceback.print_exc()

    def _rebuild_grid_map(self):
        self.grid_map.reset()
        result = self.isam.calculateEstimate()
        for i in range(1, self.pose_index + 1):
            key = symbol("x", i)
            sensor_data = self.sensor_data.get(i, None)
            if sensor_data and result.exists(key):
                pose = result.atPose2(key)
                self._update_grid_map(pose, sensor_data)
        self.grid_map.justify_map()

    def _reanchor_grid_map_nodes(self):
        """Reanchor grid map nodes to the latest isam estimate"""
        result = self.isam.calculateEstimate()
        for node_id, (
            node_pid,
            node_x,
            node_y,
            node_desc,
        ) in self.grid_map.nav_nodes.items():
            node_key = symbol("x", node_pid)
            if result.exists(node_key):
                pose = result.atPose2(node_key)
                self.grid_map.nav_nodes[node_id] = (
                    node_pid,
                    pose.x(),
                    pose.y(),
                    node_desc,
                )

    def _update_grid_map(self, pose: gtsam.Pose2, lidar_data: dict[str, float]):
        """Update occupancy grid map using sensor data"""
        try:
            x, y, theta = pose.x(), pose.y(), pose.theta()

            # Convert world coordinates to grid coordinates
            grid_x, grid_y = self.grid_map.world_to_grid(x, y)

            # If current grid coordinates are close to map boundary, readjust the map
            if (
                grid_x < self.grid_map.width // 10
                or grid_x >= self.grid_map.width // 10 * 9
                or grid_y < self.grid_map.height // 10
                or grid_y >= self.grid_map.height // 10 * 9
            ):
                self.grid_map.justify_map(factor=1.4)

            # Update current position as free
            self.grid_map.update_cell(grid_x, grid_y, False)

            # Process lidar data
            for direction, distance in lidar_data.items():
                # Calculate obstacle position in world coordinates
                angle_rad = self.direction_to_angle[direction] + theta

                # Interpolation to accelerate mapping
                interpolates = [0]
                t = np.arctan(2.0 / distance)
                if t > 0.09:  # 5 degrees
                    interpolates.extend(np.arange(0.09, t, 0.09))
                    interpolates.extend(-np.arange(0.09, t, 0.09))

                for offset in interpolates:
                    angle = angle_rad + offset
                    angle = math.atan2(math.sin(angle), math.cos(angle))

                    obstacle_x = x + distance * math.cos(angle)
                    obstacle_y = y + distance * math.sin(angle)

                    # Convert to grid coordinates
                    obstacle_grid_x, obstacle_grid_y = self.grid_map.world_to_grid(
                        obstacle_x, obstacle_y
                    )

                    # Update obstacle position as occupied
                    if distance <= SENSOR_MAX_RANGE:
                        self.grid_map.update_cell(
                            obstacle_grid_x, obstacle_grid_y, True
                        )

                    # Update points along the ray (Bresenham algorithm)
                    self._update_ray(grid_x, grid_y, obstacle_grid_x, obstacle_grid_y)
        except Exception as e:
            print(f"Error updating grid map: {e}")
            traceback.print_exc()

    def _update_ray(self, x0, y0, x1, y1):
        """Update cells along a ray using Bresenham algorithm"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while x0 != x1 or y0 != y1:
            self.grid_map.update_cell(x0, y0, False)

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def save(self, save_path: str = "slam_data"):
        """Save SLAM data"""
        try:
            # Save map data
            with open(os.path.join(save_path, "map.pkl"), "wb") as f:
                pickle.dump(self.grid_map, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save trajectory data
            result: gtsam.Values = self.isam.calculateEstimate()
            trajectory = []
            for key in result.keys():
                if chr(symbolChr(key)) == "x":
                    pose = result.atPose2(key)
                    trajectory.append(pose)
            with open(os.path.join(save_path, "trajectory.pkl"), "wb") as f:
                pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save graph structure
            graphs = [g for g, _ in self.segments]
            with open(os.path.join(save_path, "graphs.pkl"), "wb") as f:
                pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
            values = [v for _, v in self.segments]
            with open(os.path.join(save_path, "values.pkl"), "wb") as f:
                pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            metadata = {
                "pose_index": self.pose_index,
                "landmark_index": self.landmark_index,
                "x": self.current_x,
                "y": self.current_y,
                "theta": self.current_theta,
            }
            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)

            # Save key frame data
            with open(os.path.join(save_path, "key_frames.pkl"), "wb") as f:
                pickle.dump(self.key_frames, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save sensor data
            with open(os.path.join(save_path, "sensor_data.pkl"), "wb") as f:
                pickle.dump(self.sensor_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            bt.logging.warning(f"GTSAM data saved successfully #{self.pose_index}")
        except Exception as e:
            print(f"Error saving SLAM data: {e}")

    def load(self, load_path: str = "slam_data"):
        # Restore map data
        with open(os.path.join(load_path, "map.pkl"), "rb") as f:
            self.grid_map = pickle.load(f)

        # Restore graph structure
        with open(os.path.join(load_path, "graphs.pkl"), "rb") as f:
            graphs: list[gtsam.NonlinearFactorGraph] = pickle.load(f)
        with open(os.path.join(load_path, "values.pkl"), "rb") as f:
            values: list[gtsam.Values] = pickle.load(f)
        self.segments = collections.deque(maxlen=10)
        self.isam = gtsam.ISAM2(self.isam_params)
        for graph, value in zip(graphs, values):
            self.segments.append((graph, value))
            self.isam.update(graph, value)

        with open(os.path.join(load_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.pose_index = metadata["pose_index"]
            self.landmark_index = metadata["landmark_index"]
            self.current_x = metadata["x"]
            self.current_y = metadata["y"]
            self.current_theta = metadata["theta"]
            self.current_pose = gtsam.Pose2(
                self.current_x, self.current_y, self.current_theta
            )

        # Restore key frame data
        with open(os.path.join(load_path, "key_frames.pkl"), "rb") as f:
            self.key_frames = pickle.load(f)
            self.last_key_frame = self.pose_index
            self.last_loop_frame = self.pose_index

        # Restore sensor data
        with open(os.path.join(load_path, "sensor_data.pkl"), "rb") as f:
            self.sensor_data = pickle.load(f)

        bt.logging.warning(f"GTSAM data loaded successfully #{self.pose_index}")
