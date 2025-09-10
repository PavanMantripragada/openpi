import numpy as np
import cv2
import pyzed.sl as sl
import glob
import cv2
import sys
from PIL import Image
import os

class ZEDCamera:
    def __init__(self, resolution=sl.RESOLUTION.HD2K, fps=15):
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = fps
        init_params.coordinate_units = sl.UNIT.MILLIMETER


        init_params.sdk_verbose = 1
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"[ZED ERROR] {status}")
            raise Exception(f"Failed to open ZED camera: {status}")
        
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()

    def get_depth_image(self):
        """Grabs a depth image and returns it as a NumPy array in millimeters."""
        depth = sl.Mat()
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depth in millimeters
            depth_np = depth.get_data().astype(np.float32)  # (H, W) float32 array
            return depth_np
        else:
            return None


    def get_rgb_image(self,side='L'):
        """Grabs an image and returns the RGB frame as a NumPy array"""
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            if side == 'L':
                self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
                frame = self.image_left.get_data()
            elif side == 'R':
                self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
                frame = self.image_right.get_data()
            elif side == 'B':
                self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
                self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
                left_np = self.image_left.get_data()
                right_np = self.image_right.get_data()
                frame = np.hstack((left_np, right_np))
            else:
                raise ValueError("Invalid side. Use 'L' for left or 'R' for right or 'B' for both.")
            # Convert RGBA to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            return rgb_image
        else:
            return None

    def get_rgbd_image(self):
        """Grabs an image and returns the RGBD frame as a NumPy arrays"""
        depth = sl.Mat()
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # depth in millimeters
            depth_np = depth.get_data().astype(np.float32)  # (H, W) float32 array
            frame = self.image_left.get_data()
            # Convert RGBA to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            return rgb_image, depth_np
        else:
            return None, None

    def get_rgb_intrinsics(self,side='L'):
        # Get calibration parameters
        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
        if side == 'L':
            params = calibration_params.left_cam
        elif side == 'R':
            params = calibration_params.right_cam

        # Intrinsic parameters
        fx = params.fx
        fy = params.fy
        cx = params.cx
        cy = params.cy
        
        # Intrinsic camera matrix (3x3)
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        
        return K

    def close(self):
        self.zed.close()
