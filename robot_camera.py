import os
import time
from typing import List
from queue import Queue

from loguru import logger
import cv2
import numpy as np
import open3d as o3d

# 奥比中光相机 sdk
from pyorbbecsdk import *
from camera_utils import frame_to_bgr_image

from settings import *

ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

# 多通道读取的变量
MAX_DEVICES = 2
MAX_QUEUE_SIZE = 5

class TemporalFilter:
    """过深度数据里滤掉无效的帧"""
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

class DaBaiCamera(object):
    """奥比中光相机"""
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.temporal_filter = TemporalFilter(alpha=0.5)
        self.save_points_dir = os.path.join(os.getcwd(), "point_clouds")
        self.last_print_time = time.time()
        self.has_color_sensor = False

    def setup_camera_color_config(self):
        """配置相机彩色相关参数"""
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                self.config.enable_stream(color_profile)
                self.has_color_sensor = True
        except OBError as e:
            logger.error(e)
            
    def setup_camera_depth_config(self):
        """配置相机深度相关参数"""
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is not None:
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)

    def save_color_frame(self, frame):
        """保存彩色帧"""
        if frame is None:
            logger.error("未捕获到相机画面!")
        else:
            width = frame.get_width()
            height = frame.get_height()
            logger.debug(f"相机画面尺寸为: {width}x{height}")

            image = frame_to_bgr_image(frame)
            if image is None:
                logger.error("相机画面转换失败!")
            else:
                # 根据时间戳保存图片
                timestamp = int(time.time())
                image_name = f"{IMAGE_PATH}/color_image_{timestamp}.png"
                logger.debug(f"保存图片到: {image_name}")
                cv2.imwrite(str(image_name), image)
                return image_name

    def start_capture(self):
        """单次拍摄一张彩色图片"""
        self.pipeline.start(self.config)
        logger.info("启动相机拍摄")
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            else:
                color_frame = frames.get_color_frame()
                if color_frame is not None:
                    image_path = self.save_color_frame(color_frame)
                    self.pipeline.stop()
                    return image_path

    def start_video_stream(self):
        """开启 RGB 视频流"""
        self.pipeline.start(self.config)
        while True:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                # Convert to RGB format
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    logger.error("failed to convert frame to image")
                    continue
                cv2.imshow("Color Viewer", color_image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY:
                    break
                if key == ord('s'):
                    self.save_color_frame(color_frame)
            except KeyboardInterrupt:
                break
        self.pipeline.stop()

    def start_depth_video_stream(self):
        """开启深度视频流"""
        self.pipeline.start(self.config)
        while True:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()

                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                depth_data = depth_data.astype(np.float32) * scale
                depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
                depth_data = depth_data.astype(np.uint16)
                # Apply temporal filtering
                depth_data = self.temporal_filter.process(depth_data)

                center_y = int(height / 2)
                center_x = int(width / 2)
                center_distance = depth_data[center_y, center_x]

                current_time = time.time()
                if current_time - self.last_print_time >= PRINT_INTERVAL:
                    logger.debug(f"center distance: {center_distance}")
                    self.last_print_time = current_time

                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

                cv2.imshow("Depth Viewer", depth_image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY:
                    break
                if key == ord('s'):
                    self.save_points_to_ply(frames, self.pipeline.get_camera_param())
                if key == ord('c'):
                    self.save_color_points_to_ply(frames, self.pipeline.get_camera_param())
            except KeyboardInterrupt:
                break
        self.pipeline.stop()
    
    def convert_to_o3d_point_cloud(self, points, colors=None):
        """
        Converts numpy arrays of points and colors (if provided) into an Open3D point cloud object.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Assuming colors are in [0, 255]
        return pcd

    def save_points_to_ply(self, frames, camera_param):
        """
        Saves the point cloud data to a PLY file using Open3D.
        """
        
        os.makedirs(self.save_points_dir, exist_ok=True)
        if frames is None:
            return 0
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return 0
        points = frames.get_point_cloud(camera_param)
        if points is None or len(points) == 0:
            logger.warning("No points to save.")
            return 0
        # Convert points to Open3D point cloud
        pcd = self.convert_to_o3d_point_cloud(np.array(points))
        points_filename = os.path.join(self.save_points_dir, f"points_{depth_frame.get_timestamp()}.ply")
        # Save to PLY file
        o3d.io.write_point_cloud(points_filename, pcd)
        logger.info(f"Saved points to {points_filename}")

    def save_color_points_to_ply(self, frames, camera_param):
        """
        Saves the color point cloud data to a PLY file using Open3D.
        """
        if frames is None:
            return 0
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return 0
        points = frames.get_color_point_cloud(camera_param)
        if points is None or len(points) == 0:
            logger.warning("No color points to save.")
            return 0
        # Assuming the color information is included in the points array
        # This part might need to be adjusted based on the actual format of the points array
        xyz = np.array(points[:, :3])
        colors = np.array(points[:, 3:], dtype=np.uint8)
        pcd = self.convert_to_o3d_point_cloud(xyz, colors)
        points_filename = os.path.join(self.save_points_dir, f"color_points_{depth_frame.get_timestamp()}.ply")
        # Save to PLY file
        o3d.io.write_point_cloud(points_filename, pcd)
        logger.info(f"Saved color points to {points_filename}")
        return 1


class MultiDeviceManager:
    """相机多通道调用类"""
    def __init__(self):
        self.color_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
        self.depth_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
        self.has_color_sensor: List[bool] = [False for _ in range(MAX_DEVICES)]
        self.stop_rendering = False
        self.curr_device_cnt = 0
        self.save_dir = os.path.join(os.getcwd(), "saved_frames")

    def on_new_frame_callback(self, frames, index: int):
        """处理新帧的回调函数"""
        assert index < MAX_DEVICES
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is not None:
            if self.color_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
                self.color_frames_queue[index].get()
            self.color_frames_queue[index].put(color_frame)
        if depth_frame is not None:
            if self.depth_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
                self.depth_frames_queue[index].get()
            self.depth_frames_queue[index].put(depth_frame)

    def save_frame(self, frame, frame_type, device_index):
        """保存帧"""
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = int(time.time())
        frame_name = f"{self.save_dir}/{frame_type}_device_{device_index}_{timestamp}.png"
        if frame_type == "color":
            image = frame_to_bgr_image(frame)
            cv2.imwrite(frame_name, image)
        elif frame_type == "depth":
            width = frame.get_width()
            height = frame.get_height()
            scale = frame.get_depth_scale()
            depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            cv2.imwrite(frame_name, depth_image)
        logger.info(f"Saved {frame_type} frame to {frame_name}")

    def save_point_cloud(self, color_frame, depth_frame, device_index):
        """保存点云图"""
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = int(time.time())
        ply_name = f"{self.save_dir}/point_cloud_device_{device_index}_{timestamp}.ply"

        # 获取彩色图像
        color_image = frame_to_bgr_image(color_frame)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 获取深度图像
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))
        depth_data = depth_data.astype(np.float32) * scale

        # 调整彩色图像的尺寸以匹配深度图像
        color_image = cv2.resize(color_image, (width, height))

        # 创建点云
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_data),
            convert_rgb_to_intensity=False
        )
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, 525, 525, width // 2, height // 2
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

        # 保存点云
        o3d.io.write_point_cloud(ply_name, pcd)
        logger.info(f"Saved point cloud to {ply_name}")

    def rendering_frames(self):
        """渲染帧"""
        while not self.stop_rendering:
            for i in range(self.curr_device_cnt):
                color_frame = None
                depth_frame = None
                if not self.color_frames_queue[i].empty():
                    color_frame = self.color_frames_queue[i].get()
                if not self.depth_frames_queue[i].empty():
                    depth_frame = self.depth_frames_queue[i].get()
                if color_frame is None and depth_frame is None:
                    continue
                color_image = None
                depth_image = None
                color_width, color_height = 0, 0
                if color_frame is not None:
                    color_width, color_height = color_frame.get_width(), color_frame.get_height()
                    color_image = frame_to_bgr_image(color_frame)
                if depth_frame is not None:
                    width = depth_frame.get_width()
                    height = depth_frame.get_height()
                    scale = depth_frame.get_depth_scale()

                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_data = depth_data.reshape((height, width))

                    depth_data = depth_data.astype(np.float32) * scale

                    depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX,
                                                dtype=cv2.CV_8U)
                    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

                if color_image is not None and depth_image is not None:
                    window_size = (color_width // 2, color_height // 2)
                    color_image = cv2.resize(color_image, window_size)
                    depth_image = cv2.resize(depth_image, window_size)
                    image = np.hstack((color_image, depth_image))
                elif depth_image is not None and not self.has_color_sensor[i]:
                    image = depth_image
                else:
                    continue
                cv2.imshow("Device {}".format(i), image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY:
                    return
                if key == ord('s'):
                    if color_frame is not None:
                        self.save_frame(color_frame, "color", i)
                    if depth_frame is not None:
                        self.save_frame(depth_frame, "depth", i)
                    if color_frame is not None and depth_frame is not None:
                        self.save_point_cloud(color_frame, depth_frame, i)

    def start_streams(self, pipelines, configs):
        """启动设备流"""
        index = 0
        for pipeline, config in zip(pipelines, configs):
            print("Starting device {}".format(index))
            pipeline.start(config, lambda frame_set, curr_index=index: self.on_new_frame_callback(frame_set, curr_index))
            index += 1

    def stop_streams(self, pipelines):
        """停止设备流"""
        for pipeline in pipelines:
            pipeline.stop()

    def main(self):
        """主函数"""
        ctx = Context()
        device_list = ctx.query_devices()
        self.curr_device_cnt = device_list.get_count()
        if self.curr_device_cnt == 0:
            print("No device connected")
            return
        if self.curr_device_cnt > MAX_DEVICES:
            print("Too many devices connected")
            return
        pipelines: List = []
        configs: List = []
        for i in range(device_list.get_count()):
            device = device_list.get_device_by_index(i)
            pipeline = Pipeline(device)
            config = Config()
            try:
                profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
                self.has_color_sensor[i] = True
            except OBError as e:
                print(e)
                self.has_color_sensor[i] = False
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            pipelines.append(pipeline)
            configs.append(config)
        self.start_streams(pipelines, configs)
        try:
            self.rendering_frames()
            self.stop_streams(pipelines)
        except KeyboardInterrupt:
            self.stop_rendering = True
            self.stop_streams(pipelines)


if __name__ == '__main__':
    # camera = DaBaiCamera()
    # camera.setup_camera_color_config()
    # camera.start_video_stream()

    manager = MultiDeviceManager()
    manager.main()