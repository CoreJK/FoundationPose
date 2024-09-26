import time
import cv2
from loguru import logger
from settings import *

# 奥比中光相机 sdk
from pyorbbecsdk import *
from camera_utils import frame_to_bgr_image


class DaBaiCamera(object):
    """奥比中光相机"""
    def __init__(self):
        self.pipline = Pipeline()
        self.config = Config()
        self.has_color_sensor = False

    def setup_camera_color_config(self):
        """配置相机彩色相关参数"""
        try:
            profile_list = self.pipline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                self.config.enable_stream(color_profile)
                self.has_color_sensor = True
        except OBError as e:
            logger.error(e)
            
    def setup_camera_depth_config(self):
        """配置相机深度相关参数"""
        depth_profile_list = self.pipline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is not None:
            depth_profile = depth_profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
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
                image_name = f"{IMAGE_PATH}/color_image_{timestamp}.jpg"
                logger.debug(f"保存图片到: {image_name}")
                cv2.imwrite(str(image_name), image)
                return image_name

    def start_capture(self):
        self.pipline.start(self.config)
        logger.info("启动相机拍摄")
        while True:
            frames = self.pipline.wait_for_frames(100)
            if frames is None:
                continue
            else:
                color_frame = frames.get_color_frame()
                if color_frame is not None:
                    image_path = self.save_color_frame(color_frame)
                    self.pipline.stop()
                    return image_path


def check_camera():
    """开启摄像头，调用摄像头实时画面，按q键退"""
    logger.info('开启摄像头, 按 q 或 Crtl+C 退出!')
    cap = cv2.VideoCapture(4)

    if not cap.isOpened():
        logger.error('未找到摄像头设备!')

    picture_num = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Image', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                elif cv2.waitKey(25) & 0xFF == ord('s'):
                    picture_num += 1
                    save_path = Path(f'src/blinx_agent_robot_arm/pic/{picture_num}.jpg')
                    cv2.imwrite(str(save_path), frame)
                    logger.info(f'图片保存到: {save_path}')
            else:
                cap.release()
                cv2.destroyAllWindows()
                break

    except KeyboardInterrupt:
        logger.info('按 Ctrl+C 键退出!')
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = DaBaiCamera()
    camera.setup_camera_color_config()
    pic_path = camera.start_capture()
    logger.info("相机图片采集完成，开始进行目标检测")
    cv2.imshow("Object Detection", pic_path)
    cv2.waitKey(0)
    camera.pipline.stop()
    cv2.destroyAllWindows()