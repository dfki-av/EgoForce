import os
import sys  

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import cv2
import torch
import numpy as np
import aria.sdk as aria

from aria_utils import TimestampIndex
from inference import Inference
from camera_models import OVR624CameraModel
from demo_utils import compose_output_frame, brighten_rgb


DEFAULT_CAMERA_WIDTH = 1408
DEFAULT_CAMERA_HEIGHT = 1408

DEFAULT_F = np.array([1220.38417234667, 1220.38417234667], dtype=np.float32) / 2
DEFAULT_C = np.array([1459.308327420149, 1446.481271789112], dtype=np.float32) / 2

DEFAULT_PARAMS = np.array([
0.3881739923440562,
-0.3505272968594388,
-0.2039745469127034,
1.616037232456187,
-1.99366280389576,
0.7186532253554115,
0.0004348725659534717,
0.0001491800990352849,
0.0007271366281008595,
2.078482331496759e-06,
-0.0001256546435329295,
-0.0001402891608396858
], dtype=np.float32)


def build_camera_model(width=DEFAULT_CAMERA_WIDTH, height=DEFAULT_CAMERA_HEIGHT):
    scale = np.array(
        [width / DEFAULT_CAMERA_WIDTH, height / DEFAULT_CAMERA_HEIGHT],
        dtype=np.float32,
    )
    f = DEFAULT_F * scale
    c = DEFAULT_C * scale
    return OVR624CameraModel(f, c, params=DEFAULT_PARAMS.copy(), width=width, height=height)


class StreamingClientObserver:
    def __init__(self):
        self.queue_size = 100

        self.rgb_data = None
        
        self.slam1_data = TimestampIndex(self.queue_size)
        self.slam2_data = TimestampIndex(self.queue_size)
        self.imu1_data = TimestampIndex(self.queue_size)
        self.imu2_data = TimestampIndex(self.queue_size)

    def on_image_received(self, image: np.array, ImageDataRecord) -> None: 
        if ImageDataRecord.camera_id == aria.CameraId.Rgb:
            self.rgb_data = [image, ImageDataRecord.capture_timestamp_ns]
        elif ImageDataRecord.camera_id == aria.CameraId.Slam1:
            self.slam1_data.add_timestamp(image, ImageDataRecord.capture_timestamp_ns)
        elif ImageDataRecord.camera_id == aria.CameraId.Slam2:
            self.slam2_data.add_timestamp(image, ImageDataRecord.capture_timestamp_ns)
        
    def on_imu_received(self, motion_data, imu_idx) -> None:
        for motion in motion_data:
            data = {
            "accel_msec2": motion.accel_msec2, 
            "accel_valid": motion.accel_valid, 
            "gyro_radsec": motion.gyro_radsec, 
            "gyro_valid": motion.gyro_valid, 

            }

            if imu_idx == 0:
                self.imu1_data.add_timestamp(data, motion.capture_timestamp_ns)
            elif imu_idx == 1:
                self.imu2_data.add_timestamp(data, motion.capture_timestamp_ns)

    def get_data(self):
        if self.rgb_data is None:
            return None
        
        rgb_image, reference_timestamp = self.rgb_data

        slam1_data = self.slam1_data.find_closest_data(reference_timestamp)
        slam2_data = self.slam2_data.find_closest_data(reference_timestamp)
        imu1_data = self.imu1_data.find_closest_data(reference_timestamp)
        imu2_data = self.imu2_data.find_closest_data(reference_timestamp)

        return {
            "rgb_image": rgb_image,
            "slam1_image": slam1_data,
            "slam2_image": slam2_data,
            "imu1_data": imu1_data,
            "imu2_data": imu2_data,
        }


def aria_device_inference():
    # update_iptables()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Trace)

    # 1. Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    device_client.set_client_config(client_config)

    # 2. Connect to the device
    device = device_client.connect()

    # 3. Retrieve the streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # 4. Set custom config for streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = "profile21"
    streaming_config.streaming_interface = aria.StreamingInterface.Usb

    # Use ephemeral streaming certificates
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # 5. Start streaming
    streaming_manager.start_streaming()

    # 6. Get streaming state
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")

    subscription_config = streaming_client.subscription_config
    subscription_config.subscriber_data_type = (
        aria.StreamingDataType.Rgb 
    )

    subscription_config.message_queue_size[aria.StreamingDataType.Rgb] = 1

    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    subscription_config.security_options = options
    streaming_client.subscription_config = subscription_config

    inference = Inference(build_camera_model())

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    window_name = "Aria Inference Demo"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            data = observer.get_data()
            if data is None:
                continue
            rgb_image = data["rgb_image"]
            rgb_image = np.rot90(rgb_image, -1)
            
            rgb_image = brighten_rgb(rgb_image)
            
            render_image, tp_image = inference.run(rgb_image.copy(), inference.device)
            stacked_rgb = compose_output_frame(rgb_image, render_image, tp_image)
            stacked_bgr = cv2.cvtColor(stacked_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow(window_name, stacked_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        print("Exiting...") 
    finally:
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        if hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()


def main():
    aria_device_inference()



if __name__ == "__main__":
    main()
