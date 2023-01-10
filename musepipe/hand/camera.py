import threading

from cv2 import (
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    COLOR_RGB2BGR,
)
from cv2 import Mat as cv_Mat
from cv2 import VideoCapture, cvtColor
from cv2 import flip as cv_flip
from cv2 import imshow as cv_imshow
from cv2 import waitKey as cv_waitKey
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from numpy import uint8
from numpy import zeros as np_zeros


class MpHandCamera:
    def __init__(
        self,
        video_src: int,
        max_num_hands: int,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        preview: bool = False,
    ) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        self.cap_status = False
        self.results_detect = False
        self.cap_lock = threading.Lock()
        self.cap: VideoCapture = VideoCapture(video_src)
        self.frame: cv_Mat
        self.preview: bool = preview

        width = self.cap.get(CAP_PROP_FRAME_WIDTH)  # camera float `width`
        height = self.cap.get(CAP_PROP_FRAME_HEIGHT)  # camera float `height`
        self.last_frame: cv_Mat = np_zeros(
            (int(height), int(width), 3), dtype=uint8
        )

    def __drawing(self, results, image: cv_Mat) -> cv_Mat:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
            )
        return image

    def start(self):
        if self.cap_status:
            print("MP Hand capturing has already been started.")
            return None
        self.cap_status = True
        self.thread = threading.Thread(target=self.update_frame, args=())
        self.thread.start()
        return self

    def get_frame(self) -> tuple[bool, cv_Mat | None]:

        success: bool = False
        with self.cap_lock:
            print(self.results_detect)
            while self.results_detect is False:
                print(self.results_detect)
                self.last_frame = self.frame.copy()
                success = self.success

        return success, self.last_frame

    def get_multi_handedness(self):
        with self.cap_lock:
            if self.results_detect:
                multi_handedness = self.multi_handedness
                return multi_handedness
            else:
                return None

    def __show_preview(self, frame: cv_Mat):
        cv_imshow("MediaPipe Hands", cv_flip(frame, 1))
        cv_waitKey(1)

    def get_multi_hand_world_landmarks(self):
        with self.cap_lock:
            if self.results_detect:
                multi_hand_world_landmarks = self.multi_hand_world_landmarks
                return multi_hand_world_landmarks
            else:
                return None

    def stop(self):
        self.cap_status = False
        self.thread.join()

    def update_frame(self):
        # For webcam input:
        self.results_detect = False
        while self.cap.isOpened() & self.cap_status:
            success, frame = self.cap.read()
            with self.cap_lock:
                self.success = success
                self.frame = frame
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            frame = cvtColor(frame, COLOR_BGR2RGB)
            self.results = self.hands.process(frame)
            # Draw the hand annotations on the image.
            frame.flags.writeable = True
            frame = cvtColor(frame, COLOR_RGB2BGR)
            if self.results.multi_hand_landmarks:
                frame = self.__drawing(self.results, frame)

            if self.results.multi_handedness:
                # print(self.results.multi_handedness)
                self.multi_handedness = self.results.multi_handedness
                self.multi_hand_world_landmarks = (
                    self.results.multi_hand_world_landmarks
                )
                self.results_detect = True

            if self.preview:
                self.__show_preview(frame)

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
