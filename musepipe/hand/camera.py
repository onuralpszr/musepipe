import threading
from types import NoneType

from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR
from cv2 import Mat as cv_Mat
from cv2 import VideoCapture as cv_VideoCapture
from cv2 import cvtColor
from cv2 import flip as cv_flip
from cv2 import imshow as cv_imshow
from cv2 import waitKey as cv_waitKey
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands


class MpHandCamera:
    def __init__(
        self,
        video_src: int,
        max_num_hands: int,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
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
        self.cap: cv_VideoCapture = cv_VideoCapture(video_src)

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

    def get_frame(self) -> tuple[bool, cv_Mat]:
        with self.cap_lock:
            frame: cv_Mat = self.frame.copy()
            success: bool = self.success
        return success, frame

    def get_multi_handedness(self):
        with self.cap_lock:
            if self.results_detect:
                multi_handedness = self.multi_handedness
                return multi_handedness
            else:
                return NoneType

    def get_multi_hand_world_landmarks(self):
        with self.cap_lock:
            if self.results_detect:
                multi_hand_world_landmarks = self.multi_hand_world_landmarks
                return multi_hand_world_landmarks
            else:
                return NoneType

    def stop(self):
        self.cap_status = False
        self.thread.join()

    def update_frame(self):
        # For webcam input:
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

            # Flip the image horizontally for a selfie-view display.
            cv_imshow("MediaPipe Hands", cv_flip(frame, 1))
            cv_waitKey(1)

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
