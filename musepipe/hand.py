from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR
from cv2 import Mat as cv_Mat
from cv2 import VideoCapture as cv_VideoCapture
from cv2 import cvtColor
from cv2 import flip as cv_flip
from cv2 import imread as cv_imread
from cv2 import imshow as cv_imshow
from cv2 import waitKey as cv_waitKey
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands


class MpHand:

    image_files: list[str]

    def __init__(
        self,
        static_image_mode: bool,
        max_num_hands: int,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def process_image(self, file_list: list[str]):
        # Read an image, flip it around y-axis
        # for correct handedness output (see above).
        # Convert the BGR image to RGB before processing.
        return [
            (
                idx,
                self.hands.process(
                    cvtColor(cv_flip(cv_imread(file), 1), COLOR_BGR2RGB)
                ),
            )
            for idx, file in enumerate(file_list)
        ]

    def camera_mode(self, video_capture: int, flip: int):
        # For webcam input:
        cap = cv_VideoCapture(video_capture)
        while cap.isOpened():
            success: bool
            image: cv_Mat
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cvtColor(image, COLOR_BGR2RGB)
            results = self.hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cvtColor(image, COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )
            # Flip the image horizontally for a selfie-view display.
            cv_imshow("MediaPipe Hands", cv_flip(image, flip))
            if cv_waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()


## Debug Hand
# mphand = MpHand(
#     static_image_mode=False,
#     max_num_hands=2,
#     model_complexity=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
# )
# mphand.camera_mode(1, 1)
