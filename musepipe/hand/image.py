import os
from pathlib import Path

from cv2 import COLOR_BGR2RGB
from cv2 import Mat as cv_Mat
from cv2 import cvtColor
from cv2 import flip as cv_flip
from cv2 import imread as cv_imread
from cv2 import imwrite as cv_imwrite
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands


class MpHandImage:
    def __init__(
        self,
        max_num_hands: int,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        annotated_image_output_path: str,
        input_path: str = ".",
    ) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        self.results_detect = False
        self.multi_hand_landmarks_list: list = []
        self.multi_handedness_list: list = []
        self.annotated_image_output_path = annotated_image_output_path
        self.__read_images(input_path)

    def drawing(
        self, results, image: cv_Mat, filename: str = "result"
    ) -> None:
        Path(self.annotated_image_output_path).mkdir(
            parents=True, exist_ok=True
        )
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                print("itercount")
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                )
                cv_imwrite(
                    self.annotated_image_output_path + "/" + filename + ".png",
                    cv_flip(image, 1),
                )
                # # Draw hand world landmarks.
                # if not results.multi_hand_world_landmarks:
                #     continue
                # for hand_world_landmarks in results.multi_hand_world_landmarks:
                #     mp_drawing.plot_landmarks(
                #         hand_world_landmarks,
                #         mp_hands.HAND_CONNECTIONS,
                #         azimuth=5,
                #     )

    def get_multi_handedness_list(self):
        return self.multi_hand_landmarks_list

    def get_multi_hand_world_landmarks_list(self):
        return self.multi_handedness_list

    def __read_images(self, file_path: str):
        # For image input:
        count = 0
        for file in os.listdir(file_path):
            if os.path.isfile(file_path + file):
                image = cv_imread(file_path + file)
                if image is not None:
                    count = count + 1
                    # Read an image, flip it around y-axis for correct handedness output
                    image = cv_flip(image, 1)
                    # Convert the BGR image to RGB before processing.
                    results = self.hands.process(
                        cvtColor(image, COLOR_BGR2RGB)
                    )

                    if not results.multi_hand_landmarks:
                        continue
                    else:
                        self.multi_handedness_list.append(
                            (count, results.multi_handedness)
                        )
                        self.multi_hand_landmarks_list.append(
                            (count, results.multi_hand_landmarks)
                        )

                        if self.annotated_image_output_path != "":
                            self.drawing(
                                results=results,
                                image=image.copy(),
                                filename=str(count),
                            )
