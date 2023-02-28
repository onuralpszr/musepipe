from cv2 import Mat as cv_Mat
from mediapipe.python.solutions import face_mesh as mp_facemesh


class MpFaceMesh:
    def __init__(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        preview: bool = False,
    ) -> None:
        self.hands = mp_facemesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.cap_status = False
        self.results_detect = False
        self.frame: cv_Mat
        self.preview: bool = preview
