from time import sleep

import typer

from musepipe.hand.camera import MpHandCamera
from musepipe.hand.image import MpHandImage

from . import __version__

app = typer.Typer()


@app.command()
def version():
    print(f"MusePipe Version {__version__}")


@app.command()
def mpcamerahand(
    max_num_hands: int = 2, timeout: int = 100, video_src: int = 1
):
    mphand = MpHandCamera(
        video_src=video_src,
        max_num_hands=max_num_hands,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    mphand.start()
    sleep(timeout)
    mphand.stop()


@app.command()
def mpimagehand(
    input_path: str,
    output_path: str = "",
    max_num_hands: int = 2,
):

    mphand_image = MpHandImage(
        input_path=input_path,
        annotated_image_output_path=output_path.strip(""),
        max_num_hands=max_num_hands,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print(mphand_image.get_multi_hand_world_landmarks_list())
    print(mphand_image.get_multi_handedness_list())


if __name__ == "__main__":
    app()
