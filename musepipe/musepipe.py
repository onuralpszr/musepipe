from time import sleep

import typer

from musepipe.hand.camera import MpHandCamera

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


if __name__ == "__main__":
    app()
