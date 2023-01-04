import typer

from musepipe.hand import MpHand

from . import __version__

app = typer.Typer()


@app.command()
def version():
    print(f"MusePipe Version {__version__}")


@app.command()
def webcamhand(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_num_hands: int = 2,
    static_image_mode: bool = False,
):
    mphand = MpHand(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    mphand.camera_mode(1, 1)


if __name__ == "__main__":
    app()
