from io import BytesIO
import requests
from tqdm import tqdm
from soundfile import read as sf_read
from sounddevice import play, wait


class TTS:
    def __init__(
            self,
            emotions: dict[str, tuple[str, str]],
    ):
        self.emotions = emotions
        self.emotion = list(emotions.keys())[0]
        self.set_emotion(self.emotion)

    def set_emotion(self, emotion: str):
        self.emotion = emotion

        requests.get("http://127.0.0.1:9880/change_refer", params={
            "refer_wav_path": self.emotions[emotion][0],
            "prompt_text": self.emotions[emotion][1],
            "prompt_language": "en"
        })

    def infer(
            self,
            text: str,
            emotion: str | None = None,
    ):
        if emotion:
            if emotion != self.emotion:
                self.set_emotion(emotion)

        response = requests.get("http://127.0.0.1:9880/", params={
            "text": text,
            "text_language": "en"
        }, stream=True)

        for i, chunk in tqdm(enumerate(response.iter_content(chunk_size=5292000)), "GPT-SoVITS"): # chunk size is *supposed* to be 30 seconds of wav file data
            if i % 2:
                #audio_data = 
                play(*sf_read(BytesIO(chunk)))
                wait()
            else:
                text = chunk.decode("utf-8")

        yield "hi", "hello"
