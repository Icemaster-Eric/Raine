import time
import wave
from io import BytesIO
import sounddevice as sd
import soundfile as sf
from gpt_sovits.GPT_SoVITS.TTS_infer_pack.TTS import TTS as TTS_Pipeline
from gpt_sovits.GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()

    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    return wav_buf.getvalue()


class TTS:
    def __init__(
            self,
            gpt_path: str,
            sovits_path: str,
            emotions: dict[str, tuple[str, str]],
            config_path: str = "gpt_sovits/GPT_SoVITS/configs/tts_infer.yaml",
    ):
        start_time = time.time()

        self.config = TTS_Config(config_path)
        self.pipeline = TTS_Pipeline(self.config)
        self.pipeline.init_t2s_weights(gpt_path)
        self.pipeline.init_vits_weights(sovits_path)

        self.emotions = emotions
        self.emotion = list(emotions.values())[0]

        self.pipeline.set_ref_audio(self.emotion[0])

        print(f"Loaded TTS in {time.time() - start_time:.2f}s")

    def set_emotion(self, emotion: str):
        self.emotion = self.emotions[emotion]

    def infer(
            self,
            text: str,
            emotion: str,
            top_k: int = 5,
            top_p: float = 1,
            temp: float = 0.75,
            rep_pen: float = 1.02,
    ):
        if self.emotions[emotion] != self.emotion:
            self.emotion = self.emotions[emotion]

        ref_audio_path, ref_text = self.emotion

        req = {
            "text": text,
            "text_lang": "en",
            "ref_audio_path": ref_audio_path,
            "prompt_text": ref_text,
            "prompt_lang": "en",
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temp,
            "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
            "batch_size": 4,
            "batch_threshold": 1,  # float. threshold for batch splitting.
            "split_bucket": False,  # bool. whether to split the batch into multiple buckets.
            "speed_factor": 1.0,  # float. control the speed of the synthesized audio.
            "fragment_interval": 0.3,  # float. to control the interval of the audio fragment.
            "return_fragment": True,
            "seed": -1,  # int. random seed for reproducibility.
            "media_type": "raw",  # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
            "streaming_mode": True,  # bool. whether to return a streaming response.
            "parallel_infer": True,  # bool.(optional) whether to use parallel inference.
            "repetition_penalty": rep_pen
        }

        tts_generator = self.pipeline.run(req)

        for sr, chunk in tts_generator:
            yield wave_header_chunk(chunk)

    @staticmethod
    def play(audio: bytes):
        sd.wait()
        sd.play(*sf.read(BytesIO(audio)))
