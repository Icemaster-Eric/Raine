class TTS:
    def __init__(
            self,
            gpt_path: str,
            sovits_path: str,
            emotions: dict[str, tuple[str, str]],
            config_path: str = "gpt_sovits/GPT_SoVITS/configs/tts_infer.yaml",
    ):
        pass

    def set_emotion(self, emotion: str):
        self.emotion = self.emotions[emotion]

    def infer(
            self,
            text: str,
            emotion: str | None = None,
            top_k: int = 5,
            top_p: float = 1,
            temp: float = 0.75,
            rep_pen: float = 1.02,
    ):
        if emotion:
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

        # api call here
