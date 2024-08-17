from io import BytesIO
from math import floor
import aiohttp
from tqdm.asyncio import tqdm
import soundfile as sf
import numpy as np
from scipy.signal import resample
from scipy.ndimage import uniform_filter1d


class TTS:
    def __init__(
            self,
            emotions: dict[str, tuple[str, str]],
    ):
        self.emotions = emotions
        self.emotion = None
        self.session = aiohttp.ClientSession(base_url="http://127.0.0.1:9880")
    
    async def close(self):
        await self.session.close()

    async def set_emotion(self, emotion: str):
        self.emotion = emotion

        async with self.session.get("/change_refer", params={
            "refer_wav_path": self.emotions[emotion][0],
            "prompt_text": self.emotions[emotion][1],
            "prompt_language": "en"
        }) as resp:
            await resp.wait_for_close()

    async def infer(
            self,
            text: str,
            emotion: str | None = None,
    ):
        if emotion:
            if emotion != self.emotion:
                await self.set_emotion(emotion)

        elif self.emotion is None:
            await self.set_emotion(list(self.emotions.keys())[0])

        async with self.session.get("/", params={
            "text": text,
            "text_language": "en"
        }) as resp:
            current_text = None

            audio_bytes = b""
            async for chunk in tqdm(resp.content.iter_chunks(), "TTS"):
                chunk, complete = chunk

                try:
                    current_text = chunk.decode()
                    if current_text != "":
                        continue
                except UnicodeDecodeError:
                    pass

                audio_bytes += chunk

                if not complete:
                    continue

                try:
                    audio = sf.SoundFile(BytesIO(audio_bytes))

                    duration = audio.frames / audio.samplerate

                    audio_array = audio.read()
                    volume_array = resample(
                        uniform_filter1d(audio_array, 4000),
                        floor(duration * 200)
                    )[::10]
                    volume_array = np.interp(volume_array, (volume_array.min(), volume_array.max()), (0.27, 0.5))

                    yield {
                        "data": audio_array,
                        "sample_rate": audio.samplerate,
                        "duration": duration,
                        "text": current_text,
                        "volume_data": volume_array, # volume level from 0-1 every 0.1s
                    }

                    audio_bytes = b""
                except sf.LibsndfileError as e:
                    if "\n" in text:
                        raise e
