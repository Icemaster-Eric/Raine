from io import BytesIO
import aiohttp
from tqdm.asyncio import tqdm
import soundfile as sf
import numpy as np
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
                    audio_array = audio.read()

                    arr = uniform_filter1d(audio_array[::1000].copy(), size=10)
                    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

                    yield {
                        "data": audio.read(),
                        "sample_rate": audio.samplerate,
                        "duration": audio.frames / audio.samplerate,
                        "text": current_text,
                        "volume_data": arr,
                    }

                    audio_bytes = b""
                except sf.LibsndfileError as e:
                    if "\n" in text:
                        raise e
