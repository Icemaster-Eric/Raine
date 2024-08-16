from io import BytesIO
import aiohttp
from tqdm import tqdm
import soundfile as sf


class TTS:
    def __init__(
            self,
            emotions: dict[str, tuple[str, str]],
    ):
        self.emotions = emotions
        self.emotion = None
        self.session = aiohttp.ClientSession(base_url="http://127.0.0.1:9880")

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

            i = 0
            complete_chunk = b""
            async for chunk in resp.content.iter_chunks():
                chunk, complete = chunk

                try:
                    current_text = chunk.decode()
                    print("decoded:", current_text)
                    if current_text != "":
                        continue
                except UnicodeDecodeError:
                    pass

                if not complete:
                    complete_chunk += chunk
                    continue

                audio_data = sf.read(BytesIO(complete_chunk))
                yield {
                    "data": audio_data[0],
                    "samplerate": audio_data[1],
                    "text": current_text
                }

                i += 1
                complete_chunk = b""
