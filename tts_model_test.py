import asyncio
import sounddevice as sd
from vts import VTS
from tts import TTS
import numpy as np


async def main():
    tts = TTS(
        {
            "default": (
                "E:/Code/Raine/tts/ref_audio/smug.wav",
                "Free from social mores and other people's opinions, and no family obligations. Hmph, must be an easy life."
            ),
        }
    )
    await tts.set_emotion("default")

    async for chunk in tts.infer("Hi, I'm Raine, your favorite Ay I veetuber!\nHello everyone, thanks for joining the stream today!\nIt really means a lot to me."):
        #sd.play(chunk["data"], chunk["sample_rate"])
        #test = np.array([1, 2, 3]).shape
        print(chunk["volume_data"])

        await asyncio.sleep(0.5)

    await tts.close()

    return

    vts = VTS()
    await vts.connect()

    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
