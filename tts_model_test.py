import asyncio
import sounddevice as sd
from vts import VTS
from tts import TTS


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

    vts = VTS()
    await vts.connect()

    vts.parameters["EyeOpenLeft"]["value"] = 0.7
    vts.parameters["EyeOpenRight"]["value"] = 0.7
    vts.parameters["MouthSmile"]["value"] = 0.6

    async for chunk in tts.infer("Hi, I'm Raine, your favorite Ay I veetuber!\nHello everyone, thanks for joining the stream today!\nIt really means a lot to me."):
        sd.play(chunk["data"], chunk["sample_rate"])

        for volume_level in chunk["volume_data"]:
            vts.parameters["MouthOpen"]["value"] = volume_level

            await asyncio.sleep(0.05)

        vts.parameters["MouthOpen"]["value"] = 0

        await asyncio.sleep(0.3)

    await tts.close()
    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
