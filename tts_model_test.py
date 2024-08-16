import asyncio
from vts import VTS
from tts import TTS


async def main():
    tts = TTS(
        {
            "smug": (
                "E:/Code/Raine/tts/ref_audio/smug.wav",
                "Free from social mores and other people's opinions, and no family obligations. Hmph, must be an easy life."
            ),
        }
    )

    for audio, text in tts.infer("ok so now I can start working on syncing audio output from the tts to the mouthOpen parameter"):
        pass

    return

    vts = VTS()
    await vts.connect()

    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
