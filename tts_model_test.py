import asyncio
from vts import VTS
from tts import TTS


async def main():
    tts = TTS(
        "gpt_sovits/GPT_weights/Raine GPT-SoVITS v1_e15_s165.pth",
        "gpt_sovits/SoVITS_weights/Raine GPT-SoVITS v1-e18.ckpt",
        {
            "smug": (
                "tts/ref_audio/smug.wav",
                "Free from social mores and other people's opinions, and no family obligations. Hmph, must be an easy life."
            ),
        }
    )

    for chunk in tts.infer("ok so now I can start working on syncing audio output from the tts to the mouthOpen parameter"):
        tts.play(chunk)

    return

    vts = VTS()
    await vts.connect()

    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
