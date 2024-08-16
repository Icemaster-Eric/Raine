import asyncio
import sounddevice as sd
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
    await tts.set_emotion("smug")

    async for chunk in tts.infer("""You often want to send some sort of data in the URL’s query string.
If you were constructing the URL by hand, this data would be given as key/value pairs in the URL after a question mark, e.g. httpbin.org/get?key=val.
Requests allows you to provide these arguments as a dictionary of strings, using the params keyword argument.
As an example, if you wanted to pass key1=value1 and key2=value2 to httpbin.org/get, you would use the following code:"""):
        sd.play(chunk["data"], chunk["samplerate"])
        print(chunk["text"])
        sd.wait()

    return

    vts = VTS()
    await vts.connect()

    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
