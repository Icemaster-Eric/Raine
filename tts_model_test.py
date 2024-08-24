import asyncio
import math
import json
from random import randint
import sounddevice as sd
from vts import VTS
from tts import TTS


def get_movements():
    movements = {}

    with open("vts/movements/blink.json", "r") as f:
        movements["blink"] = json.load(f)

    return movements


async def move(vts: VTS, movement_data: list[dict]):
    for move in movement_data:
        for param_name, value in move.items():
            vts.parameters[param_name]["value"] = value

        await asyncio.sleep(0.05)

    # return model to original parameters?


async def blink_animation(vts: VTS, movement_data: list[dict]):
    while True:
        for move in movement_data:
            for param_name, value in move.items():
                vts.parameters[param_name]["value"] = value

            await asyncio.sleep(0.05)

        await asyncio.sleep(randint(35, 43) / 10)


async def head_animation(vts: VTS):
    while True:
        await asyncio.sleep(randint(2, 4))

        for i in range(40):
            time = i * 0.05
            value = 12 * math.sin(time * math.pi)

            vts.parameters["FaceAngleZ"]["value"] = value

            await asyncio.sleep(0.05)


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

    movements = get_movements()

    vts = VTS()
    await vts.connect()

    vts.parameters["EyeOpenLeft"]["value"] = 0.7
    vts.parameters["EyeOpenRight"]["value"] = 0.7
    vts.parameters["MouthSmile"]["value"] = 0.4

    asyncio.create_task(blink_animation(vts, movements["blink"]))
    asyncio.create_task(head_animation(vts))

    async for chunk in tts.infer("""Love looks not with the eyes, but with the mind
And therefore is wing'd Cupid painted blind.
Nor hath love's mind of any judgment taste"""):
        sd.play(chunk["data"], chunk["sample_rate"])

        for i, volume_level in enumerate(chunk["volume_data"]):
            vts.parameters["MouthOpen"]["value"] = volume_level

            await asyncio.sleep(0.05)

        vts.parameters["MouthOpen"]["value"] = 0

    await tts.close()
    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
