import asyncio
import math
import json
from random import randint
import sounddevice as sd
from vts import VTS
from tts import TTS


def calc_movement():
    movement = []

    for i in range(60):
        time = i * 0.05
        value = time

        movement.append({})

    return movement


with open("vts/movements/new.json", "w") as f:
    json.dump(calc_movement(), f)


def get_movements():
    movements = {}

    with open("vts/movements/sway.json", "r") as f:
        movements["sway"] = json.load(f)

    with open("vts/movements/nod.json", "r") as f:
        movements["nod"] = json.load(f)

    with open("vts/movements/shake.json", "r") as f:
        movements["shake"] = json.load(f)

    with open("vts/movements/blink.json", "r") as f:
        movements["blink"] = json.load(f)

    return movements


async def move(vts: VTS, movement_data: list[dict]):
    for move in movement_data:
        for param_name, value in move.items():
            vts.parameters[param_name]["value"] = value

        await asyncio.sleep(0.05)

    # return model to original parameters?


async def blink(vts: VTS, movement_data: list[dict]):
    while True:
        for move in movement_data:
            for param_name, value in move.items():
                vts.parameters[param_name]["value"] = value

            await asyncio.sleep(0.05)

        #await asyncio.sleep(randint(35, 43) / 10)
        await asyncio.sleep(1.8)


async def main():
    """tts = TTS(
        {
            "default": (
                "E:/Code/Raine/tts/ref_audio/smug.wav",
                "Free from social mores and other people's opinions, and no family obligations. Hmph, must be an easy life."
            ),
        }
    )
    await tts.set_emotion("default")"""

    movements = get_movements()

    vts = VTS()
    await vts.connect()

    vts.parameters["EyeOpenLeft"]["value"] = 0.7
    vts.parameters["EyeOpenRight"]["value"] = 0.7
    vts.parameters["MouthSmile"]["value"] = 0.4

    asyncio.create_task(blink(vts, movements["blink"]))

    """async for chunk in tts.infer("Hi, I'm Raine, your favorite Ay I veetuber!\nHello everyone, thanks for joining the stream today!\nIt really means a lot to me."):
        sd.play(chunk["data"], chunk["sample_rate"])

        for volume_level in chunk["volume_data"]:
            vts.parameters["MouthOpen"]["value"] = volume_level

            await asyncio.sleep(0.05)

        vts.parameters["MouthOpen"]["value"] = 0

        await asyncio.sleep(0.3)"""

    await move(vts, movements["sway"])

    await move(vts, movements["nod"])

    await move(vts, movements["shake"])

    #await tts.close()
    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
