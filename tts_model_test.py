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
    # the way I implemented this is probably really silly and trash but whatever, it *should* work
    bias = -0.3 # -1 = always left, 1 = always right

    while True:
        direction = "left" if bias + randint(-10, 10) / 10 < 0 else "right"

        for i in range(randint(10, 15)):
            if direction == "left":
                if vts.parameters["FaceAngleX"]["value"] > 12:
                    direction = "right"

                vts.parameters["FaceAngleX"]["value"] += randint(5, 10) / 10

            elif direction == "right":
                if vts.parameters["FaceAngleX"]["value"] < -12:
                    direction = "left"

                vts.parameters["FaceAngleX"]["value"] -= randint(5, 10) / 10

            # small z movement
            if vts.parameters["FaceAngleZ"]["value"] > 3:
                vts.parameters["FaceAngleZ"]["value"] -= randint(0, 10) / 5

            elif vts.parameters["FaceAngleZ"]["value"] < -3:
                vts.parameters["FaceAngleZ"]["value"] += randint(0, 10) / 5

            else:
                vts.parameters["FaceAngleZ"]["value"] += randint(-10, 10) / 5

            await asyncio.sleep(0.05)

        await asyncio.sleep(randint(1, 3))


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

    async for chunk in tts.infer("""Yoh, now why might you be looking for me, hm?
Oh, you didn't know? I'm the 77th Director of the Wangsheng Funeral Parlor, Hu Tao.
Though by the looks of you... Radiant glow, healthy posture...
Yes, you're definitely here for something other than that which falls within my regular line of work, aren't you?
Wanna come over for tea?
One client, two clients, three clients!
When the sun's out, bathe in sunlight. But when the moon's out, bathe in moonlight~
Lemme show you some fire tricks. First... Fire! And then... Whoosh! Fire butterfly! Be free!
Run around all you like during the day, but you should be careful during the night.
When I'm not around, best keep your wits about you.
Fighting's a pain. For me, it's not an objective so much as a means to an end.
Using the means to reach the end, to fight for that which I will not compromise on — it's in this way that you and I are the same."""):
        #sd.play(chunk["data"], chunk["sample_rate"])

        for volume_level in chunk["volume_data"]:
            vts.parameters["MouthOpen"]["value"] = volume_level

            await asyncio.sleep(0.05)

        vts.parameters["MouthOpen"]["value"] = 0

        await asyncio.sleep(0.3)

    await tts.close()
    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
