import asyncio
from vts import VTS


async def main():
    vts = VTS()
    await vts.connect()

    print([p for p in vts.parameters])

    vts.parameters["MouthSmile"]["value"] = 1
    vts.parameters["MouthOpen"]["value"] = 1

    vts.trigger(4)

    await asyncio.sleep(5)

    await vts.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
