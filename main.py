import asyncio
from vts import VTS


async def main():
    vts = VTS()
    await vts.connect()

    for i in range(len(vts.hotkeys)):
        await vts.trigger(i)
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
