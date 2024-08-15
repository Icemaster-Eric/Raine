import asyncio
from vts import VTS


async def main():
    vts = VTS()
    await vts.connect()

    print([p["name"] for p in vts.parameters])

    while True:
        param_name, value = input(": ").split()
        value = float(value)

        async with vts.vts_request():
            await vts.set_parameter_value(param_name, value)


if __name__ == "__main__":
    asyncio.run(main())
