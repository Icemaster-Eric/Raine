from typing import Literal
from contextlib import asynccontextmanager
import asyncio
import pyvts


class VTS:
    def __init__(self):
        self.connected: bool = False
        self.vts = pyvts.vts()
        self.hotkeys: list = []
        self.parameters: dict[str, dict[
            Literal["value", "min", "max", "defaultValue"], float
        ]] = {} # (change "value" to change the model's parameter values)
        self.request_queue = asyncio.Queue()

    async def connect(self):
        await self.vts.connect()
        await self.vts.request_authenticate_token()
        await self.vts.request_authenticate()

        hotkey_data = await self.vts.request(self.vts.vts_request.requestHotKeyList())
        self.hotkeys = hotkey_data["data"]["availableHotkeys"]

        parameter_data = await self.vts.request(self.vts.vts_request.requestTrackingParameterList())
        self.parameters = {
            param["name"]: {
                "value": param["value"],
                "defaultValue": param["defaultValue"],
                "min": param["min"],
                "max": param["max"]
            } for param in parameter_data["data"]["defaultParameters"]
        }

        await self.vts.close()

        self.connected = True

        asyncio.create_task(self.send_requests())

    async def disconnect(self):
        if self.vts.get_connection_status():
            await self.vts.close()

        self.hotkeys: list = []
        self.parameters = {}

        self.connected = False

    @asynccontextmanager
    async def vts_request(self):
        """Call this before making a vts request.

        ```python
        # example usage
        myvts = VTS()
        with myvts.vts_request():
            myvts.some_request()
        ```
        """
        await self.vts.connect()
        await self.vts.request_authenticate()
        yield
        await self.vts.close()

    async def set_parameter_values(self, parameters: list[str], values: list[float], weights: list[float] | float = 1, face_found: bool = False, mode: str = "set") -> dict:
        """Set a list of parameters to specific values.

        Args:
            parameters (list[str]): Name of the parameter.
            values (list[float]): Value of the data, [-1000000, 1000000]
            weight (list[float] | float, optional): You can mix the your value with vts face tracking parameter, from 0 to 1. Defaults to 1.
            face_found (bool, optional): if true, you will tell VTubeStudio to consider the user face as found, s.t. you can control when the "tracking lost". Defaults to False.
            mode (str, optional): Defaults to "set".

        Returns:
            dict: Vtube Studio API Response
        """
        return await self.vts.request(
            self.vts.vts_request.requestSetMultiParameterValue(
                parameters,
                values,
                weights,
                face_found,
                mode
            )
        )

    async def send_requests(self):
        """Constantly sends requests to vtube studio to keep the model at the current parameters.

        Stops when the connected property is set to False.
        """
        async with self.vts_request():
            while self.connected:
                while not self.request_queue.empty():
                    request = self.request_queue.get_nowait()

                    await self.vts.request(request)

                await self.set_parameter_values(
                    [param_name for param_name in self.parameters.keys()],
                    [param["value"] for param in self.parameters.values()],
                )

                await asyncio.sleep(0.3)

    def trigger(self, hotkey: int | dict) -> dict:
        """Triggers a hotkey.

        Args:
            hotkey (int | dict): Hotkey to trigger, can be the index or the hotkey dict.
        """
        if isinstance(hotkey, int):
            hotkey = self.hotkeys[hotkey]["name"]

        self.request_queue.put_nowait(self.vts.vts_request.requestTriggerHotKey(hotkey))
