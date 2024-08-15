from contextlib import asynccontextmanager
import pyvts


class VTS:
    def __init__(self):
        self.vts = pyvts.vts()
        self.hotkeys: list = [] # not initialized yet
        self.parameters: list = [] # (default params) not initialized yet

    async def connect(self):
        await self.vts.connect()
        await self.vts.request_authenticate_token()
        await self.vts.request_authenticate()

        hotkey_data = await self.vts.request(self.vts.vts_request.requestHotKeyList())
        self.hotkeys = hotkey_data["data"]["availableHotkeys"]

        parameter_data = await self.vts.request(self.vts.vts_request.requestTrackingParameterList())
        self.parameters = parameter_data["data"]["defaultParameters"]

        await self.vts.close()

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
    
    async def set_parameter_value(self, parameter: str, value: float, weight: float = 1, face_found: bool = False, mode: str = "set") -> dict:
        """Set a parameter to a specific value.

        Args:
            parameter (str): Name of the parameter.
            value (float): Value of the data, [-1000000, 1000000]
            weight (float, optional): You can mix the your value with vts face tracking parameter, from 0 to 1,. Defaults to 1.
            face_found (bool, optional): if true, you will tell VTubeStudio to consider the user face as found, s.t. you can control when the "tracking lost". Defaults to False.
            mode (str, optional): Defaults to "set".

        Returns:
            dict: Vtube Studio API Response
        """
        return await self.vts.request(
            self.vts.vts_request.requestSetParameterValue(
                parameter,
                value,
                weight,
                face_found,
                mode
            )
        )
    
    async def set_parameter_values(self, parameters: list[str], values: list[float], weight: float = 1, face_found: bool = False, mode: str = "set") -> dict:
        """Set a parameter to a specific value.

        Args:
            parameters (list[str]): Name of the parameter.
            values (list[float]): Value of the data, [-1000000, 1000000]
            weight (float, optional): You can mix the your value with vts face tracking parameter, from 0 to 1,. Defaults to 1.
            face_found (bool, optional): if true, you will tell VTubeStudio to consider the user face as found, s.t. you can control when the "tracking lost". Defaults to False.
            mode (str, optional): Defaults to "set".

        Returns:
            dict: Vtube Studio API Response
        """
        return await self.vts.request(
            # checking the code and stuff
        )

    async def trigger(self, hotkey: int | dict) -> dict:
        """Triggers a hotkey.

        Args:
            hotkey (int | dict): Hotkey to trigger, can be the index or the hotkey dict.
        """
        if isinstance(hotkey, int):
            hotkey = self.hotkeys[hotkey]["name"]

        return await self.vts.request(self.vts.vts_request.requestTriggerHotKey(hotkey))
