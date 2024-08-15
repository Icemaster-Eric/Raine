import pyvts


class VTS:
    def __init__(self):
        self.vts = pyvts.vts()
        self.hotkeys = [] # not initialized yet

    async def connect(self):
        await self.vts.connect()
        await self.vts.request_authenticate_token()
        await self.vts.request_authenticate()

        hotkey_data = await self.vts.request(self.vts.vts_request.requestHotKeyList())
        self.hotkeys = hotkey_data["data"]["availableHotkeys"]

        await self.vts.close()

    async def trigger(self, hotkey: int | dict):
        """Triggers a hotkey.

        Args:
            hotkey (int | dict): Hotkey to trigger, can be the index or the hotkey dict.

        Raises:
            RuntimeError: Raises this error whenever an APIError is returned.
        """
        await self.vts.connect()
        await self.vts.request_authenticate()

        if isinstance(hotkey, int):
            hotkey = self.hotkeys[hotkey]["name"]

        response = await self.vts.request(self.vts.vts_request.requestTriggerHotKey(hotkey))

        if response["messageType"] == "APIError":
            print(hotkey)
            raise RuntimeError(response["data"]["message"])

        await self.vts.close()
