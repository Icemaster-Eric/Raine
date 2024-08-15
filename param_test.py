from nicegui import app, ui, events
from vts import VTS


vts = VTS()


async def stop_vts():
    await vts.disconnect()


async def startup():
    await vts.connect()

    with ui.grid(columns=6):
        for param_name, param_data in vts.parameters.items():
            ui.label(param_name)
            ui.slider(min=param_data["min"], max=param_data["max"], value=param_data["value"], step=(param_data["max"]-param_data["min"])/10, on_change=lambda e, pn=param_name: change_param(e, pn))


def change_param(e: events.ValueChangeEventArguments, param_name: str):
    vts.parameters[param_name]["value"] = e.value


app.on_startup(startup)
app.on_shutdown(stop_vts)


ui.run()
