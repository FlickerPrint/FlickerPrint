import platform
from shiny import App, render, ui, reactive
from shiny.types import ImgData, FileInfo
import webbrowser
# from pathlib import Path
from numpy import random 
import pandas as pd
import matplotlib.pyplot as plt
import shinyswatch
import sys
import pathlib
import os
import signal
plt2 = platform.system()
if plt2 == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# Modules
from modules.image_upload_module import image_upload_module_ui, image_upload_module_server
from modules.locator_module import locator_module_ui, locator_module_server
from modules.config_upload_module import config_upload_module_ui, config_upload_module_server



# UI
app_ui = ui.page_fluid(
        ui.page_sidebar(
            ui.sidebar(
                image_upload_module_ui("global_file_upload"),
                config_upload_module_ui("global_config_upload"),
                bg = "#F0F0F0",
                fillable=True,
            ),
        
        ui.panel_title("FlickerPrint Parameter Selection Tool", "FlickerPrint"),
        ui.navset_tab(
            # Nav elements
            ui.nav_panel("Condensate Detection", 
                # graph_module_ui(id="detection", label="Condensate Detection", plot_input_options=overlap_hist_plot_input_options)
                locator_module_ui("locator_module")
            ),
            ui.nav_spacer(),
            ui.nav_control(ui.input_action_button(id="exit", label="Close App", style = "color:#FF0000")),
            
        ),
        ),
)
   
# Server
def server(input, output, session):
    # Handle file upload
    microscope_image_reactive_value: reactive.Value[list[int]] = image_upload_module_server("global_file_upload")#TODO change the type here.

    # Handle config upload
    config_params_reactive_value: reactive.Value[dict] = config_upload_module_server(id="global_config_upload")

    locator_module_server(id="locator_module", microscope_image_reactive_value=microscope_image_reactive_value, config_params_reactive_value=config_params_reactive_value)

    # Handle shutting down the app
    @reactive.Effect
    @reactive.event(input.exit, ignore_none=True)
    async def stop_app():
        m = ui.modal(
            "You can safely close this browser tab.",
            title="Application Closed",
            easy_close=False,
            footer=None,
        )
        ui.modal_show(m)
        await session.app.stop()
        os.kill(os.getpid(), signal.SIGTERM)
        print("\n\nAppllication closed. It is now safe to close this terminal window. \n\n")
        
app = App(ui=app_ui, server=server)
webbrowser.open("http://127.0.0.1:8000", new=2) # Open web browser
