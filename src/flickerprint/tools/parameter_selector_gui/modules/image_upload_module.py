import pandas as pd
import pickle as pkl
import h5py
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, render, ui, module, reactive
from shiny.types import FileInfo
from time import sleep

@module.ui
def image_upload_module_ui():
    return (
        ui.tooltip(
            ui.input_file("Microscope_Video", "Open Microscope Video", multiple=True),
            "Open microscope video files to begin! You can select multiple files at once.",
            id="microscope_video_upload_tool_tip",
            options={
                "show":True
            },
            show=True
        ),
        # ui.input_file("graunle_image_data", "Upload image data", accept=[".ims"], multiple=False)
    )

@module.server
def image_upload_module_server(input: Inputs, output: Outputs, session: render) -> reactive.Value[list[int]]:
    uploaded_file = reactive.Value()

    @reactive.Effect
    @reactive.event(input.Microscope_Video)
    def set_uploaded_file():
        """
            Gives the file paths of the uploaded microscope video files, to be read in later.
            Sets the results to the reactive value container.
        """
        f: list[FileInfo] = input.Microscope_Video()
        # Map original filenames to their temporary datapaths
        files: dict[str, str] = {fi["name"]: fi["datapath"] for fi in f}
        uploaded_file.set(files)
        

    return uploaded_file