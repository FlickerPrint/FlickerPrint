import pandas as pd
import pickle as pkl
import h5py
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, render, ui, module, reactive
from shiny.types import FileInfo


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
def image_upload_module_server(input: Inputs, output: Outputs, session: render):
    uploaded_file = reactive.Value({})

    @reactive.effect
    @reactive.event(input.Microscope_Video)
    def set_uploaded_file():
        """
        Gives the file paths of the uploaded microscope video files, to be read in later.
        Cleans up previously uploaded files from disk, then sets the new files.
        """  
        f = input.Microscope_Video()

        old_files = uploaded_file.get()
        
        if old_files:
            for _name, old_path in old_files.items():
                p = Path(old_path)
                if p.exists():
                    try:
                        p.unlink()
                    except Exception as e:
                        continue

        new_files = {fi["name"]: fi["datapath"] for fi in f}
        
        uploaded_file.set(new_files)

    return uploaded_file