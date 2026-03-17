import pandas as pd
import pickle as pkl
import h5py
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, render, ui, module, reactive
from shiny.types import FileInfo
from time import sleep

from flickerprint.common.configuration import config

@module.ui
def config_upload_module_ui():
    return (
        ui.tooltip(
            ui.input_file("Config_File", "Open Configuration File", multiple=False),
            "If you would like to edit an existing configuration file, please open it here.",
            id="config_upload_tool_tip",
            options={
                "show":True
            },
            show=True
        ),
        # ui.input_file("graunle_image_data", "Upload image data", accept=[".ims"], multiple=False)
    )

@module.server
def config_upload_module_server(input: Inputs, output: Outputs, session: Session) -> reactive.Value[dict]:
    config_params = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.Config_File)
    def update_parameters():
        # Read the config file and update the UI elements accordingly
        config_file: list[FileInfo] = input.Config_File()
        if not config_file:
            return
        config_path = Path(config_file[0]["datapath"])
        config.refresh(config_path)
        
        ui_params = {
            "image_path": str(config("workflow", "image_dir")),
            "pixel_size": float(config("image_processing", "pixel_size")),
            "detection_method": str(config("image_processing", "method")),
            "fill_threshold_slider": float(config("image_processing", "fill_threshold")),
            "experiment_name": str(config("workflow", "experiment_name")),
            "image_regex": str(config("workflow", "image_regex")),
            "temperature": float(config("spectrum_fitting", "temperature")),
            "granule_images": str(config("image_processing", "granule_images")),
            "plot_spectra_and_heatmaps": str(config("spectrum_fitting", "plot_spectra_and_heatmaps")),
            "min_size_slider": float(config("image_processing", "granule_minimum_radius")),
            "max_size_slider": float(config("image_processing", "granule_maximum_radius")),
            "intensity_threshold_slider": float(config("image_processing", "granule_minimum_intensity")),
        }
        print(ui_params)
        
        # Store the config parameters in the reactive value
        config_params.set(ui_params)
    
    return config_params
