from shiny import App, Inputs, Outputs, Session, module, render, ui, reactive
from shiny.types import ImgData, FileInfo
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as co
import io
import numpy as np
from typing import Callable
from pathlib import Path
from itertools import islice



import flickerprint.common.frame_gen as fg
import flickerprint.common.granule_locator as gl
import flickerprint.common.configuration as cfg

@module.ui
def locator_module_ui():
    return ui.layout_column_wrap(
        ui.row(
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Raw Image"),
                    ui.output_plot("raw_plot", height="50vh")
                ),
                ui.card(
                    ui.card_header("Detected Condensates"),
                    ui.output_plot("detected_plot", click=True, height="50vh")
                )
            ),
            ui.row(
            ui.column(2,
                ui.card(
                    ui.input_select(
                        id = "image_selector",
                        label = "Select an Image to Display",
                        choices = [],
                        multiple=False
                    ),
                    ui.input_text(
                        id="frame_number",
                        label="Frame Number",
                        value = "1",
                        placeholder='1'
                    ),
                    ui.tooltip(
                    ui.input_text(
                        id='pixel_size', 
                        label='Pixel Size (µm)', 
                        value='0.0', 
                        placeholder='0.0'
                    ),
                    "When working with .tif[f] files, please enter the pixel size to enable accurate physical measurements.",
                    id="pixel_size_tool_tip",
                    options={
                        "show":True
                    },
                    show=True
                    )
                ),
            ),
            ui.column(8,
                ui.row(
                    ui.card(
                        ui.layout_column_wrap(
                            *[ui.input_slider(
                                id = param["id"],
                                label = param["label"],
                                min = param["min"],
                                max = param["max"],
                                value = param["value"],
                                step = param["step"],
                                post=param["post"]) for param in parameters],
                            ui.input_select(
                                id = "detection_method",
                                label = "Detection Method",
                                choices = ["gradient", "intensity"],
                                selected = "gradient",
                                multiple=False),
                                min_width="250px"
                        )
                    )
                ),
                ui.row(
                    ui.card(
                        ui.layout_column_wrap(
                            ui.input_text(
                                id='image_path',
                                label='Image Path',
                                value='default_images',
                                placeholder='default_images'
                            ),
                            ui.input_text(
                                id='image_regex',
                                label='Image Regular Expression',
                                value='*',
                                placeholder='*'
                            ),
                            ui.input_text(
                                id='experiment_name',
                                label='Experiment Name',
                                value='experiment_name',
                                placeholder='experiment_name'
                            ),
                            ui.input_text(
                                id='temperature',
                                label='Experiment Temperature (ºC)',
                                value='37',
                            ),
                            ui.input_select(
                                id='granule_images',
                                label='Plot Detector Images',
                                choices=['True', 'False'],
                                selected='False',
                                multiple=False
                            ),
                            ui.input_select(
                                id='plot_spectra_and_heatmaps',
                                label='Plot Fluctuation Spectra',
                                choices=['True', 'False'],
                                selected='False',
                                multiple=False
                            ),
                            min_width="250px"
                        )
                    )
                ),
            ),
            ui.column(2,
                ui.card(
                    ui.input_text(
                        id='save_path', 
                        label='Save Path', 
                        value='~/Downloads/', 
                        placeholder='Path to directory for saving config file' ),
                    ui.input_action_button(
                        id="save_parameters",
                        label="Save Parameters",
                        style="background-color: #b6e0b1; padding: 8px 8px;"
                    ),
                    ui.input_action_button(
                        id="reset_parameters",
                        label="Reset Parameters",
                        style="padding: 8px 8px;"
                    )
                ),
            )
        )
        ),
    )
parameters = [{
    "id": "min_size_slider",
    "label": "Condensate Minimum Radius (µm)",
    "min": 0,
    "max": 20,
    "value": 0.3,
    "step": 0.1,
    "post": " µm"
},
{
    "id": "max_size_slider",
    "label": "Condensate Maximum Radius (µm)",
    "min": 0,
    "max": 20,
    "value": 3.0,
    "step": 0.1,
    "post": " µm"
},
{
    "id": "intensity_threshold_slider",
    "label": "Intensity Threshold",
    "min": 0,
    "max": 1,
    "value": 0.1,
    "step": 0.001,
    "post": ""
},
{
    "id": "fill_threshold_slider",
    "label": "Fill Threshold",
    "min": 0,
    "max": 1,
    "value": 0.6,
    "step": 0.01,
    "post": ""
}
]

def plot_raw_image(frame):
    fig, ax = plt.subplots()
    ax.imshow(frame, cmap='gray')
    ax.axis('off')
    return fig


def run_detection(frame, imaging_parameters):
    detector = gl.GranuleDetector(frame)
    threshold = imaging_parameters["intensity_threshold"]
    min_size = imaging_parameters["min_size"]
    max_size = imaging_parameters["max_size"]
    fill_threshold = imaging_parameters["fill_threshold"]
    detection_method = imaging_parameters["detection_method"]
    
    try:
        detector.labelGranules(intensity_threshold=threshold, min_size=min_size, max_size=max_size, fill_threshold=fill_threshold, method=detection_method)
    except gl.GranuleNotFoundError:
        raise gl.GranuleNotFoundError("No condensates found with the current parameters.")
    
    fig, ax = plt.subplots()
    detector.plot(ax=ax, cmap='gray')
    ax.axis('off')
    return fig

@module.server
def locator_module_server(input: Inputs, output: Outputs, session: render, microscope_image_reactive_value: reactive.Value[dict[str, Path]], config_params_reactive_value: reactive.Value[dict] = None) -> None:

    selected_frame = reactive.Value()

    @reactive.Effect
    @reactive.event(config_params_reactive_value)
    def update_ui_from_config():
        """
            Updates UI elements when a config file is uploaded.
        """
        if config_params_reactive_value is None:
            return
        
        params = config_params_reactive_value()
        if not params:
            return
        
        # Update all UI elements with values from config
        ui.update_text("image_path", value=params["image_path"], session=session)
        ui.update_text("pixel_size", value=str(params["pixel_size"]), session=session)
        ui.update_selectize("detection_method", selected=params["detection_method"], session=session)
        ui.update_slider("fill_threshold_slider", value=params["fill_threshold_slider"], session=session)
        ui.update_text("experiment_name", value=params["experiment_name"], session=session)
        ui.update_text("image_regex", value=params["image_regex"], session=session)
        ui.update_text("temperature", value=str(params["temperature"]), session=session)
        ui.update_selectize("granule_images", selected=params["granule_images"], session=session)
        ui.update_selectize("plot_spectra_and_heatmaps", selected=params["plot_spectra_and_heatmaps"], session=session)
        ui.update_slider("min_size_slider", value=params["min_size_slider"], session=session)
        ui.update_slider("max_size_slider", value=params["max_size_slider"], session=session)
        ui.update_slider("intensity_threshold_slider", value=params["intensity_threshold_slider"], session=session)

    @reactive.Effect
    @reactive.event(input.image_selector, input.frame_number)
    def update_selected_frame():
        """
            Updates the selected frame based on the selected image and frame number.
        """
        files = microscope_image_reactive_value()
        if not files:
            print("No files uploaded yet.")
            return
        
        selected_image_name = input.image_selector()
        input_frame_number = 1 if input.frame_number() is '' else input.frame_number()
        frame_number = int(input_frame_number) - 1 # Convert to 0-based index

        if selected_image_name not in files:
            return
        
        image_path = files[selected_image_name]
        try:
            image_frames = islice(fg.gen_opener(image_path), frame_number, frame_number + 1)
            frame = next(image_frames)
        except Exception as e:
            m = ui.modal(
                ui.p(f"An error occurred when opening the following image:"),
                ui.p(ui.span(f"{input.image_selector()}", style="font-weight: bold;")),
                ui.br(),
                ui.p(f"Error Message:"), 
                ui.p(ui.span(f"{e}", style="font-weight: bold; color:#B11226")),
                ui.p("Please remove this image and try again."),
                ui.br(),
                ui.input_action_button(
                    "remove_image_button",
                    "Remove Image from the Application"
                ),
                title="Application Error",
                easy_close=False,
                footer=None,
            )
            ui.modal_show(m)
            return
        selected_frame.set(frame)

    @output
    @render.plot
    @reactive.event(selected_frame)
    def raw_plot():
        frame = selected_frame()
        fig = plot_raw_image(frame.im_data)
        return fig
    
    @output
    @render.plot
    @reactive.event(selected_frame, input.min_size_slider, input.max_size_slider, input.intensity_threshold_slider, input.fill_threshold_slider, input.detection_method)
    def detected_plot():
        frame = selected_frame()
        
        imaging_parameters = {
            "min_size": input.min_size_slider(),
            "max_size": input.max_size_slider(),
            "intensity_threshold": input.intensity_threshold_slider(),
            "fill_threshold": input.fill_threshold_slider(),
            "detection_method": input.detection_method()
        }

        fig = run_detection(frame, imaging_parameters)
        
        return fig
    
    @reactive.Effect
    @reactive.event(microscope_image_reactive_value)
    def update_image_selector():
        """
            Updates the image selector choices based on the uploaded files.
        """
        files = microscope_image_reactive_value()
        if not files:
            return
        
        image_names = list(files.keys())
        ui.update_selectize(id="image_selector", choices=image_names, selected=image_names[0] if image_names else None)

    @reactive.Effect
    @reactive.event(input.reset_parameters)
    def reset_parameters():
        """
            Resets the detection parameters to their default values.
        """
        for param in parameters:
            ui.update_slider(id=param["id"], value=param["value"])
        ui.update_selectize(id="detection_method", selected="gradient")

    @reactive.Effect
    @reactive.event(input.save_parameters)
    def save_parameters():
        """
            Saves the current detection parameters to a file.
        """
        imaging_parameters = {
            "granule_minimum_radius": input.min_size_slider(),
            "granule_maximum_radius": input.max_size_slider(),
            "granule_minimum_intensity": input.intensity_threshold_slider(),
            "fill_threshold": input.fill_threshold_slider(),
            "method": input.detection_method(),
            "pixel_size": float(input.pixel_size()),
            "image_dir": input.image_path(),
            "image_regex": input.image_regex(),
            "experiment_name": input.experiment_name(),
            "temperature": float(input.temperature()),
            "granule_images": input.granule_images(),
            "plot_spectra_and_heatmaps": input.plot_spectra_and_heatmaps(),
        }
        input_save_path = input.save_path() if input.save_path() else "~/Downloads/"
        save_path = input_save_path + "/" + "config.yaml"
        try:
            save_path = str(Path(save_path).expanduser())
            cfg.write_config(imaging_parameters, Path(save_path))
            m=ui.modal(
                ui.p("Parameters saved successfully to:"),
                ui.p(ui.span(f"{save_path}", style="font-weight: bold;")),
                title="Save Successful",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
        except FileNotFoundError:
            m=ui.modal(
                ui.p("Error: Could not save parameters to the specified path."),
                ui.p(ui.span(f"{save_path}", style="font-weight: bold;")),
                ui.br(),
                ui.p("Please check that the directory exists and you have permission to access it."),
                title="Save Failed",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
        

    @reactive.Effect
    @reactive.event(input.remove_image_button)
    def remove_image():
        """
            Removes the selected image from the uploaded files.
        """
        files = microscope_image_reactive_value()
        selected_image_name = input.image_selector()
        if selected_image_name in files:
            del files[selected_image_name]
            microscope_image_reactive_value.set(files)
        ui.update_selectize(id="image_selector", choices=list(files.keys()), selected=None)
        ui.modal_remove()