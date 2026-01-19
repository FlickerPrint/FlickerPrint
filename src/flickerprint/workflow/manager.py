#!/usr/bin/env python
""" Provide a central script to handle the running of the other commands.

This is to prevent the package from having to provide too many executables to the user.
Preventing collisions and making it simpler to use cli help.
"""

import argparse
from pathlib import Path
from os.path import dirname

from shiny import run_app

from flickerprint.workflow import process_image, extract_physical_values, bayesian_optimisation
from flickerprint.common import create_project_dir
from flickerprint.analysis import cli_analysis, gui
from flickerprint.tools import parameter_selector_gui
import flickerprint.version as version


def run(cores: int = 1):
    """A helper function for automatic running of the main FlickerPrint workflow.
    This handles the image processing and spectrum fitting steps.
    """
    process_image.main(cores=cores)
    extract_physical_values.main(working_dir=Path('.'), cores=cores)
    cli_analysis.main(input_file=Path('./aggregate_fittings.h5'), output_dir=Path('./figs'))

def run_gui():
    """A helper function for starting the graphical user interface."""
    gui_path  = dirname(gui.__file__)
    print(gui_path)
    run_app('app:app', app_dir=str(gui_path))

def run_parameter_selector_gui():
    """A helper function for starting the parameter selection graphical user interface."""
    parameter_selector_gui_path = dirname(parameter_selector_gui.__file__)
    print(parameter_selector_gui_path)
    run_app('app:app', app_dir=str(parameter_selector_gui_path))

def parse_arguments():
    """ Read command line arguments.

    This breaks the arguments into groups, each providing arguments to a different
    workflow.
    """
    module_description = f"Here is the description for the modules"
    workflow_description = f"Description for the workflows"

    parser = argparse.ArgumentParser(description=module_description)
    parser.set_defaults(func=parser.print_help)

    subparsers = parser.add_subparsers(
        title="Workflows", description=workflow_description
    )

    #
    # Boundary Extraction
    #
    parser_process_image = subparsers.add_parser(
        "process-image", help="Extract the granule boundaries from the image."
    )
    parser_process_image.add_argument(
        "--input_image", type=Path, help="Path to input image.", default=None
    )
    parser_process_image.add_argument(
        "-o",
        "--output_dir",
        default=".",
        type=Path,
        help="Directory for the output files.",
    )
    parser_process_image.add_argument(
        "-q", "--quiet", action="store_true", help="Supress progress bar."
    )
    parser_process_image.add_argument(
        "--max-frame",
        type=int,
        default=None,
        help="Stop the analysis on this frame. Used for debugging.",
    )
    parser_process_image.add_argument(
        "-c","--cores",
        type=int,
        default=1,
        help="Number of cores to use for multiprocessing. Default is 1. Not required for single files.")
    
    parser_process_image.set_defaults(func=process_image.main)

    #
    # Spectrum Fitting
    #
    parser_spectrum = subparsers.add_parser(
        "spectrum-fitting",
        help=(
            "Calculate the properties of the granule by fitting the experimental "
            "spectrum to a theoretical model."
        ),
    )
    parser_spectrum.add_argument("working_dir", type=Path, help="Experiment directory")
    parser_spectrum.add_argument("--plotting", action="store_true")
    parser_spectrum.add_argument(
        "-c", "--cores", type=int, default=1, help="Number of cores to use"
    )
    parser_spectrum.set_defaults(func=extract_physical_values.main)

    #
    # Project creation
    #
    parser_project = subparsers.add_parser(
        "create-project", help="Create the directory structure for the project."
    )
    parser_project.add_argument(
        "project_dir", type=Path, help="Location of the new project",
    )
    parser_project.add_argument(
        "-d",
        "--dry",
        action="store_true",
        help="Don't create the directories, instead print expected files",
    )
    parser_project.add_argument(
        "-p", "--parent", action="store_false", help="Check for parent directory"
    )
    parser_project.set_defaults(func=create_project_dir.main)

    #
    # Version
    #

    parser_version = subparsers.add_parser(
        "version", help="Print version number of current code"
    )
    parser_version.set_defaults(func=_get_version)

    #
    # Parameter estimation
    #
    parser_bayes = subparsers.add_parser(
        "parameter-estimation", help="Estimate parameters for image analysis using samples from one experiment."
    )
    parser_bayes.add_argument(
        "experiment_dir", type=Path, help="Location of the project", default="."
    )
    parser_bayes.set_defaults(func=bayesian_optimisation.main)


    # 
    # Automated Workflow
    # 
    parser_run = subparsers.add_parser(
        "run", help="Run the full FlickerPrint workflow."
    )
    parser_run.add_argument(
        "-c", "--cores", type=int, default=1, help="Number of cores to use"
    )
    parser_run.set_defaults(func=run)

    # 
    # Graphing GUI
    # 

    parser_gui = subparsers.add_parser(
        "view-output", help="Start the graphical user interface to view the parameter distributions."
    )
    parser_gui.set_defaults(func=run_gui)

    # 
    # Command line graphing
    # 
    parser_cli_graph = subparsers.add_parser(
        "view-output-terminal", help="View the parameter distributions using the command line."
    )
    parser_cli_graph.add_argument(
        "input_file", type=Path, help="Path to input file."
    )
    parser_cli_graph.add_argument(
        "-o",
        "--output_dir",
        default=".",
        type=Path,
        help="Directory for the output files.",
    )
    parser_cli_graph.add_argument(
        "-b", "--bins", type=int, default=8, help="Number of bins for histogram."
    )
    parser_cli_graph.add_argument(
        "-d", "--density", type=str, default=False, help="Plot density histograms."
    )
    parser_cli_graph.add_argument(
        "-l", "--latex", type=str, help="Use latex for plotting.", default=True
    )
    parser_cli_graph.add_argument(
        "-i", "--img_path_filter", type=str, help="Filter the image paths."
    )
    parser_cli_graph.set_defaults(func=cli_analysis.main)

    #
    # Parameter Selection GUI
    #
    parser_parameter_selector_gui = subparsers.add_parser(
        "parameter-selection", help="Start the parameter selection graphical user interface."
    )
    parser_parameter_selector_gui.set_defaults(func=run_parameter_selector_gui)


    return parser.parse_args()

def _get_version():
    print(version.__version__)

def dispatch_args(args: argparse.Namespace):
    """ Run the function with the provided kwargs.

    This removes the global variables and the ``func`` key, so that we may pass the rest
    of the command line arguments directly to the function to be called.
    """
    # Extra values may be excluded, but "func" must always be excluded
    excluded_args = ["func", "first"]
    func_kwargs = {}

    for key, value in args.__dict__.items():
        if key not in excluded_args:
            func_kwargs[key] = value

    args.func(**func_kwargs)

def main():
    """The main helper function for running FlickerPrint analysis directly from the command line."""
    args = parse_arguments()
    dispatch_args(args)


if __name__ == "__main__":
    main()
