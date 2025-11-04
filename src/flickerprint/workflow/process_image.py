#!/usr/bin/env python

""" Extract the Fourier components of all the granules in a time series.

Outline
-------

The first step calculates the rough shape and position of the granules in the frame,
from this we score the granules and rank them.

Of the granules that pass, we then extract the border and use this to generate the
Fourier terms.

Method
------

We use the ``frame_gen`` object to return the image frame by frame, this is required
in order to properly handle metadata and the proprietary microscope formats (and the
resultant JavaVM).

For each frame we isolate the granules from the background image using
``granule_locator``, by default this is uses a Difference of Gassian method. From each
of these granules we draw the boundary around it using the methods in
``boundary_extraction``. The final step is then to track the granules across the frames,
accounting for both the creation and deletion of granules.

There is the option to create plots of both the granule detection and the boundary
drawing, which is configured in the config file. 
We store all of the Fourier terms, into a ``.hdf5`` database, along with links
to any images created and metadata for each frame.

Multiprocessing
---------------

The combination of multiprocessing and javabridge often lead to large memory leaks,
particularly when initialising the worker pool. This means that only a single core can be used
to analyse a single microscope image. However, the analysis is typically fast enough that only
using a single core on a single microscope image is not a problem.

Parallelisation can be achieved by passing a directory of images into main() and setting the 
number of cores to use with the -c flag. This will analyse each image in a separate process 
and save the results as normal. Resources are allocated dynamically so only the required number
of cores are used, up to a maximum of the number specified with the -c flag.

"""

import argparse
from pathlib import Path
from functools import partial

import h5py
import subprocess
import platform
import pandas as pd
import pickle as pkl
import tqdm
import os
import warnings
import multiprocessing as mp
from time import sleep
from more_itertools import peekable
import numpy as np
import matplotlib.pyplot as plt

from flickerprint.common.utilities import strtobool
import flickerprint.common.boundary_extraction as be
import flickerprint.common.frame_gen as fg
import flickerprint.common.granule_locator as gl
import flickerprint.common.granule_locator_fft as glf
import flickerprint.tools.plot_tools as pt
from flickerprint.common.configuration import config
import flickerprint.version as version


def parse_arguments():
    """ Read command line arguments. """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--input", type=Path, help="Path to input image or directory of input images.", default=None)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=".",
        help="Directory for the output files.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Supress progress bar."
    )

    parser.add_argument(
        "--max-frame",
        type=int,
        default=None,
        help="Stop the analysis on this frame. Used for debugging.",
    )

    parser.add_argument(
        "-c","--cores",
        type=int,
        default=1,
        help="Number of cores to use for multiprocessing. Default is 1. Not required for single files.")

    args = parser.parse_args()
    return args

def main(
        input_image: Path | str = None, output_dir: Path | str = ".", quiet: bool = False, max_frame: int = None, cores = 1
):
    """
    Takes an image or a directory of images and processes them to extract the granule boundaries and Fourier terms.
    These are retruned as a .h5 file in the experiment's 'fourier' directory.

    Parameters:
    -----------

    input_image: Path | str 
        The path to the image or directory of images to be processed. If a directory is provided, all images in the directory will be processed.
        If 'default_images' is provided, the default images in the 'images' directory will be processed.
        If no input is provided, the default image directory set in the config file will be used.
    
    output_dir: Path | str
        The directory to save the output files to. If no output directory is provided, the current directory will be used.

    quiet: bool
        If True, the progress bar will be suppressed. Default is `False`.
    
    max_frame: int
        The maximum number of frames to process. Default is `None` which processes all frames.
    
    cores: int
        The number of cores to use for multiprocessing. Default is 1. Only required if a directory of images is provided.
        If the number of cores requested exceeds the number of available cores, the number of available cores will be used instead.

    Debugging images to show the location and boundary of the detected granules. These images are saved in the 'tracking' directory in the 'detection' and 'outline' subdirectories.
    Debugging images can be configured using the 'granule_images' parameter in the config file.

    """

    print(f"\n================\nImage Processing\n================\n")

    config_location = Path(output_dir) / "config.yaml"
    print(f"\nConfiguration file location: {config_location}")
    config.refresh(config_location)

    tracking_threshold = float(config("image_processing", "tracking_threshold"))
    if tracking_threshold < 10 or tracking_threshold > 15:
        warnings.warn(
            f"Tracking threshold is set to {tracking_threshold}, which is outside the recommended range of 10 <= threshold <= 15. This may lead to poor tracking results.",
            UserWarning,
        )

    # Pull any required args from the config file
    if input_image is None:
        input_image = str(config("workflow", "image_dir"))
        if input_image == "":
            raise ValueError("No input image provided and no default image directory set in the config file.")
    
    if input_image == "default_images":
            input_image = "./images"
    input_image = Path(input_image)

    if input_image.is_dir():

        image_regex = [str(config("workflow", "image_regex"))]
        if image_regex == [""]:
            warnings.warn("No image suffix provided in the config file. Defaulting to allow all supported file formats.", UserWarning)
            image_regex = ["*.tif", "*.tiff", "*.png", "**.ome.tiff", "*.ims", "*.lif", "*.tiff.ome", "*.tif.ome"]

        files = []

        for regex in image_regex:
            files.extend(Path(input_image).glob(regex))
        # Need to change this to be a regex.

        if files == []:
            raise FileNotFoundError(f"No images found in {input_image} with the provided regex: {image_regex}")

        if cores > os.cpu_count():
            cores = os.cpu_count()
            warnings.warn(f"Number of cores requested exceeds available cores. Only {os.cpu_count()} cores are available.", UserWarning)
        if cores > len(files):
            cores = len(files)
        if cores == 1:
            print(f"Using 1 core")
        else:
            print(f"Using {cores} cores")
        
        print(f"Image directory: {str(input_image)}")
        print(f"Number of images to process: {len(files)}\n")
        
        with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
            # This handles the multiprocessing stage.
            # Since the JVM is not thread safe, we need to analyse each image in it's own process. 
            args = []
            for pbar_bos, file in enumerate(files):
                    args.append((Path(file), Path(output_dir), quiet, max_frame, pbar_bos))
            pool.starmap(process_single_image, args)
            
    else:
        # If there is only one image, then just process it directly.
        if cores != 1:
            print("Using 1 core as only a single image to be analysed.")
        print(f"\n")
        process_single_image(input_image, output_dir, quiet, max_frame)

    if bool(strtobool(config("image_processing", "granule_images"))):
        # If the debug images are saved, zip them up at the end to make them easier to transfer.
        try:
            detection_return = subprocess.call(f"zip -r detection.zip detection", shell =True, cwd=f"{output_dir}/tracking", stdout=subprocess.DEVNULL)
            outline_return = subprocess.call(f"zip -r outline.zip outline", shell =True, cwd=f"{output_dir}/tracking", stdout=subprocess.DEVNULL)
            if detection_return == 0 and outline_return == 0:
                subprocess.call(f"rm -rf outline", shell =True, cwd=f"{output_dir}/tracking", stdout=subprocess.DEVNULL)
                subprocess.call(f"rm -rf detection", shell =True, cwd=f"{output_dir}/tracking", stdout=subprocess.DEVNULL)
                subprocess.call(f"mkdir tracking/outline tracking/detection", shell=True)
            else:
                print("Zipping detection and outline images unsuccessful. Images will be available as separate files instead.")
            subprocess.call(f"cd {output_dir}", shell=True)
        except Exception:
            subprocess.call(f"cd {output_dir}", shell=True)
            print("Zipping detection and outline images unsuccessful. Images will be available as separate files instead.")
    print(f"\n\nFourier analysis complete\n-------------------------\n")

@fg.vmManager
def process_single_image(
    input_image: Path, output_dir: Path, quiet: bool = False, max_frame: int = None, _pbar_pos: int = 0
):
    """
    Locates the granules in a single image and extracts the Fourier terms. The Fourier terms are written to a .h5 file in the 'fourier' directory.
    
    Parameters:
    -----------
    
    input_image: Path
        The path to the image to be processed.
        
    output_dir: Path
        The directory to save the output files to.
        
    quiet: bool
        If True, the progress bar will be suppressed. Default is False.
    
    max_frame: int
        The maximum number of frames to process. Default is None which processes all frames.
    
    _pbar_pos: int
        (Internal use only) The position of the progress bar. Default is None. Only required for multiprocessing.

    Debugging images to show the location and boundary of the detected granules. These images are saved in the 'tracking' directory in the 'detection' and 'outline' subdirectories.
    Debugging images can be configured using the 'granule_images' parameter in the config file.
    """

    config_location = Path(output_dir) / Path("config.yaml")
    config.refresh(config_location)
    output_dir = Path(output_dir)

    validate_args(input_image, output_dir, quiet)
    # Making the generator peekable allows us to get the first item for metadata without advancing
    # the iterator. Looking at the implementation, this shouldn't cost us any extra
    # memory/performance
    image_frames = peekable(fg.gen_opener(input_image))
    
    use_fft = bool(strtobool(config("image_processing", "use_fft")))
    if use_fft:
        frame = image_frames.peek()
        blurrer = _get_blurrer(frame)
        detector_function = partial(glf.GranuleDetectorFFT, blurrer=blurrer)
    else:
        detector_function = gl.GranuleDetector

    fourier_frames = []
    max_distance = float(config("image_processing", "tracking_threshold"))
    granule_tracker = be._GranuleLinker(memory=10,max_distance=max_distance)

    print(f"#{_pbar_pos+1} Working on image: {input_image}")
    # Add a 0.5 second sleep to ensure that the progress bars appear in the correct place.
    sleep(0.5)
    # Set up a process bar to track the frame counts.
    disable_bar = True if quiet else None
    process_bar = tqdm.tqdm(enumerate(image_frames), disable=disable_bar, position=_pbar_pos, unit="frame", desc=f"#{_pbar_pos+1}")

    for frame_num, frame in process_bar:
        # Update the progress bar to account for the number of frames
        if frame_num == 0 and not quiet:
            total_frames = frame.total_frames if max_frame is None else max_frame
            process_bar.reset(total_frames)

        if bool(strtobool(config("image_processing", "granule_images"))):
            plot_frame = frame_num % 100 == 0
        else:
            plot_frame = False

        # I've found the individual plotting of granule boundaries to be much less helpful than the
        # frame overview (and take much more space), so splitting the varibles here. For now this
        # just behaves as an easy place to override this, but could be moved out to the config
        # later.
        plot_granules = plot_frame

        detector = detector_function(frame)

        # Detect the granules within the frame
        try:
            detector.labelGranules()
        except gl.GranuleNotFoundError:
            if frame_num == 0:
                print("No granules found on first frame, quitting")
                process_bar.close()
                raise gl.GranuleNotFoundError(
                    f"\n\nNo granules found in {input_image}. Please check the values in the config file and try again.")
            else:
                continue

        figure_dir = output_dir / "tracking"
        figure_dir.mkdir(exist_ok=True)
        detection_dir = figure_dir / "detection"
        detection_dir.mkdir(exist_ok=True)

        outline_dir = figure_dir / "outline"
        outline_dir.mkdir(exist_ok=True)

        # Show the detected granules in the image
        if plot_frame:
            cmap = plt.get_cmap("tab20")
            cmap.set_bad((0, 0, 0, 0))
            fig, axs = pt.create_axes(2)

            masked_granules = np.ma.masked_equal(detector.labelled_granules, 0)
            # masked_granules = detector.labelled_granules
            axs[0].imshow(masked_granules, cmap=cmap, interpolation="none")
            axs[1].imshow(frame.im_data, cmap="inferno", )

            axis_len = max(frame.im_data.shape)
            tick_spacing = 256 if axis_len > 2000 else 128

            titles = ["Detected", "Original"]

            for ax, title in zip(axs, titles):
                ax.set_title(title)
                ticks = np.arange(0, axis_len+1, step=tick_spacing)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

            plot_save_name = (
                detection_dir
                / f"{input_image.stem}--F{frame_num:03d}.png"
            )
            pt.save_figure_and_trim(plot_save_name, dpi=330)

        # Get the approximate boundary for each granule
        # skip frame if there are no granules
        boundary_method = config("image_processing", "method")
        try:
            granule_boundries = [
                be.BoundaryExtraction(granule,boundary_method) for granule in detector.granules()
            ]     
        except gl.GranuleNotFoundError:
            continue

        # Tidy these Fourier terms per frame
        # This is an iterative function that reuses results from the previous frames.
        try:
            aggregate_terms = be.collect_fourier_terms(
                granule_boundries, frame, granule_tracker, plot_granules, output_dir
            )
            fourier_frames.append(aggregate_terms)
        except gl.GranuleNotFoundError:
            continue

        if max_frame is not None and frame_num >= max_frame:
            process_bar.close()
            break

    # Merge all of the frame data and save
    fourier_frames_pd = pd.concat(fourier_frames, ignore_index=True)
    # fourier_table = consolidate_fourier_terms(fourier_frames_pd)

    # Save a .csv file for debugging
    save_name = f"fourier/{input_image.stem}"
    if max_frame is not None:
        save_name += "--DEBUG"
    save_path = output_dir / (save_name + ".h5")

    # Save a HDF5 file for better long-term storage with metadata
    frame_data = {
        "num_frames": frame.total_frames,
        "input_path": str(input_image.resolve()),
        "pixel_size": frame.pixel_size,
    }
    hdf_save_path = save_path.with_suffix(".h5")
    print(f"\n#{_pbar_pos+1} Fourier file save location: {hdf_save_path}\n")
    write_hdf(hdf_save_path, fourier_frames_pd, frame_data)


def consolidate_fourier_terms(fourier_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge the Fourier terms into a pivot table.

    In theory this is a much more compact way of representing the data, but for some
    reason it is poorly behaved with pivot tables.

    This would also require some conversion step to melt it back into the table.
    """
    fourier_table = fourier_frame.pivot_table(
        values="magnitude", index=["frame", "granule_id"], columns="order"
    )

    trimmed_data = fourier_frame.query("order == 2").drop(columns="magnitude")

    return fourier_table.merge(
        trimmed_data, on=["frame", "granule_id"], validate="one_to_one"
    )


def validate_args(input_image: Path, output_dir: Path, quiet: bool = False):
    """ Ensure that the provided parameters are sane. """
    if not input_image.exists():
        raise FileNotFoundError(f"Provided image does not exist: {input_image}")
    if not output_dir.is_dir():
        raise IOError(f"Provided output_dir is not directory: {input_image}")


def write_hdf(save_path: Path, fourier_frames: pd.DataFrame, frame_data):
    """ Write the data out as hdf5 files.

    This is more stable and portable than the previous pickle method. It also allows
    storage of metadata in a more sane manner.
    """
    if platform.system()=="Darwin" and "ARM64" in platform.version():
        # Doing it this way will ensure we still catch Apple Silicon Macs even when using Rosetta 2.
        # The 'else' case below should catch all other platforms where writing to hdf5 should work normally.
        try:
            fourier_frames.to_hdf(save_path, key="fourier", mode="w", complib="bzip2")

            # Add attributes to the frames
            with h5py.File(save_path, "a") as f:
                fourier_hdf = f["fourier"]
                # Add the user defined keys
                for key, val in frame_data.items():
                    fourier_hdf.attrs[key] = val

                # Add configuration values
                config_yaml, _ = config._aggregate_all()
                fourier_hdf.attrs["config"] = config_yaml
                fourier_hdf.attrs["version"] = version.__version__
        except Exception:
            config_yaml, config_summary = config._aggregate_all()
            file = open(f'{str(save_path)[:-3]}.pkl', 'wb')
            pkl.dump({'fourier': fourier_frames, "frame_data": frame_data, "configuration": config_yaml, "version": version.__version__}, file=file)
    else:
        fourier_frames.to_hdf(save_path, key="fourier", mode="w", complib="bzip2")

        # Add attributes to the frames
        with h5py.File(save_path, "a") as f:
            fourier_hdf = f["fourier"]
            # Add the user defined keys
            for key, val in frame_data.items():
                fourier_hdf.attrs[key] = val

            # Add configuration values
            config_yaml, _ = config._aggregate_all()
            fourier_hdf.attrs["config"] = config_yaml
            fourier_hdf.attrs["version"] = version.__version__

def _get_blurrer(frame) -> glf.DeltaBlurrer:
    """Precomputes the kernels required for the FFT deconvolution."""
    plane_shape = frame.im_data.shape

    min_size = float(config("image_processing", "granule_minimum_radius"))
    max_size = float(config("image_processing", "granule_maximum_radius"))

    pixel_size = frame.pixel_size
    min_sigma = gl._convertToSigma(min_size, pixel_size)
    max_sigma = gl._convertToSigma(max_size, pixel_size)
    sigmas = glf.generate_sigmas(min_sigma, max_sigma)

    # In theory, there is some performance penalty for running at size (k * 2**N), where k is
    # a small prime, compared to a pure power of two, but the increase in required size means
    # that the small prime case has been faster in all cases I've tested. 
    allow_power_of_three = True
    allow_power_of_five = True

    # Keeping the FFT same size as the image can give siginificant speedups, but at the cost of
    # significant errors around the edges. ENABLE AT YOUR OWN RISK.
    trim_fft = False
    if trim_fft:
        fft_len = max(plane_shape)
    else:
        fft_len = None

    blurrer = glf.DeltaBlurrer(
        sigmas=sigmas,
        plane_shape = plane_shape,
        fft_len = fft_len,
        _allow_power_of_five=allow_power_of_five,
        _allow_power_of_three=allow_power_of_three,
    )
    return blurrer


if __name__ == "__main__":
    args = parse_arguments()

    main(args.input, args.output, args.quiet, args.max_frame, args.cores)
