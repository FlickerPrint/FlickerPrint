#!/usr/bin/env python

""" Read metadata and images from microscopy files.

Frame Generator
===============

This module provides the ``gen_opener`` generator, which given a path a microscope file
returns the frames in the image one-by-one wrapped in a ``MicroscopeFrame`` object. This
contains the pixel information in the frame and the metadata of the frame such as the
corresponding pixel size in the image.

Javabridge
----------

Opening microscopy files is surprisingly non-trivial, requiring the use of bftools [link]
and python-javabridge. This introduces a great deal of complications into the software:

- The javabridge can only be started once and care must be taken to shut it down in the
  case of exception, otherwise the process will hang on error.
- Javabridge is not compatible with multiprocessing - hence why we have deal with only
  one file per run when getting the Fourier terms.
- Java garbage collection spawns many threads in a high core system, such as a server and
  consumes arbitrary resources unless the environment variable ``JAVA_THREADS = 0``,
  which is set in the ``multicore_saftey.py`` module.
- Java logging interacts poorly with the expected python output on stdout

To deal with this we provide a ``@vmManager`` wrapper, this ensures that the javaVM is
shutdown properly even if the function exits with an error (``finally`` clause) and
suppresses java logging messages below ERROR. However, we are still unable to open the
javabridge a second time within the same python process!

Reading metadata
----------------

While there is a open standard for microscopy data metadata and format, these are not
strictly adhered to and so we have to handle these cases explicitly

Microscope files are surprisingly in-homogeneous, we currently support the following file
types:

- DONE: Andor: .ims, .lif, .ome.tif
- ON-GOING: Zeiss: .czi

Outline
-------

Returns a generator for the frames of the image, this is designed to be called once per
files. This allows better organization by tools such as ``snakemake``. While this limits
the parallelism somewhat, ``bioformats`` and ``javabridge`` handles parallelism badly.

If there are multiple Z-planes then we return these in a 3d array, then we iterate
through the time steps.

Objects
-------

FrameGen
    Generator for frames, contains metadata that is common to each file type.

MicroscopeFrame
    Container for the frame image and metadata unique to that frame.


"""

import datetime
import enum
import xml.etree.ElementTree as ET
from typing import Optional
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
import warnings
import logging

import bioformats as bf
import javabridge
import numpy as np

from flickerprint.common.configuration import config

JAVAVM_STARTED = False


@dataclass
class MicroscopeFrame:
    """Store a single image frame, along with metadata, from the microscope.

    Included metadata:

      - ``frame_num``: The index of this frame in the time series
      - ``total_frames``: The total number of frames in the time series
      - ``timestamp``: [Optional] The image was taken according to the microscope
      - ``pixel_size``: [Optional] The size of the image pixel on the sample, typically measured in μm/pixel
      - ``actual_pixel_size``: [bool] Is the ``pixel-size`` included in the metadata
      - ``actual_timestamp``: [bool] Is the ``timestamp`` recorded from the image

    """

    im_data: np.ndarray
    im_path: Path
    frame_num: int
    total_frames: int
    timestamp: int = 0
    pixel_size: float = 1.0
    actual_pixel_size: bool = True
    actual_timestamp: bool = False
    enhanced_image: Optional[np.ndarray] = field(init=False, default=None)

    @property
    def summaryRow(self):
        """ Create a summary row of the data as a dictionary. """
        return dict(
            frame_num=self.frame_num,
            im_path=str(self.im_path),
            total_frames=self.total_frames,
            pixel_size=self.pixel_size,
            actual_pixel_size=self.actual_pixel_size,
        )


class GeneratorTypes(enum.Enum):
    """A list of the possible generator types required.

    We use an enumeration as this ensures that the type can be checked unambiguously and
    much faster than with simple string comparisons.

    For instance, ``image == "bioformats"`` is more prone to awkward to address bugs if
    either the variable or test condition is spelt wrong.

    """

    BIOFORMATS = 1
    TIFF = 2
    PNG = 3


def gen_opener(im_path):
    """Return a generator based on the provided ``image_path``.

    For now we simply search for the correct extension.
    A generator for the ``MicroscopeFrames`` s and common metadata.
    """
    im_path = Path(im_path)
    image_type = _getType(im_path)
    if image_type == GeneratorTypes.BIOFORMATS:
        return bioformatsGen(im_path)
    else:
        raise NotImplementedError("Currently not handling not-bioformats files.")


def _getType(im_path: Path):
    """ Return the required generator type based on the image extension. """
    extentions = im_path.suffixes
    if extentions[-2:] == ['.ome','.tif'] or extentions[-2:] == ['.ome','.tiff']:
        return GeneratorTypes.BIOFORMATS
    elif extentions[-1] in [".ims",".lif", ".nd2"]:
        return GeneratorTypes.BIOFORMATS
    elif extentions[-1] == "png":
        return GeneratorTypes.PNG
    elif extentions[-1] in [".tiff", ".tif"]:
        return GeneratorTypes.BIOFORMATS
    else:
        logging.info(f"Unrecognised extension {extentions}, trying bioformats")
        return GeneratorTypes.BIOFORMATS

def bioformatsGen(im_path):
    """ Load an image from a bioformats file. """
    # Get some metadata from the OMEXML data
    md = bf.get_omexml_metadata(str(im_path))
    o = bf.OMEXML(md)
    # Extract the relevant terms
    pixelData = o.image().Pixels
    pixel_size = pixelData.PhysicalSizeX #in microns
    if pixel_size == None:
        pixel_size = float(config("image_processing", "pixel_size"))
        warnings.warn(f"Warning: pixel size missing from metadata, falling back on value in config file: {pixel_size} microns.")
        pixel_units = "µm"
    else:
        pixel_sizey = pixelData.PhysicalSizeY
        if abs(pixel_size - pixel_sizey) > 1e-3:
            raise ValueError("Error: pixels must be square")

        pixel_units = pixelData.PhysicalSizeXUnit

        if pixel_units != "µm":
            warnings.warn(f"Warning: pixel size not in microns: attempting to convert.")
            if pixel_units == "m":
                pixel_size *= 1e6
            elif pixel_units == "cm":
                pixel_size *= 1e4
            elif pixel_units == "mm":
                pixel_size *= 1e4
            elif pixel_units == "nm":
                pixel_size *= 1e-3
            elif pixel_units == "pm":
                pixel_size *= 1e-6
            else:
                raise ValueError(f"Error: Pixel unit conversion failed. Units {pixel_units} not recognised")        

    n_slices = pixelData.SizeZ
    if n_slices == None:
        n_slices = 1
        warnings.warn("Warning: number of z slices missing from metadata, assuming 1")
    n_frames = pixelData.SizeT
    if n_frames == None:
        raise ValueError("Number of frames missing from microscope file metadata")
    # Get the image microscope format
    im_extension =im_path.suffixes
    if n_slices > 1:
        raise ValueError("Currently we don't support 3D images. If the microscope image is supposed to be a time series, please use the hyperstacks feature in Fiji to convert it.")

    if im_extension[-1] == ".ims":
        time_stamps = _getIMStimeStamps(n_frames, md)
        actual_timestamp = True
    elif im_extension[-1] == ".lif":
        time_stamps = _getLIFtimeStamps(n_frames, o)
        actual_timestamp = True
    else:
        time_stamps = np.zeros(n_frames)
        actual_timestamp = False
        warnings.warn("Warning: Cannot read timestamps from this file-type. Setting times to 0")

    if not JAVAVM_STARTED:
        startVM()
    try:
        with bf.ImageReader(str(im_path)) as reader:
            # For frame in frame_nums
            for frame_num in range(n_frames):
                frame_data = reader.read(t=frame_num, z=0, c=0, rescale=False)
                yield MicroscopeFrame(
                    im_data=frame_data,
                    im_path=im_path,
                    frame_num=frame_num,
                    total_frames=n_frames,
                    timestamp=time_stamps[frame_num],
                    pixel_size=pixel_size,
                    actual_pixel_size=True,
                    actual_timestamp=actual_timestamp,
                )
    except AttributeError:
        closeVM()

def _getIMStimeStamps(n_frames, md) -> np.ndarray:
    """ Return an array with the timestamps for each frame. """
    time_stamps = np.zeros(n_frames, dtype="datetime64[ms]")
    structuredData = _getStucturedData(md)
    for child in structuredData:
        entry = child[0][0]
        key = entry[0].text
        value = entry[1].text
        if key.startswith("TimePoint"):
            # Remove the TimePoint string, and account for the 1-indexing
            frameNum = int(key[9:]) - 1
            # time_stamps[frameNum] = value
            time_stamps[frameNum] = datetime.datetime.strptime(
                value, "%Y-%m-%d %H:%M:%S.%f"
            )

    return time_stamps

def _getLIFtimeStamps(n_frames, o) -> np.ndarray:
    """ Return an array with the timestamps for each frame. !jl"""
    time_stamps = np.zeros(n_frames, dtype="datetime64[ms]")
    date = o.image().get_AcquisitionDate().replace('T',' ')
    startTime = datetime.datetime.strptime(date,"%Y-%m-%d %H:%M:%S")
    for i in range(n_frames):
        t = datetime.timedelta(seconds=o.image().Pixels.Plane(i).DeltaT)
        time_stamps[i] =  startTime + t 
        
    return time_stamps


def _getStucturedData(md):
    """Return the structured annotation from the markdown.

    This may move in the xml, so we have to search for this directly. If it is not found,
    then return None.

    """
    root = ET.fromstring(md)
    for child in root:
        if child.tag.endswith("StructuredAnnotations"):
            return child

    return None


def _indent(elem, level=0):

    """ Format an xml string into human readable form.

    """

    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            _indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem  

def vmManager(bioformatsFunc):
    """ Decorator to ensure the sane shutdown of the vm. """

    @wraps(bioformatsFunc)
    def wrapper(*args, **kwargs):
        if not JAVAVM_STARTED:
            startVM()
        try:
            funcOut = bioformatsFunc(*args, **kwargs)
        finally:
            closeVM()
        return funcOut

    return wrapper


def startVM():
    """ Start the java vm for bioformats. """
    global JAVAVM_STARTED
    JAVAVM_STARTED = True
    javabridge.start_vm(
        class_path=bf.JARS,
        max_heap_size="8G",
        run_headless=True,
    )
    try:
        stfuLogging()
    except javabridge.JavaException:
        pass
    return

def closeVM():
    """ Close the javabridge. """
    global JAVAVM_STARTED
    if JAVAVM_STARTED:
        javabridge.kill_vm()
        JAVAVM_STARTED = False


def stfuLogging(level="ERROR"):
    """ Suppress Logging nonsense """
    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level",
                                   "WARN",
                                   "Lch/qos/logback/classic/Level;")

    javabridge.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)
    return


if __name__ == "__main__":
    startVM()

    @vmManager
    def main():
        current_file = Path(__file__).resolve()
        project_dir = current_file.parents[1] / "tests/data"
        image_path = project_dir / "lif_test.lif"

        image_gen = bioformatsGen(image_path)
        next(image_gen)
        print("Works so far...")

    main()
