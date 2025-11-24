#!/usr/bin/env python

""" Locate the granules within the ``MicroscopeFrame``.

Outline
=======

We provide two methods to locate the granule within the Frame.

Methods
-------

DoG
    Difference of Gaussian - this

Objects
-------

Granule
    An image of the granule and the surrounding area, along with the location and other
    metadata, including a rough estimate of the granule boundary.

"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import skimage as ski
from skimage import feature
from skimage import filters
from scipy import ndimage
from skimage.feature import blob_dog, blob_log
from skimage import segmentation

import flickerprint.tools.plot_tools as pt
from flickerprint.common.configuration import config
from flickerprint.common.frame_gen import MicroscopeFrame


class GranuleNotFoundError(Exception):
    """Raise this Exception if no granules are found.

    This can be useful to explicitly catch awkward edge cases where empty arrays of
    granules will cause troubles.
    """

    pass


class Granule:
    """Container for a crop of the the microscope image containing the granule.

    Extends the crop from the granule detection by ``padding`` pixels to help with
    boundary drawing. This retains the metadata from the microscope frame.
    """

    def __init__(self, im_data, property_row, padding=5):
        """ Get properties from the granule detector. """
        self._im_height, self._im_width = im_data.shape

        if hasattr(property_row, "_asdict"):
            self.properties = property_row._asdict()
        else:
            self.properties = property_row

        self.im_cropped = self._slice_image(im_data, padding=padding)
        self.im_raw = self.im_cropped.copy()
        self.padding = padding

        self.smoothing_width = float(config("image_processing", "smoothing"))
        self.im_smoothed = ski.filters.gaussian(self.im_cropped, self.smoothing_width)
        self.crop_width, self.crop_height = self.im_cropped.shape

        # Centre of the granule relative to the entire image
        self.image_centre = [self.properties[f"weighted_centroid_{i}"] for i in (0, 1)]
        bbox = [self.properties[f"bbox_{i}"] for i in range(4)]
        self.bbox = self._extend_bbox(*bbox, padding)

        # Weighted centre for the cropped image
        M = ski.measure.moments(self.im_cropped)
        self.local_centre = np.array((M[0, 1] / M[0, 0], M[1, 0] / M[0, 0]),)

    def _slice_image(self, image, padding=5):
        """Slice to include an image around the selected granule.

        While there is a similar function already provided by the ``regionprops_table``
        we wish to include a larger region around the granule.
        """
        bbox = [self.properties[f"bbox_{i}"] for i in range(4)]
        bbox_extended = self._extend_bbox(*bbox, padding)
        im_data = image[
            bbox_extended[0] : bbox_extended[2], bbox_extended[1] : bbox_extended[3],
        ]
        return im_data

    def _extend_bbox(self, x_min, y_min, x_max, y_max, padding):
        """Extend the bounding box by ``padding`` pixels in each dimension.

        We have to be careful to account for the boundaries of the image.
        """
        # Test the arguments are sane
        if x_min >= x_max:
            raise ValueError(f"x_min >= x_max -- {x_min} >= {x_max} for bbox extension")
        if y_min >= y_max:
            raise ValueError(f"y_min >= y_max -- {y_min} >= {y_max} for bbox extension")
        if padding < 0:
            raise ValueError(f"Padding must be positive, currently {padding}")

        # Account for the boundary cases
        x_new_min = max(x_min - padding, 0)
        x_new_max = min(x_max + padding, self._im_width - 1)
        y_new_min = max(y_min - padding, 0)
        y_new_max = min(y_max + padding, self._im_height - 1)

        return x_new_min, y_new_min, x_new_max, y_new_max


@dataclass
class GranuleDetector:
    """Detector for the granules in the image.

    Attributes
    ----------

    mask: np.ndarray, dtype=bool

        A binary image, with pixels belonging to a granules 1, and background 0.

    labelled_granules: np.ndarray, dtype=int

        An image with each the extent of granule given as an unique integer, 0
        corresponds to the background.

    Parameters
    ----------

    Frame: MicroscopeFrame
        Current MicroscopeFrame
    min_size: float = 0.3
        Smallest size of granule to detect. If a pixel size is provided then this will
        be assumed to be in μm, otherwise this will be in pixels.
    max_size: float = 3.0
        Largest granule to detect. See above.
    processed_image: the processed version of the image to be used for granule detection

    Methods
    -------

    granules:
        Iterator of ``Granule`` objects

    labelGranules:
        Create the ``labelled_granules`` array from the starting image.
        This should perform all steps required to generate ``Granules``.

    plot:
        Show the detection of granules within the frame.

    """

    frame: MicroscopeFrame

    min_size: float = field(init=False)

    max_size: float = field(init=False)

    def __post_init__(self):
        self.min_size = float(config("image_processing", "granule_minimum_radius"))
        self.max_size = float(config("image_processing", "granule_maximum_radius"))

    def granules(self, padding=5) -> Iterator[Granule]:
        """ Iterator for the granules detected by in this frame. """
        if not hasattr(self, "labelled_granules"):
            self.labelGranules()

        granule_table = self._getTable()

        for row in granule_table.itertuples():
            yield Granule(self.frame.im_data, row, padding=padding)

    def labelGranules(self):
        """Label the granules within the images.

        This creates an integer array with each granules labeled with a difference
        integer.
        """

        threshold = float(config("image_processing", "granule_minimum_intensity"))
        method = config("image_processing","method")

        if (method == "gradient"):
            if self.frame.enhanced_image is not None:
                self.processed_image = self.frame.enhanced_image
            else:
                self.processed_image = self.frame.im_data

        elif (method == "intensity"):
            self.processed_image = _process_vesicles(self.frame.im_data)
        else:
            raise ValueError("no granule detection method {}".format(method))


        self.granule_locations = _detect_granules_dog( 
                self.processed_image,
                self.min_size,
                self.max_size,
                self.frame.pixel_size,
                threshold=threshold,
            ) 

        self.labelled_granules = self._fillGranules()

    def plot(self, ax=None, save_path: Path = None, cmap='viridis'):
        """Show the labelled granules within the image.

        If no `ax` is provided then create a new matplotlib axis and save to ``save_path``.
        """
        axes_created = ax is None
        if ax is None:
            fig, ax = pt.create_axes(1)

        image = np.where(self.labelled_granules != 0, 1, 0)
        ax.imshow(image, cmap=cmap)
        if axes_created:
            pt.hide_axis_lables(ax)
            pt.save(save_path, padding=0, tl_padding=0)

    def _getTable(self) -> pd.DataFrame:
        """Return a list of the properties of the granules."""
        if not hasattr(self, "labelled_granules"):
            raise AttributeError("`labelGranulues` must be called before `_getTable`")

        # We separate the properties by an underscore, this is more robust as a variable
        # name.
        properties = [
            "label",
            "area",
            "perimeter",
            "bbox",
            "mean_intensity",
            "weighted_centroid",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
        ]

        regionprops = ski.measure.regionprops_table(
            self.labelled_granules,
            self.frame.im_data,
            properties=properties,
            separator="_",
        )

        # Define a score to sort the granules by
        table = pd.DataFrame(regionprops)
        table["score"] = table["area"] * table["mean_intensity"]
        #TODO this should not be necessary !jl
        table.sort_values(by="score", ascending=False, inplace=True)
        return table

    def _scoreGranules(self, row: pd.DataFrame):
        """ Score the granule on a given criteria. """
        print(row)
        return row["area"]

    def _fillGranules(self) -> np.ndarray:
        """Return a labelled image corresponding to individual granules.

        These are split using the flood fill method.
        """
        # Generate a masked binary image for each of the granules, removing those
        # where we fail to find a granule
        masks = [self._fillGranule(b) for b in self.granule_locations]
        masks = list(filter(lambda m: m is not None, masks))
        masks = np.array(masks, dtype=np.int16)

        if len(masks) == 0:
            raise GranuleNotFoundError

        # OR all of the masks into one image, this ignores overlapping regions
        self.mask = masks.any(axis=0)
        labelledImage = ski.measure.label(self.mask)
        labelledImage = segmentation.clear_border(labelledImage)
        return labelledImage

    def _fillGranule(self, blob):
        """Attempt to flood fill a given granule.

        We test for the size of the granule to ensure that the flood fill does not cover
        too large of a region. This typically the case when a granule is not very
        distinct compared to the background and so the flood fill covers a large area of
        background as well.

        Outline
        -------

        Decrease the threshold limit from the maximum magnitude of the disk, if we don't
        find a granule within below a given size then we return None.

        Parameters
        ----------

        max_area:
        largest granule size that is accepted, we perform another iterative step
        if it's too large.

        """
        min_intensity_lim = float(config("image_processing", "granule_minimum_intensity"))

        threshold = float(config("image_processing", "fill_threshold"))

        method = config("image_processing","method")

        x, y, r = blob
        x = int(x)
        y = int(y)

        # Move x, y to the brightest spot in the surrounding region
        prev_intensity = self.processed_image[x, y]
        x, y = self._refineCentre(x, y, radius=5)
        new_intensity = self.processed_image[x, y]

        if new_intensity < prev_intensity:
            print(f"{prev_intensity:6d} {new_intensity:6d}", end=" ")
            print("Error!")

        # Filter on the magnitude of the central point
        center_intensity = self.processed_image[x, y]
        # TODO: Scale this by the maximum intensity of the image
        max_intensity = self.processed_image.max()
        if center_intensity < max_intensity * min_intensity_lim:
            return None

        if (method == "gradient"):
            # The tolerance is the difference from centre point in abs. intensity
            tolerance = center_intensity * (1.0 - threshold)
        elif (method == "intensity"):
            # mask is binary so no tolerance needed
            tolerance = None
        else:
            raise ValueError("no granule detection method {}".format(method))

        # First test
        mask = ski.morphology.flood(
            self.processed_image, (x, y), tolerance=tolerance, connectivity=1
        )

        area = mask.sum()

        # Filter out granules with areas which fall outside of the threshold range.
        pixel_size = self.frame.pixel_size
        if pixel_size == None:
            pixel_size = 1
        if area > int(np.pi * self.max_size ** 2 / pixel_size ** 2):
            return None
        if area < int(np.pi * self.min_size ** 2 / pixel_size ** 2):
            return None

        return mask

    def _refineCentre(self, x, y, radius=2):
        """Find the brightest pixel within a small area of the point.

        DoG (and LoG) do not always return the brightest spot in the blob, this can
        cause trouble with the thresholding, as the more intense areas may also be
        excluded,

        We search a grid that with edge size 2*``radius`` + 1.
        """
        width, height = self.processed_image.shape

        # Account for points chosen near the boundary of the image
        xMin = max(0, x - radius)
        xMax = min(x + radius, width - 1)
        yMin = max(0, y - radius)
        yMax = min(y + radius, height - 1)

        # Get the pixel with the highest itensity
        intensity_grid = self.processed_image[xMin : xMax + 1, yMin : yMax + 1]
        max_grid_point = np.unravel_index(intensity_grid.argmax(), intensity_grid.shape)

        # Move the point from grid coordinates to image coordinates
        x_new, y_new = max_grid_point
        x_new = xMin + x_new
        y_new = yMin + y_new

        if False:
            print(f"old {x:4d}, {y:4d}")
            print(f"new {x_new:4d}, {y_new:4d}")
            print()
        return x_new, y_new


def _detect_granules_dog(
    image, min_size, max_size, pixel_size=None, threshold=0.1, overlap=0
):
    """Blob detection based on the Difference of Gaussian (DoG) method.

    This is an approximation of the LoG method that scales much better for larger
    granules, but is less able to detect smaller blobs. It retains the LoG's
    resistance to noise.

    A maximum and minimum size of granule needs be provided. If pixel_size if
    provided, then this is used to relate the size of the pixel to the physical size
    of the granule.

    Parameters
    ----------

    image : np.ndarray
        A 2D or 3D array containing the image
    min_size : float
        The minimum expected size of the granules to search for
    max_size : float
        The maximum expected size of the granules to search for
    pixel_size: float


    Returns
    -------

    blobs
        [(x, y, r), ...] : return the center points of the granules.

    """
    min_sigma = _convertToSigma(min_size, pixel_size)
    max_sigma = _convertToSigma(max_size, pixel_size)

    return blob_dog(
        image, min_sigma, max_sigma, threshold=None, threshold_rel=threshold, overlap=overlap,
    )

def _process_vesicles(image):
    edges = feature.canny(image/255.0,sigma=1,low_threshold=0.001)
    blur = filters.gaussian(edges, 0.5)
    return ndimage.binary_fill_holes(blur)

def thresholdDetector():
    """ Otsu based detection. """


def _convertToSigma(radius_physical, pixel_size=None):
    """Convert the estimated physical radius into a sigma value.

    in the LoG and DoG methods the radius is approximately r = sqrt(2) σ.

    Parameters
    ----------

    radius_physical:
        size of the blob to search for, typically in μm
    pixel_size:
        size of the pixel in real space, typically μm / pixel

    While the units are given in μm, this is not required; as long as the units are
    consistent between the two variables. Therefore, if a ``pixel_size`` is not given,
    then we can take this to be 1.
    """

    # In LoG and DoG the radius is roughly, r = sqrt(2)σ.
    scale = 1.0 / np.sqrt(2)

    # Default to a single pixel
    if pixel_size is None:
        pixel_size = 1

    # Size in pixels of the search radius
    radius_pixels = radius_physical / pixel_size
    sigma = scale * radius_pixels

    return sigma


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    TEST_DIR = Path(__file__).resolve().parents[1]
    DATA_CACHE_DIR = TEST_DIR / "tests/data/"

    pickle_path = DATA_CACHE_DIR / "ims_test_frame.pkl"

    with open(pickle_path, "rb") as f:
        frame = pickle.load(f)

    detector = GranuleDetector(frame)
    detector.labelGranules()

    for num, granule in enumerate(detector.granules(padding=10)):
        if num >= 10:
            break
        save_path = Path(f"/tmp/granule-{num:02d}.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(granule, f)

        fig, axs = pt.create_axes(2, axes_height=4, sharey=True)
        imshow_kwargs = dict(origin="lower", cmap="inferno")
        axs[0].imshow(granule.im_cropped, **imshow_kwargs, vmax=65536)
        axs[1].imshow(granule.im_cropped, **imshow_kwargs)

        plt.tight_layout()
        pt.save(save_path.with_suffix(".png"))
