#!/usr/bin/env python

""" Extract the Fourier terms from a ``Granule``.

Boundary Extraction
===================

- Draw the boundary around the granule
- Fourier transform the boundary to get the perturbation
- Group the same granule across frames

Outline
-------

Given a ``Granule`` object we use a directional gradient to calculate the boundary of
the granule. From this boundary we perform a Fourier transform to extract amplitude of
each of these modes.

This analysis is performed on a granule by granule basis, we provide two functions to
group the granules and aggregate the results, into a singe frame then across the frames.

When the granules are grouped across the frames we group the granules by their position
in the frame, tracking the granule as it moves.

See ``workflow.process_image.py`` for examples of how to use this module.

Provides
--------

BoundaryExtractionGradient(granule:Granule):
    A class for measuring the boundary of the granules.


"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist

import flickerprint.tools.plot_tools as pt
from flickerprint.common import kernels
from flickerprint.common.configuration import config
from flickerprint.common.frame_gen import MicroscopeFrame
from flickerprint.common.granule_locator import Granule
from flickerprint.common.granule_locator import GranuleNotFoundError

class BoundaryExtraction:
    """ Base for the boundary extraction. """

    def __init__(self, granule: Granule, boundary_method):
        self.granule = granule

        self.angles = None
        self.radii = None
        self.processed_image = None

        if boundary_method == "gradient":
            self.imageProcessor = _BoundaryExtractionGradient(granule)
        elif boundary_method == "intensity":
            self.imageProcessor = _BoundaryExtractionMagnitude(granule)
        else:
            raise Exception("invalid boundary method, check config file")

    def validate_boundary(self):
        """ Look for discontinuities in the boundary """

        # This is a bit crude but should work
        diffs = abs(np.diff(self.radii))
        return not any(diffs > 0.5)

    @property
    def mean_radius_pixels(self):
        """ Return the mean radius of the granule in pixel units.

        This is then meant to be scaled by the physical size of the pixels in the
        microscope image. This is left to the user however to avoid a large pass through
        of variables.
        """
        if self.radii is None:
            raise AttributeError("Run angle_sweep before accessing the mean radius.")
        return self.radii.mean()

    @staticmethod
    def get_fourier_terms(radii):
        """ Perform the Fourier analysis.

        Returns
        -------

        components: np.ndarray
            The (complex) magnitudes of the boundary in Fourier space

        fft_freq:
            The mode number of each of the Fourier terms

        centre: The first fourier mode, which is required for recreating the droplet outline

        """
        if radii is None:
            raise AttributeError("Run angle sweep before ``get_fourier_terms``")

        n_theta = len(radii)

        # The perturbation is given relative to the granule radius
        perturbation = radii / radii.mean()
        # Remove the constant component
        perturbation -= 1

        # Scale the FFT results so that the implementation matches the definition used
        # in the paper.
        # TODO IS THIS IT !!!!!!!! 
        components = np.fft.rfft(perturbation)
        components *= 1.0 / n_theta

        fourier_limit = 60
        fft_freq = np.fft.rfftfreq(n_theta, 1.0 / n_theta)
        return components[2:fourier_limit], fft_freq[2:fourier_limit], components[1]

    def plot(self, ax=None, save_name: Path = None, im=None, dpi=None):
        """ Plot the detected edge of the granule. """
        create_plot = ax is None
        if create_plot:
            fig, ax = pt.create_axes(1, fig_width=3)

        if im is None:
            im = self.granule.im_smoothed
        ax.imshow(im, cmap="inferno")
        x_centre, y_centre = self.granule.local_centre
        ax.scatter(x_centre, y_centre)

        if self.angles is not None:
            valid = self.validate_boundary()
            colour = "white" if valid else "red"
            x, y = pt.polar2cart(self.angles, self.radii)

            y += self.granule.local_centre[0]
            x += self.granule.local_centre[1]
            ax.scatter(y, x, s=3, c=colour)

        if create_plot and save_name is not None:
            pt.save_figure_and_trim(save_name, dpi=dpi)

    def angle_sweep(self, n_angles, samples_per_pixel=5, order=3):
        """Measure the border of the image.

        This works by sampling the image at ``n_angles`` evenly spaced and calculating
        the radius that best corresponds to the boundary. This evenly spaced nature
        allows us to Fourier transform the results.

        Parameters
        ----------

        n_angles: int
             We sample N angles evenly spaced in the range [0, 2π)

        samples_per_pixel: float
             How many points are sampled per pixel during interpolation. Typically > 1.

        Attributes
        ----------

        n_angles: np.ndarray
            The sampling angles for the boundary
        radii: np.ndarray
            The best estimate for the boundary values.

        """
        #Create a processed image whose maximum intensity corresponds to the boundary
        if self.processed_image is None:
            self.processed_image = self.imageProcessor.process_image()

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii = np.zeros_like(angles)

        precalculate_spline = True
        if precalculate_spline:
            spline_image = ndi.spline_filter(input=self.processed_image, order=order)
            image_to_process = spline_image
        else:
            image_to_process = self.processed_image

        for num, angle in enumerate(angles):
            sample_length = self.granule.crop_width
            sample_count = int(sample_length) * samples_per_pixel

            sample = self._sample_at_angle(
                angle=angle,
                sample_length=sample_length,
                sample_count=sample_count,
                im=image_to_process,
                order=order,
                is_spline_image=precalculate_spline
            )
            peak_location = self._get_peak_location(sample)
            radii[num] = peak_location

        # Return the position in terms of the original length
        radii /= samples_per_pixel

        self.angles = angles
        self.radii = radii
        return angles, radii

    def _get_interploation_coordinates(self, angle, sample_length, sample_count):
        """ Helper for interpolation functions.

        Converts the given direction vector into a list of coordinates to sample at.

        Parameters
        ----------

        sample_length: float
            Length of the sample strip, given in pixel units
        sample_count: int
            Number of pixels to sample in the image
        """
        # Start the sample line at the centre
        y0, x0 = self.granule.local_centre

        # Finish at the given point
        x1 = x0 + sample_length * np.cos(angle)
        y1 = y0 + sample_length * np.sin(angle)

        # Create the sample line
        x = np.linspace(x0, x1, sample_count)
        y = np.linspace(y0, y1, sample_count)

        interpolation_coords = np.vstack((x, y))
        return interpolation_coords

    def _sample_at_angle(self, angle, sample_length, sample_count, im, order=3, is_spline_image: bool = False):
        """ Sample the image at a given number of points.

        Parameters
        ----------
        angle: float
            Angle of the sample line, relative to the x-axis
        sample_length: float
            Length of the sample in pixels in the original image.
        sample_count: int
            Total number of samples in the returned value.
        im:
            The image to sample
        order:
            Polynomial order of the estimation.
        is_spline_image: bool
            Has the image already been passed through ``spline_filter`` this should allow us to
            avoid us repeating some work
            

        Returns
        -------
        array
            Interpolated values.

        """
        interpolationCoords = self._get_interploation_coordinates(
            angle=angle, sample_length=sample_length, sample_count=sample_count
        )

        # The ``prefilter`` keyword is a bit counter-intuitive, it is True when the
        # values need to be calculated, ie we don't use the cache.
        zi = ndi.map_coordinates(im, interpolationCoords, order=order, prefilter=not is_spline_image)
        return zi

    @staticmethod
    def _get_peak_location(sample):
        """ Get the peak in the given sample.

        We return the maximum of the given sample, however this might be extended to
        include more complex methods.
        """
        return np.argmax(sample)


class _BoundaryExtractionMethod:
    """ interface class for boundary detection methods
    Input Parameters
    ----------

    granule: Granule
        Extract the boundary from this granule.


    Methods
    -------

    process_image
        return an processed "image" with maximum intesity at the granule boundary  
    """
    def process_image(self):
        raise Exception("this is an interface class and should not be called")

    def __init__(self,granule: Granule):
        self.granule = granule


class _BoundaryExtractionGradient(_BoundaryExtractionMethod):
    """ Extract the boundary of granule using a directional gradient.
        this is for granule which appear as a solid blob in the microscope.

    Input parameters
    ----------

    granule: Granule
        Extract the boundary from this granule.


    Methods
    -------

    process_image
        calculate directional gradients for each pixel

    """
    def process_image(self):
        """ Create a directional gradient of the image.

        This calculates the component of the gradient along the radial vector of
        the granule.

        This is much more resistant to other granules in the local area.
        Further, the maximum of the gradient is much more reliable than some
        arbitrary threshold value; while the sobel is useful for this, the use of
        an absolute value of the gradient caused problems.
        """

        x_grad, y_grad = self.calculate_gradient()
        x_rad, y_rad = self.get_angle_from_centre()

        self.processed_image = x_grad * x_rad + y_grad * y_rad
        return self.processed_image

    def get_angle_from_centre(self):
        """Return a normalised vector field of the angle from the local centre of the
        granule.
        """
        # Get a vector with the distance from the centre in the x and y directions
        yDist = np.arange(self.granule.crop_width) - self.granule.local_centre[1]
        xDist = np.arange(self.granule.crop_height) - self.granule.local_centre[0]

        # Turn this into a field
        xx, yy = np.meshgrid(xDist, yDist)

        # Normalise to unit vectors
        mag = -np.sqrt(xx ** 2 + yy ** 2)

        return xx / mag, yy / mag

    def calculate_gradient(self, image: np.ndarray = None):
        """ Calculate the gradient field of the image.

        Parameters
        ----------
        image:np.ndarray
            The image to use, if none is provided then use the raw image of the
            granule.

        Returns
        -------
        np.ndarray:
            A XxYx2 array with the gradient field of the granule, the top most slice
            is in the x direction and the second the y direction.

        """
        if image is None:
            image = self.granule.im_smoothed

        kern = kernels.fourth_order

        x_grad = kern.gradient_x(image)
        y_grad = kern.gradient_y(image)
        return x_grad, y_grad
    
class _BoundaryExtractionMagnitude(_BoundaryExtractionMethod):
    """ Extract the boundary of granule using the intensity of the image.
        this is for vesicle-like droplets which appear as a loop under microscopy

    Input Parameters
    ----------

    granule: Granule
        Extract the boundary from this granule.


    Methods
    -------

    process_image
        calculate directional gradients for each pixel

    """
    def process_image(self):
        processed_image = self.granule.im_smoothed
        return processed_image


def collect_fourier_terms(
    fourier_terms: Iterable[BoundaryExtraction],
    frame: MicroscopeFrame,
    granule_tracker,
    plot: bool = False,
    output_dir: Path = None,
) -> pd.DataFrame:
    """ Gather a list of Fourier terms into a single form and add metadata.

    This gathers all the information from a given into a ``pd.DataFrame``.
    """

    components = []
    new_pos = [fourier.granule.image_centre for fourier in fourier_terms]

    granule_ids = granule_tracker.link_granules(new_pos)

    for granule_id, fourier in zip(granule_ids, fourier_terms):

        angles, radii = fourier.angle_sweep(400, samples_per_pixel=15, order=4)
        magnitude, orders, order_1 = fourier.get_fourier_terms(radii)

        major_axis = fourier.granule.properties["major_axis_length"] * frame.pixel_size
        minor_axis = fourier.granule.properties["minor_axis_length"] * frame.pixel_size
        eccentricity = fourier.granule.properties["eccentricity"]
        mean_intensity = fourier.granule.properties["mean_intensity"]
        df_temp = pd.DataFrame(
            {
                "granule_id": granule_id,
                "order": orders.astype(int),
                "magnitude": magnitude,
                "order_1": order_1,
                "x": fourier.granule.image_centre[0],
                "y": fourier.granule.image_centre[1],
                "bbox_left": fourier.granule.bbox[0],
                "bbox_bottom": fourier.granule.bbox[1],
                "bbox_right": fourier.granule.bbox[2],
                "bbox_top": fourier.granule.bbox[3],
                "mean_radius": fourier.mean_radius_pixels * frame.pixel_size,
                "valid": fourier.validate_boundary(),
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "eccentricity": eccentricity,
                "mean_intensity": mean_intensity,
                "timestamp": str(frame.timestamp)
            }
        )
        components.append(df_temp)

        # Plot the outline of the granule
        if plot: #!jl and granule_id < 35:
            plot_save_name = (
                output_dir
                / "tracking/outline"
                / (
                    f"{frame.im_path.stem}--F{frame.frame_num:03d}--G{granule_id:03d}.png"
                )
            )

            fourier.plot(save_name=plot_save_name, dpi=110)

    aggregate = pd.concat(components, ignore_index=False)
    aggregate.insert(0, "frame", frame.frame_num)
    aggregate.insert(0,"im_path",str(frame.im_path))
    return aggregate


class _GranuleLinker:
    """A class for matching points frame by frame, in cases that points are soetimes missing from a frame."""
    def __init__(self, memory=3, max_distance=15):
        self._stored_positions = None
        self._stored_labels = None
        self._memory_counter = {}
        self.max_id = 0
        self.memory = memory
        self.max_distance = max_distance
        self._init = True

    def link_granules(self, positions):
        """Link the new granules to an older granule in the previous ``self.memory`` frame.

        Outline
        -------

        If there are the same number of elements in ``stored_positions``, then they will be
        arranged to minimise the distance between position the arrays.

        If a new granule is near to a granule in a previous frame then we assign the label
        of this granule to the new granule as well. If the granule is new or not near a
        previous graunle then we assign it a new ID that is greater than all other granules.

        Granules will be compared to granules that disapeared in the last ``memory``
        frames

        Outputs
        -------
        labels: The labels for the the input granules in order.
        """ 
       
        if self._init:
            #On the first frame no linking is needed
            self._stored_positions = positions.copy()
            self._stored_labels = np.arange(len(positions), dtype=int)
            labels = np.arange(len(positions), dtype=int)
            self.max_id = len(positions) - 1
            self._init = False
            return labels
        
        #Otherwise, link
        n_granules = len(positions)
        n_stored = len(self._stored_positions)
        # Labels are set to a "bad" initial value, so we can catch the unmatched labels later
        labels = np.ones(n_granules, dtype=int) * -1
        if n_granules < 1:
            raise GranuleNotFoundError

        handled_granules = np.full(len(positions),False)
        handled_stored = np.full(len(self._stored_positions),False)    

        distances =cdist(positions,self._stored_positions)

        smallest_array = min(n_granules,n_stored)

        for i in range(smallest_array):
            # Find the granule that has moved the least between the two frames
            new_index, old_index = np.unravel_index(
                np.nanargmin(distances, axis=None), distances.shape
            )

            # Test if the distances between the granule is larger than a given distance
            displacement = distances[new_index, old_index]
            if self.max_distance is not None and displacement > self.max_distance:
                break

            #if not, these are the same granule
            label = self._stored_labels[old_index]
            labels[new_index] = label
            self._stored_positions[old_index] = positions[new_index]
            handled_granules[new_index] = True
            handled_stored[old_index] = True
            distances[new_index, :] = np.NaN
            distances[:,old_index] = np.NaN

        #handle newly appeared granules
        unhandled = [positions[i] for i,x in enumerate(handled_granules) if x == False]
        if unhandled != []:
            tail_labels = np.arange(self.max_id + 1, self.max_id + 1 + len(unhandled))
            self._stored_labels = np.append(self._stored_labels,tail_labels)
            self._stored_positions = np.append(self._stored_positions,unhandled,axis=0)
            self.max_id += len(unhandled)

            # Fill in unhandled granule IDs
            tail_index = 0
            for i in range(len(labels)):
                if labels[i] == -1:
                    labels[i] = tail_labels[tail_index]
                    tail_index += 1


        #handle missing granules
        missing = [i for i,x in enumerate(handled_stored) if x == False] #position in _storedlists
        missing_ids = [self._stored_labels[i] for i in missing] #granule index
        for_removal = []
        found = []

        for granule_id in self._memory_counter:
            if granule_id not in missing_ids:
                found.append(granule_id)
        for granule_id in found:
            del self._memory_counter[granule_id]

        for granule_id,index in zip(missing_ids,missing):
            if granule_id in self._memory_counter:
                if self._memory_counter[granule_id] == 1:
                    del self._memory_counter[granule_id]
                    for_removal.append(index)
                else:
                    self._memory_counter[granule_id] -= 1
            else:
                self._memory_counter[granule_id] = self.memory

        self._stored_positions = np.delete(self._stored_positions,for_removal,axis=0)
        self._stored_labels = np.delete(self._stored_labels,for_removal)
        # print("In2: ",self._memory_counter, for_removal,self.max_id)
        # print(self._stored_labels)
        # print(self._stored_positions)
        return labels
