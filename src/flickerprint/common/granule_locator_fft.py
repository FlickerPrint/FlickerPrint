#!/usr/bin/env python

"""A faster implementation of the DoG method
=========================================

We make use of the FFT to speed up the blurring step in the difference of Gaussian method.
We can also exploit the linearity of the FFT to save on a transform, by pre-diffing the kernels.


Boundary Conditions
-------------------

The FFT inheriently assumes that the signal is periodic, which corresponds to a "wrapping" of the
convolution around the edges of the images, which rarely makes sense for our images. So, we can pad
the image with zeros for a size (k/2 + 1)? to match the behaviour of the "constant" 0 edge
condition. This does have some performance implications, however, so this can be turned off at the
cost of edge artifacts.

Image Size
----------

FFTs are naturally faster on images of size $2^N$, which typically aligns with the image size
provided by the microscope, however, the addition of padding to account for boundary conditions,
often requires an image of size $2^{N+1}$, significantly slowing things down.

Instead, we can pad the image to size $3 * 2^{N-1}$ or $5 * 2^{N-2}$ instead, which are typically
slightly faster than the larger $2^{N+1}$ sizes.

"""

import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.fft import irfft2, rfft2
from skimage.feature import blob

import flickerprint.common.granule_locator as gl
from flickerprint.common.configuration import config


@dataclass
class GranuleDetectorFFT(gl.GranuleDetector):
    """An FFT based implementation of the DoG method."""

    blurrer: "DeltaBlurrer"

    def labelGranules(self):

        threshold = float(config("image_processing", "granule_minimum_intensity"))
        method = config("image_processing", "method")

        if method == "gradient":
            self.processed_image = self.frame.im_data
        elif method == "intensity":
            self.processed_image = gl._process_vesicles(self.frame.im_data)
        else:
            raise ValueError("no granule detection method {}".format(method))

        self.granule_locations = _detect_granules_dog_fft(
            image=self.processed_image,
            blurrer=self.blurrer,
            threshold=threshold,
        )

        self.labelled_granules = self._fillGranules()


def _detect_granules_dog_fft(image, blurrer: "DeltaBlurrer", threshold=0.1, overlap=0):
    fft_blobs = blurrer.difference_of_guassians(
        image, overlap=overlap, threshold=None, threshold_rel=threshold
    )
    return fft_blobs


def generate_sigmas(min_sigma, max_sigma, sigma_ratio=1.6):
    """Generate a geometric progression of sigma values.

    We note that the final value may be larger than ``max_sigma``
    """
    max_min_ratio = max_sigma / min_sigma
    if max_min_ratio <= 1:
        raise ValueError("Max sigma should be greater than min_sigma")

    # From skimage
    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(np.log(max_min_ratio) / np.log(sigma_ratio) + 1)

    sigma_list = np.array([min_sigma * (sigma_ratio**i) for i in range(k + 1)])
    return sigma_list


class _Blurrer(ABC):
    def __init__(
        self,
        sigmas: np.ndarray,
        plane_shape,
        truncate_length: int = 4,
        fft_len: Optional[int] = None,
        _allow_power_of_three: bool = True,
        _allow_power_of_five: bool = True,
    ):
        self.sigmas = np.array(sigmas)
        self.max_sigma_actual = int(self.sigmas.max() + 0.5)
        self.base_kernel_size = int(truncate_length * 2 * self.max_sigma_actual + 1)

        if fft_len is None:
            largest_image_dim = np.max(plane_shape)
            min_fft_size = largest_image_dim + self.base_kernel_size - 1

            next_power_of_two = int(
                get_next_power_of_2(largest_image_dim + self.base_kernel_size - 1)
            )

            power_of_three_term = 3 * next_power_of_two // 4
            power_of_five_term = 5 * next_power_of_two // 8

            if _allow_power_of_five and power_of_five_term > min_fft_size:
                logging.info(
                    f"Using power of five term: {power_of_five_term} > {min_fft_size}"
                )
                self.fft_size = power_of_five_term
            elif _allow_power_of_three and power_of_three_term > min_fft_size:
                logging.info(
                    f"Using power of three term: {power_of_three_term} > {min_fft_size}"
                )
                self.fft_size = power_of_three_term
            else:
                logging.info(
                    f"Falling back to power of two term {next_power_of_two} from {min_fft_size}"
                )
                self.fft_size = next_power_of_two
        else:
            self.fft_size = fft_len

        self.kernels = self.calculate_kernels()

    def _convolve(self, image: np.ndarray):
        """
        Convolve the image with the kernels.

        This works with, and returns, the oversized padded images.
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got: {image.shape}")
        image_fft = rfft2(image)

        plane_shape = image.shape
        out_im = np.zeros([len(self.kernels), *plane_shape])

        for index_, K in enumerate(self.kernels):
            Y = K * image_fft
            # In theory, the out keyword should help here, but it's not backwards compatible
            # and doesn't seem to work anyway...
            out_im[index_] = irfft2(Y, s=plane_shape)

        return out_im

    def convolve_image(self, image: np.ndarray):
        """Convolve the image with the kernels.

        This expects a YX image and will return a σYX image, which is the form expected for the
        peak detection. 

        Note that for the `DeltaBlurrer` this will return the difference
        of Gaussian version.
        """

        if image.ndim != 2:
            raise ValueError(f"Expected image in form YX, got shape {image.shape}")

        image_padded = np.zeros((self.fft_size, self.fft_size))
        image_padded[: image.shape[0], : image.shape[1]] += image

        blurred_image = self._convolve(image_padded)
        offset = (self.base_kernel_size - 1) // 2

        return blurred_image[
            ..., offset : offset + image.shape[0], offset : offset + image.shape[1]
        ]

    @property
    def n_sigma(self):
        return len(self.sigmas)

    @abstractmethod
    def calculate_kernels(self):
        pass


class Blurrer(_Blurrer):
    """Apply the Guassian blur to some images.

    This can then be used to calcute the image cube requried for the Difference of gaussian blob
    detection. Note however, that the ``DeltaBlurrer`` class can do this directly and somewhat
    faster.
    """

    def calculate_kernels(self):
        def create_padded_kernel(sigma: float):
            base_kernel = gaussian_kernel_2d(self.base_kernel_size, sigma)
            padded_kernel = np.zeros((self.fft_size, self.fft_size))
            padded_kernel[: self.base_kernel_size, : self.base_kernel_size] = (
                base_kernel
            )

            fft_kernel = rfft2(padded_kernel)

            return fft_kernel

        return [create_padded_kernel(sigma) for sigma in self.sigmas]


class DeltaBlurrer(_Blurrer):
    """
    Locate blobs in the image using a Difference of Gaussian method.
    ================================================================

    This method is significantly faster than the scikit ``blob_dog`` method due to using
    multiplication in Fourier space rather than convolution. We can further speed up the method by
    precomputing the FFTs of the Gaussian kernels as these are the same across all images and also
    by "pre-diffing" the kernels and doing "G_i - G_{i+1}" in Fourier space, as this saves us an
    iFFT.

    Workflow
    --------

    This class precomputes the Fourier Gaussian kernels and then applies them to an image in the
    ``difference_of_gaussians`` method.

    ```
    blurrer = DeltaBlurrer(sigmas, plane_shape)
    for z in range(0, n_z, block_size):
        sub_image = image[z : z + block_size]
        blobs = blurrer.difference_of_gaussians(sub_image, **blob_detection_kwargs)
    ```

    FFT Length and Edge Effects
    ---------------------------

    FFTs are typically much faster when working on arrays of size 2**N, which typically matches our
    microscopy sizes. However, FFTs treat the images as periodic and so would introduce "wrapping"
    artifacts which are typically undesirable. To get around this, we can pad the array with zeros
    (or, in theory, a reflection), this requires an array of size ``(image_len + kernel_size - 1)``.

    For our typical image size of 1,024, using the next power of two 2,048 can slow the FFT approach
    to the point where this is slower than the scikit approach. Instead, we can use a size of form
    3*2**N or 5*2**N and still see sizable improvements.

    By default, ``DeltaBlurrer``, will try to find the lowest FFT array size of the form 3*2**N or
    5*2**N that is larger than (image_len + kernel_size - 1), however, this can be overridden with
    the ``fft_len`` keyword, setting this to ``next_power_of_2(image_len)`` will typically be
    somewhat faster at the cost of edge artifacts.

    Kernel Size
    -----------

    By default, we follow scikit convention and use 4 standard deviations (of the largest sigma) as
    the cut-off for the Gaussian kernel (as given by the ``truncate_length`` keyword). The FFT
    approach is much less sensitive to the size of the kernel, unless this means crossing over a new
    ``fft_len`` boundary. Tweaking this might lead to a small improvement in performance at the
    cost of accuracy.

    """

    def calculate_kernels(self):
        def create_padded_kernel(sigma_one: float, sigma_two: float):

            kernel_one = gaussian_kernel_2d(self.base_kernel_size, sigma_one)
            padded_kernel_one = np.zeros((self.fft_size, self.fft_size))
            padded_kernel_one[: self.base_kernel_size, : self.base_kernel_size] = (
                kernel_one
            )

            kernel_two = gaussian_kernel_2d(self.base_kernel_size, sigma_two)
            padded_kernel_two = np.zeros((self.fft_size, self.fft_size))
            padded_kernel_two[: self.base_kernel_size, : self.base_kernel_size] = (
                kernel_two
            )

            fft_kernel = rfft2(padded_kernel_one - padded_kernel_two)

            return fft_kernel

        return [
            create_padded_kernel(sigma_one, sigma_two)
            for sigma_one, sigma_two in itertools.pairwise(self.sigmas)
        ]

    def difference_of_guassians(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        overlap: float = 0.5,
        threshold_rel: Optional[float] = None,
    ):
        """
        Perform the Difference of Gaussian blob detection.
        ==================================================

        Expects an image with dimensions, YX.  We've had some luck when the CPU cache can be
        saturated, so performance gains can be had when T ~ 8 compared to running single planes,
        however, this messes with the flow program flow quite a lot, so we don't use it for now.

        This returns an array of [Y, X, R] values, where Y and X are the coordinates of the blob,
        and R, the approximate radius.

        Parameters match those found in the skimage ``blob_dog`` method, see the ``__init__`` method
        for more info.

        """
        if image.ndim != 2:
            raise ValueError(
                f"Image should be of two dimensions, got shape {image.shape}"
            )

        blurred_image = self.convolve_image(image)

        # Scale the blurred image for consistency with skimage, otherwise ``threshold_abs`` breaks
        sigma_ratio = self.sigmas[1] / self.sigmas[0]
        scale_factor = 1 / (sigma_ratio - 1)
        blurred_image *= scale_factor

        # Convert the image into YXσ order to match scipy version
        sigma_delta = np.transpose(blurred_image, (1, 2, 0))
        local_maxima = blob.peak_local_max(
            sigma_delta,
            threshold_abs=threshold,
            threshold_rel=threshold_rel,
            exclude_border=False,
            footprint=np.ones((3,) * blurred_image.ndim),
        )

        #
        # This mostly follows as in skimage.feature.blob
        #
        if len(local_maxima) == 0:
            return

        # The local maxima function gives the last column as the plane with the maxima
        # point, as this is the sigma dimension, we then convert this index into a sigma
        # value.
        # We've simplified somewhat as we don't have isotropic sigmas
        lm = local_maxima.astype(float)
        sigmas_of_peaks = self.sigmas[local_maxima[:, -1]]

        # Remove sigma index and replace with sigmas
        lm[:, -1] = sigmas_of_peaks

        pruned_blobs = blob._prune_blobs(lm, overlap, sigma_dim=1)
        return pruned_blobs


def get_next_power_of_2(n):
    """Return the first power of two >= n."""
    return np.power(2, int(np.ceil(np.log2(n - 1))))


def gaussian_kernel_2d(size, sigma: float):
    offset = 0
    ax = np.arange(size) - size // 2 + offset
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    factor = 2 * np.pi * sigma**2
    return kernel / factor
