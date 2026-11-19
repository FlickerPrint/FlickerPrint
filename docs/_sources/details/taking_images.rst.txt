.. _image_requirements:

Image Requirements
==================

FlickerPrint requires the use of confocal microscopy images as input in order to conduct the flicker spectroscopy analysis.
Details of experiments which are suited to this analysis are given in the `software paper <https://doi.org/10.1101/2025.03.24.645013>`_.

Images in :ref:`these <citing_flickerprint>` works were taken using an Andor dragonfly spinning disk, we used a 100x 1.49NA lens with a 1.5x magnification, or when this was out of service a 60x with a 2x magnification. In another system, we would prefer the highest resolution possible, although consider the trade-off with the NA of the lens if the 60x is better.

Our primary goals in the imaging are to get the highest resolution images and reduce the exposure time of the microscope; the higher order terms of the perturbations fluctuate rapidly, this creates blurring in the interface that we want to minimize. On our Andor spinning disk we can get this down to 5 or 10 ms, this of course requires an increase in the laser power (some care should be taken with HyD sensors on Leica microscopes, they can be damaged by high laser intensities). This causes bleaching of the droplets so only perform this burst once per droplet.

A large time series of images should be taken; this is required us accurately record the fluctuations of the boundary; we typically collect 1,000 frames over about 40 seconds.
The total imaging time should be long enough to cover several fluctuation periods, while ensuring that the properties of the object do not change over the course of the imaging (for example, during the liquid-solid *ageing* transition experienced by some condensates).

Spinning disk notes
-------------------

The following are notes from our personal experience with microscopes, in particular the our Andor dragonfly.

The rate of the image acquisition isn't necessarily linked to the exposure time of the microscope as there may be some significant readout time for the CCD. This can be reduced by using the ``overlap`` setting on the andor and the ``finite burst`` mode, this mode works by gathering a given number of frames as rapidly as possible.

Provided that the exposure time remains constant, increasing the rate of imaging will not change the image quality, but will naturally mean that more data can be collected. Cropping the AOI can increase the imaging rate by only using part of the sensor, this is only really useful if then is only a small area to image. Notably, we want the overall imaging time to cover a few fluctuation periods of the droplets.

There is some subtlety to the spinning disk: the laser will typically illuminate a larger area than the imaging area, causing unnecessary bleaching in the surrounding area. This can be reduced by two settings ``illumination apperature`` and ``illumination mode``. ``Illumination apperature`` is simply the size of the illuminated area, this should by chosen to be as small as possible; it will be obvious if this is too small as edges of the image will be dark. ``Illumination mode`` should be set to ``PD2``, this is best suited to image a smaller area with a high intensity.

Line scan microscope
--------------------

Line scanning microscopes are better able focus on a single granule, with a much lower exposure time and for small FOVs (the size of a granule or two) the LSM is able to take images much more rapidly, however, this scales poorly with the increase in the imaging area. This soon means that the LSM becomes much slower, there is also a subtle effect when the image gets too large; as the image builds up line-by-line, the granule interface may move between these lines, leading to a rolling shutter effect.

In this case the "optimal" resolution should be used--the highest resolution that doesn't run into diffraction limits.

Super resolution methods
------------------------

Most super resolution methods are computational, they rely on taking many images in quick succession and combining these in some manner that allows us to resolve more detail. Unfortunately, this approach does not work well for things rapidly fluctuating within the cytoplasm, as the movements of the interface between frames makes the reconstruction impossible.

We have had some luck with the Ziess Airyscan module, this is an optical super resolution method, this uses a number of sensors and records information from the higher order terms from the Airy disk which leads to an increase in the achieved resolution. STED is also an optical super resolution method, but we have had less luck with this.

We suspect it may be possible to use 3D sim methods, however, we haven't been able to test this ourselves.
