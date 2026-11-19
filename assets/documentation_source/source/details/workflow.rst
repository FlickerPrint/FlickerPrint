.. _workflow:

========
Workflow
========

This page covers the setps of the analysis in more detail, including details of the workflow and the files created.
The automatic workflow can be called using 

.. _automatic_workflow:
.. code-block:: bash

  flickerprint run [-c CORES]

and is the easiest way to use FlickerPrint.
However, the details on this page may be useful if you wish to customise the workflow.
For detailed information on the funtions called during each step, see the :ref:`module_source_code` section.


There are broadly five main steps to using FlickerPrint.
These are:

1. Creating the experiment directory
2. Setting parameters in the config file
3. Processing the Microscope Images
4. Fitting the flikcer spectrum
5. Analysis of the resulting data (optionally using FlickerPrint or your favourite software)

Here, we go through each.

Before you use FlickerPrint, you will also need to collect suitable microscopy images.
Details on how to do this can be found in :ref:`image_requirements`.

Creating the Experiment Directory
=================================

All alaysis using FlickerPrint is completed inside of an *experiment directory*.
This is the save location for all intermediate and final datafiles, as well as figures.
Optionally, microscope images can also be stored in this directory.

Experiment directories can be created by calling

.. code-block:: bash

  flickerprint create-project PROJECT_NAME

which will create an experiment directoy called ``PROJECT_NAME`` in the current directory.
For details on the contents of thw experiment directory, please see :ref:`directory_structure`.

Setting Parameters in the Config File
=====================================

The :ref:`configuration file<configuration_values>` stores all of the settings for a particular experiment.

The imaging parameters are also partiuclarly important as they control which condensates are detected (though generally, they do not affect the determined boundary; see the `FlickerPrint release paper <https://doi.org/10.1101/2025.03.24.645013>`_ for details).

For consistency reasons, similar experiments should use the same imaging parameters.

The ``config.yaml`` file can be edited directly, though **the easiest way to modify the imaging parameters is to use the parameter selection tool** which can be called using

.. code-block:: bash

  flickerprint parameter-selection

which runs the tool locally on your computer and opens it in a web browser.
This allows you to open microscope videos, adjust the imaging parameters and see live feedback on which condensates are detected. The imaging parameters can then be saved to the ``config.yaml`` file.




FlickerPrint also contains an optional Bayesian Inference tool to automatically determine the best imaging parameters to use by maximising the number of condensates found, whilst minimising the number that will fail the condensate filters.
Often, it is easiest to use the graphical parameter selection tool, though the Bayesian inference tool can be useful if analysing very large numbers of videos.
This can be called using

.. code-block:: bash

  flickerprint parameter-optimisation

which selects 2 frames from up to 12 videos in order to determine the optimal imaging parameters.
The parameters are written into the config file and the old parameters in the config file are written to ``old_parameter.txt``.

.. _image_processing:

Image Processing
================

The first step of the main analysis is to locate the objects of interest within each frame and determine their bounaries.
A Fourier transform is then used to extract the fluctuation modes of the boundaries.
This is the first stage of the :ref:`automatic_workflow`; FlickerPrint will check the :ref:`configuration file<configuration_values>` and run the analysis on all microscope images in the specified ``image_directory`` which match the ``image_regex``.

.. Note::

  ``image_directory`` can either be a single microscope image or a directory of images.

The image processing step can be paralellised over multiple cores up to a maximum of one core per microscopy file.
This limitation is due to the Java virtual machine which is used to open the images.

For each microscope file a corresponding :ref:`fourier.h5` file containing the location of the objects of interest and their experimental spectra is created in the ``fourier`` directory.

Optional validation images can also be created to show the locations of the granules detected within the image and the location of their detected boundaries; this is controlled using the ``granule_images`` parameter in the configuration file.
To save space, only data from every 100 :sup:`th` frame is plotted.

The image processing step can be manually run on files by calling

.. code-block:: bash

   flickerprint process-image [-i INPUT_IMAGE] [-o OUTPUT_DIR] [-c CORES]

.. seealso::

  The above steps are managed by :ref:`process_image` including saving the results to :ref:`fourier.h5` and the creation of image detection figures.

A break down of the steps used is covered in the following:

Opening Microscopy Files
------------------------

A ``FrameGen`` object returns the frames, and metadata, of the microscope file one-by-one.

:ref:`Frame Gen Details<frame_gen>`

Granule Detection
-----------------

We use the Difference of Gaussians method which is a blob-detection algortihm to determine the location of objects of interest in the image.
A flood fill algorithm then returns athe approximate extent of the objects so that they can be filtered and their boundaries determined.

:ref:`Granule Detection Details<granule_detection>`

Boundary Drawing
----------------

There are two methods which can be used to determine the boundary of the objects of interest, depending on whether the objects appear in the microscoopy images as filled 'blobs' or as outlines:

* ``gradient`` - Best for filled objects such as condensates
* ``intensity`` - Best for outlines such as vesicles

The ``gradient`` method allows for the boundary of a filled 'blob' to be determined to sub-pixel (typically 1/15 of a pixel) precision.
The ``inensisty`` method allows the boundary to be determined as a contour of maximum intensity within an image.
Therefore, it is best used with outline images (and in particular vesicles).

The boundary is then Fourier transformed to extract the fluctuation modes.

This step also handles linking the objects of interest between frames.

:ref:`Boundary Extraction Details<boundary_extraction>`

Spectrum Fitting
================

In the second step of the main analysis, once the experimental spectrum has been measured, we fit a theoretical spectrum to find the best fit values of interfacial tension :math:`\sigma` and bending rigidity :math:`\kappa`.
The theoretical spectrum for the time average of the square of the :math:`q`:sup:`th` Fourier Component, :math:`\left\langle \left| \mathcal{v}_q\right|^2\right\rangle` is given by

.. math::
  \left\langle \left| \mathcal{v}_q\right|^2\right\rangle=\frac{k_BT}{\kappa} \sum_{l=q}^{l_\max} \frac{N_{l q}^2 P_{l q}^2(0)}{(l+2)(l-1)[l(l+1)+\bar{\sigma}]},

where :math:`k_BT` is the thermal energy of the system and :math:`N_{lq}P_{lq}(0)` is the value of the normalised Legendre polynomial at the equator.
A full derivation of this model can be found in the Supplementary Material of `our paper <https://doi.org/10.1126/sciadv.adg0432>`_.

We use a parameter grid search, followed by a least squares fit with a custom minimiser given by

.. math::
  \varepsilon_{log}=\sum_q\left|\log _{10}\left[\frac{\left|F_{q, \text { theo}}\right|^2}{\left|F_{q, \text { exp}}\right|^2}\right]\right|

to ensure that the fit of each Fourier mode is equally weighted even when their amplitudes vary by several orders of magnitude.

As with the image processing step, this can be run as part of the :ref:`automated workflow<automatic_workflow>`. 
However, it can also be called manaually using 

.. code-block:: bash

  flickerprint spectrum-fitting WORKING_DIR [-c CORES]

You can set ``WORKING_DIR`` to ``.`` if you are currently in the eperiment directory.

This step takes in the :ref:`fourier.h5` files created in the image processing step for each microscope image and returns a single :ref:`aggregate_fittings.h5` file.

Since the fitting of spectra for several thousand condensates can be quite computationally expensive, this process can be split across multiple cores (up to a maximum of one per microscope file) using the ``-c`` flag.

.. seealso::

  Individual spectrum fitting is handled :ref:`here <spectrum_fitting>` and is handled by a :ref:`manager<extract_physical_values>`.

Analysing the Results
=====================

The outputs of the main FlickerPrint analysis are stored in the :ref:`aggregate_fittings.h5` file in the experiment directory.
All values are stored as standard c-types so can be read by your favourite data analysis software (e.g. your own Python scripts).

However, FlickerPrint also provides two methods for filtering and analysing these datasets; one with a Graphical User Interface and one based on the command-line.
Both allow for datasets to be compared based on the experiment name.

Graphical Analysis Tool
-----------------------

You can launch the graphical application by calling

.. code-block:: bash

  flickerprint view-output

from the terminal which will open a web page to show the graphical user interface.

.. warning::

  It is important that you do not close the terminal window that you started running the analysis app from as this is where the app is actually running on your computer; the web page just displays the output and allows you to interact with the contents.

From this web page, you can open your aggregate_fittings files, filter the data and produce plots to show the distributions of the measured parameters, which can then be saved for later use.

Command-line Analysis Tool
--------------------------

The command-line tool works in much the same way as the graphical tool.
It produces plots to show the distribution of each measured parameter, as well as 2D histograms to allow for comparison between parameters.
These plots are saved into the ``figs`` folder of the experiment directory.
The data are filtered according to the following fliters:

* ``fitting_error`` :math:`< 0.5` - Ensures that only condensates where the theoretical model is a good fit to the experimental spectrum are included.
* ``sigma`` :math:`> 1e-10` - Some condensates have spuriously low interfacial tensions so are flitered out.
* ``pass_rate`` :math:`> 0.6` - Ensures that only condensates which pass the internal flitering (e.g. for having a valid boundary) in at least 60% of the frames they are in are counted.
* ``fitting_difference`` :math:`> 0.03` - Only include condensates where bending rigidity makes a contribution to the flicker spectrum.

The command-line analysis tool can be run by calling

.. code-block:: bash

  flickerprint view-output-terminal