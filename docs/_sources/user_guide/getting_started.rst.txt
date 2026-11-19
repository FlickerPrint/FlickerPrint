.. _getting_started:

===============
Getting started
===============

This page contains information on how to get started with FlickerPrint, including creating an experiment, configuring the analysis, running the analysis, and viewing the results.
More detailed information is available in the :ref:`workflow` section of the documentation.

Creating an experiment
++++++++++++++++++++++

FlickerPrint runs all analysis from an ``experiment directory``.
This contains the experiment's configuration file, where settings can be adjusted, as well as the save locations for all datafiles and figures.
See :ref:`directory_structure` for more details.

Running

.. code-block:: bash

  flickerprint create-project EXPERIMENT_NAME

will create an experiment directory ``EXPERIMENT_NAME`` on disk relative to the current location.

Configuring the Analysis
++++++++++++++++++++++++

FlickerPrint analysis is configured using a configuration file, ``config.yaml``, which is created in the experiment directory when you create the project.
The easiest way to configure the analysis is by using the parameter selection tool which runs locally on your computer and opens in a brower window. 
The parameter selection tool can be started using the command

.. code-block:: bash

  flickerprint parameter-selection

which will open it in your default web browser.

Open videos in the parameter selection tool by clicking the ``Open Microscope Video`` button and selecting the videos you want to analyse, or dragging and dropping them onto the ``Open Microscope Video`` button.
You can open multiple videos at once.

Note that if your video does not contain pixel size information, you will need to enter the pixel size in the parameter selection tool.

Use the sliders to adjust the parameters for the analysis, to maximise the number of condensates which are detected.
It is advised that you check the parameters against multiple frames in different videos to ensure that the parameters are robust and not overfitted to a single frame.
The detection method should be set to ``gradient`` for condensates and ``intensity`` for vesicles.

You should also configure the ``Additional Parameters`` needed for the analysis:

``Image Path``
  The path to the image files (or the path to a single image file). Leaving this as ``default_images`` specifies the ``images`` folder of the experiment sub-directory.

``Image Regular Expression``
  `Glob pattern <http://man7.org/linux/man-pages/man7/glob.7.html>`_ for images to include. FlickerPrint version 1.0 and newer support `Bioformat-compliant microscopy <https://doi.org/10.1083/jcb.201004104>`_ file formats.

``Experiment Name``
  Human readable name for the experiments, used in organisation and plotting. Experiments with the same name will be combined by the plotting routines.

``Experiment Temperature``
  The temperature at which the experiment was performed, in degrees Celsius.

``Plot Detector Images``
  Whether to plot the images of the detected condensates and their boundaries.

``Plot Fluctuation Spectra``
  Whether to plot the fluctuation spectra for each condensate.

Once you are happy with the parameters, set a save location for the configuration file (save to the ``experiment directory`` directly or save to yuor downloads folder and move the file manually), and click ``Save Parameters``.


See :ref:`configuration_values` for more information about additional advanced parameters in the configuration file.

Running FlickerPrint
++++++++++++++++++++


FlickerPrint's manager handles running the full analysis through the command line, though you can also run parts of the analysis separately or use individual functions from the package if you wish.
The manager is made available to you when you install FlickerPrint.

To use FlickerPrint on the experiment directory you just created, move your terminal into the directory and call

.. code-block:: bash

  flickerprint run

This will analyse all images in the directory you have specified in the config file.
You can optionally add the ``-c CORES`` flag to specify the maximum number of cores that should be used (this is 1 by default).

Analysing results
+++++++++++++++++

The results from FlickerPrint are stored in the ``aggregate_fittings.h5`` file in the experiment directory. 
All values are standard c-types so you can use your favourite software to filter and analyse the results.
For more information, see :ref:`aggregate_fittings.h5`.

FlickerPrint also contains a Graphical User Interface tool for looking at the output of the analysis.
These can be called using

.. code-block:: bash

  flickerprint view-output

which will open the GUI in your default web browser.




Version number
++++++++++++++

The version of FlickerPrint which is currently installed can be found using

.. code-block:: bash

  flickerprint version