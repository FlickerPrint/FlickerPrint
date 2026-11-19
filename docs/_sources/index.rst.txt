
#################################################
Welcome to the FlickerPrint documentation!
#################################################

FlickerPrint is a Python package for determining the interfacial tension and bending ridgidity (as well as other properties) of a population of soft bodies such as condensates or vesicles by analysing their shape fluctuations in confocal microscopy images.
This flicker spectroscopy analysis can be parallised across multiple cores to allow whole populations of soft bodies to be analysed.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    user_guide/user_guide
    details/details

.. toctree::
    :maxdepth: 1

    source_code/module_source_code

.. toctree::
    :maxdepth: 2
    
    find_out_more/find_out_more
    find_out_more/citing_flickerprint



Flicker Spectroscopy
====================

Flicker spectroscopy is a method to measure the properties of biomolecular condensates by measuring the thermal fluctuations of their boundary (measuring the properties of vesicles is also supported).
See the `paper <https://doi.org/10.1126/sciadv.adg0432>`_ for more information about the theory used.


Quick Install
=============

See :ref:`installation` for details of the prerequesites for the package.
FlickerPrint can be installed using pip:

.. tab-set::

   .. tab-item:: Windows

      .. code-block:: bash

         python -m pip install flickerprint

   
   .. tab-item:: macOS and Linux

      .. code-block:: bash

         python3 -m pip install flickerprint


Recording Images
================

FlickerPrint requires the use of confocal microscopy images, taken at (approximately) the equator of the objects of interest.
Systems can be either *in vitro* or in live cells, provided that the condensates are in thermal equilibrium so are not being impacted by external forces or energies and that the objects display shape fluctuations above the resolution limit of FlickerPrint (typically :math:`\approx` 0.1 pixels).

See :ref:`image_requirements` for a guide to take images best suited for analysis.


Run Analysis
============

The package can be run from the command-line.
Create an experiment directory with

.. code-block:: bash

    flickerprint create-project EXPERIMENT_NAME

and run the analysis from that directory with

.. code-block:: bash

    flickerprint run

See :ref:`getting_started` for a short guide or :ref:`workflow` for more details.


Index and Search
================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
