.. _installation:

Installation
============

Prerequisites
+++++++++++++

FlickerPrint is available for Windows, macOS and Linux and requires a working installation of Python 3 (Python 3.9 - Python 3.11, `Python 3.10.8 <https://www.python.org/downloads/release/python-3108/>`_ is recommended) and `Java <https://www.oracle.com/uk/java/technologies/downloads/>`_.
We recommend that you install Java in your home directory for easy access.
There is also the option to use `LaTeX <https://www.latex-project.org/get/>`_ for typsetting of figures, however this is not required.

System-Specific Prerequisites
+++++++++++++++++++++++++++++++

.. tab-set::

   .. tab-item:: Windows

      In addition to Python and Java, you will need to install a C compiler. 
      Please install Visual Studio 2022 Community Edition (or higher) from `here <https://visualstudio.microsoft.com/vs/community/>`_ and install the **Desktop Development with C++** tools and **.NET dektop development** tools.

   .. tab-item:: macOS

      In addition to Python and Java, you will need to install the xcode command line tools if you do not have them already.
      To do this, open the terminal and type:

      .. code-block:: zsh

         xcode-select --install

      .. .. Note

      ..    If you are using an Apple Silicon Mac (likely any Mac made after 2020), you may also need to manually install the h5py python library using a compatable version of HDF5.
      ..    At the time of writing, the version of HDF5 which ships with h5py does not support Apple Silicon.
      ..    This module is used to read and write the data files produced during the analysis.

      ..    To do so, first install `homebrew <https://brew.sh>`_ if you do not have it already:

      ..    .. code-block:: zsh

      ..       /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      ..       echo >> /Users/dev/.zprofile
      ..       echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/dev/.zprofile
      ..       eval "$(/opt/homebrew/bin/brew shellenv)"

      ..    Then use it to install HDF5 and h5py:

      ..    .. code-block:: zsh

      ..       brew install hdf5
      ..       export HDF5_DIR="$(brew --prefix hdf5)"
      ..       python3 -m pip install --no-binary=h5py h5py

   .. tab-item:: Linux

      No Linux-specific software is required, though you will still need a working installation of Python and Java, and a C compiler.

Installation via pip
+++++++++++++++++++++

FlickerPrint can be installed directly using pip:

.. tab-set::

   .. tab-item:: Windows

      .. code-block:: bash

         python -m pip install flickerprint

   
   .. tab-item:: macOS

      .. code-block:: zsh

         python3 -m pip install flickerprint
   
   .. tab-item:: Linux

      .. code-block:: bash

         python3 -m pip install flickerprint


.. Note::

   The ``JAVA_HOME`` and ``PATH`` environment variables must be set correctly so that the Java installation can be found by the Python modules that require it.
   FlickerPrint will attempt to set these automatically on installation, though you can also set them manually using the instructions below.

Building from Source
++++++++++++++++++++

To install FlickerPrint from source, download the source code from `GitHub <https://github.com/FlickerPrint/FlickerPrint>`_.
Open the terminal and navigate to the ``FlickerPrint/src`` directory.
Then install it using pip:

.. tab-set::

   .. tab-item:: Windows

      .. code-block:: bash

         python -m pip install .

   
   .. tab-item:: macOS

      .. code-block:: zsh

         python3 -m pip install .

   .. tab-item:: Linux

      .. code-block:: bash

         python3 -m pip install .

Setting Environment Variables
+++++++++++++++++++++++++++++

As part of the installation process, FlickerPrint will attempt to set the ``JAVA_HOME`` and ``PATH`` environment variables automatically.
If this fails, you may need to set them manually.
Instructions are provided below for each operating system.

.. tab-set::

      .. tab-item:: Windows

         When Java is installed on Windows, the ``JAVA_HOME`` environment variable is set automatically.
         However, if you need to set enviroment vairables manually, you can find instructions for doing so `here <https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/set_1>`_.

      
      .. tab-item:: macOS

         You will need to set the ``JAVA_HOME`` environement variable and add it to your ``PATH``.
         To do this open the terminal and type (changing ``<pathToJava/home>`` to the location of your Java installation):

         .. code-block:: zsh

            export JAVA_HOME=<pathToJava/home>
            export PATH=$JAVA_HOME/bin:$PATH

         You may find it helpful to add the above lines to your config file for easy access in future (optional).
         To do so, open the appropriate config file for your system:

         Likely

         .. code-block:: zsh
            
            nano ~/.zprofile

         Then add the above two lines to the file, save it and relaunch the terminal.
      
      .. tab-item:: Linux

         You will need to set the ``JAVA_HOME`` environement variable and add it to your ``PATH``.
         To do this open the terminal and type (changing ``<pathToJava/home>`` to the location of your Java installation):

         .. code-block:: bash

            export JAVA_HOME=<pathToJava>
            export PATH=$JAVA_HOME/bin:$PATH

         You may find it helpful to add the above lines to your config file for easy access in future (optional).
         To do so, open the appropriate config file for your system:

         Likely

         .. code-block:: bash

            nano ~/.bashrc

         Then add the above two lines to the file, save it and relaunch the terminal.

Installation with Docker
++++++++++++++++++++++++

A Dockerfile is available in the root directory of the `GitHub repository <https://github.com/FlickerPrint/FlickerPrint>`_ for building a `Docker <https://www.docker.com>`_ image of FlickerPrint.
To build the image, open a terminal and navigate to the root directory of the repository.
Then run the following command:

.. code-block:: bash

   docker build -t flickerprint .

To run the image, use the following command:

.. code-block:: bash
   
   docker run -it flickerprint
