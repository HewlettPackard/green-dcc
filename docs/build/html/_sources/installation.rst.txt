Installation
============

This document shows you how to get Sustain-Cluster up and running.

Prerequisites
-------------
- Python 3.10
- Git  
- A Unix‐style shell (bash, zsh) or PowerShell on Windows

Clone the repository
--------------------
First, grab the code:

.. code-block:: bash

   git clone https://github.com/HewlettPackard/sustain-cluster.git
   cd sustain-cluster

Create and activate a virtual environment
-----------------------------------------
On macOS/Linux:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

On Windows (PowerShell):

.. code-block:: powershell

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

Install Python dependencies
---------------------------
All runtime requirements are listed in `requirements.txt`. Install them with:

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt

(Optional) Install Sphinx for docs
----------------------------------
If you plan to build the docs yourself, you will also need Sphinx and the Furo theme (plus Napoleon for Google-style docstrings). You will need to add the following libraries to the requirements.txt:

.. code-block:: text

    # Add the following to requirements.txt
    sphinx>=6.0
    furo
    sphinx-autodoc-typehints

Build the documentation
-----------------------
From the project root:

.. code-block:: bash

   cd docs
   make html

This will generate HTML under ``docs/build/html``. You can then:

.. code-block:: bash

   # on macOS
   open docs/build/html/index.html

   # on Linux
   xdg-open docs/build/html/index.html

   # on Windows (PowerShell)
   start docs\build\html\index.html

That’s it! You now have Sustain-Cluster installed and its documentation built locally.
