============
Installation
============

Follow the steps below to setup your |F| environment and install the necessary dependencies.

Dependencies
------------

1. Linux OS (tested on Ubuntu 20.04)
2. Python 3.10+
3. Ray **2.4.0** (installed when installing the :code:`requirements.txt` file)
4. Conda_ (for creating virtual environments, optional)
5. Git_ (distributed version control system, optional)
6. Dependencies listed in the :code:`requirements.txt` file

.. _Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
.. _Git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

Installation Steps
--------------------

Clone the latest |F| version from GitHub using:

.. code-block:: bash
    
    git clone https://github.com/HewlettPackard/green-dcc.git

If using SSH, execute:

.. code-block:: bash
    
    git clone git@github.com:HewlettPackard/green-dcc.git

Navigate to the project repository:

.. code-block:: bash
    
    cd green-dcc

Create a Conda_ environment if you prefer using a virtual Python environment to manage packages for this project (optional):

.. code-block:: bash
    
    conda create -n greendcc python=3.10
    conda activate greendcc


Install the required packages using :code:`pip`:

.. code-block:: bash

    pip install -r requirements.txt


