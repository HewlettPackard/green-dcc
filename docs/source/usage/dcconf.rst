.. _dcconf_ref:

Data Center Configuration File
==============================

This file defines one or more data center entries under the top-level `datacenters` key. Each entry describes the compute, memory, and geographic characteristics of a single data center. Below is an example structure; see the field descriptions that follow.

.. code-block:: yaml

   datacenters:
     - dc_id: 1
       location: "US-CAL-CISO"
       timezone_shift: -7
       population_weight: 0.18
       total_cores: 50000
       total_gpus: 1000
       total_mem: 80000
       dc_config_file: "configs/dcs/dc_config.json"

     - dc_id: 2
       location: "DE-LU"
       timezone_shift: 1
       population_weight: 0.22
       total_cores: 85000
       total_gpus: 600
       total_mem: 80000
       dc_config_file: "configs/dcs/dc_config.json"

     - dc_id: 3
       location: "CL-SIC"
       timezone_shift: -5
       population_weight: 0.20
       total_cores: 110000
       total_gpus: 300
       total_mem: 60000
       dc_config_file: "configs/dcs/dc_config.json"

     - dc_id: 4
       location: "SG"
       timezone_shift: 8
       population_weight: 0.25
       total_cores: 15000
       total_gpus: 700
       total_mem: 50000
       dc_config_file: "configs/dcs/dc_config.json"

     - dc_id: 5
       location: "AU-NSW"
       timezone_shift: 11
       population_weight: 0.15
       total_cores: 25000
       total_gpus: 300
       total_mem: 60000
       dc_config_file: "configs/dcs/dc_config.json"

Field Reference
---------------

**datacenters** (list)  
:   A sequence of data center definitions.

For each item in the list:

- **dc_id** (integer)  
  Unique identifier for the data center.

- **location** (string)  
  A short code representing the geographic region (e.g., `"US-CAL-CISO"`, `"DE-LU"`, `"SG"`).

- **timezone_shift** (integer)  
  Offset in hours from UTC for local time alignment (e.g., `-7` for PDT, `+1` for CET).

- **population_weight** (float)  
  Relative weight reflecting the fraction of total user population served by this data center.

- **total_cores** (integer)  
  Total number of CPU cores available in the data center.

- **total_gpus** (integer)  
  Total number of GPU devices available.

- **total_mem** (integer)  
  Total memory capacity in GB.

- **dc_config_file** (string)  
  Path to an additional JSON file containing advanced or region-specific parameters (e.g., rack layouts, power limits).

Usage
-----

1. Place your `datacenters.yaml` in the `configs/` directory.  
2. Read this file via the `load_yaml` loader in `train_rl_agent.py`.  

See also :ref:`mainconf_ref` for how to integrate this configuration into the main environment setup.
