|F|
==========

|F| is a benchmark environment designed to evaluate dynamic workload distribution techniques for sustainable Data Center Clusters (DCC). It aims to reduce the environmental impact of cloud computing by distributing workloads within a DCC that spans multiple geographical locations. The benchmark environment supports the evaluation of various control algorithms, including reinforcement learning-based approaches.

This page contains the documentation for the GitHub `repository <https://github.com/HewlettPackard/green-dcc>`_ for the paper `"Green-DCC: Benchmarking Dynamic Workload Distribution Techniques for Sustainable Data Center Cluster" <https://openreview.net/forum?id=DoDz1IXjDB>`_


.. image:: images/hier.png
   :scale: 18 %
   :alt: Green-DCC Framework for Data Center Cluster Management 
   :align: center

Demo of |F| 
------------------


A demo of |F| is given in the Google Colab notebook below

.. image:: images/colab-badge.png
   :alt: GoogleColab
   :target: https://colab.research.google.com/drive/1NdU2-FMWxEXN2dPM1T9MSVww5jpnFExP?usp=sharing



Key features of |F|
---------------------

  - Dynamic time-shifting of workloads within data centers and geographic shifting between data centers in a cluster.
  - Incorporation of non-uniform computing resources, cooling capabilities, auxiliary power resources, and varying external weather and carbon intensity conditions.
  - A dynamic bandwidth cost model that accounts for the geographical characteristics and amount of data transferred.
  - Realistic workload execution delays to reflect changes in data center capacity and demand.
  - Support for benchmarking multiple heuristic and hierarchical reinforcement learning-based approaches.
  - Customizability to address specific needs of cloud providers or enterprise data center clusters.

|F| provides a complex, interdependent, and realistic benchmarking environment that is well-suited for evaluating hierarchical reinforcement learning algorithms applied to data center control. The ultimate goal is to optimize workload distribution to minimize the carbon footprint, energy usage, and energy cost, while considering various operational constraints and environmental factors. The figure above illustrates the hierarchical |F| framework for Data Center Cluster management. In this framework:

  - **Top-Level Agent:** Controls the geographic distribution of workloads across the entire DCC
  - **Lower-Level Agents:** Manage the time-shifting of workloads and the cooling processes within individual data centers
  - **Additional Controls:** Can include energy storage, among other capabilities. These controls further enhance the system's ability to optimize for multiple objectives

The hierarchical structure allows for coordinated, multi-objective optimization that considers both global strategies and local operational constraints.

.. image:: images/GreenDCCv3.png
   :scale: 15 %
   :alt: Green-DCC Framework demonstrating Geographic and Temporal Load Shifting Strategies
   :align: center

|F| uses two main strategies to optimize data center operations and reduce carbon emissions: geographic and temporal load shifting strategies.

  - **Geographic Load Shifting:** Dynamically moves workloads between different data centers based on decisions made by the Top-Level Agent
  - **Temporal Load Shifting:** Defers non-critical/shiftable tasks to future time periods within a single data center when conditions are more favorable for energy-efficient operation

.. toctree::
   :hidden:
   
   installation
   gettingstarted
   overview/index
   usage/index
   evaluation
   Code<code/modules>
   contribution_guidelines
   references
   genindex
   modindex

