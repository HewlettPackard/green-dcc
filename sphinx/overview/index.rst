========
Overview
========

Data Center Cluster Model
--------------------------

High-level overview of the operational model of a |F| data center is given in the figure below.


.. figure:: ../images/hier.png
   :scale: 18 %
   :alt: GreenDCC Framework for Data Center Cluster Management
   :align: center


The high-level components of |F| are:

  - **Top-Level Agent:** Controls the geographic distribution of workloads across the entire DCC. This agent makes strategic decisions to optimize resource usage and sustainability across multiple locations.
  - **Lower-Level Agents:** Manage the time-shifting of workloads and the cooling processes within individual data centers. These agents implement the directives from the top-level agent while addressing local operational requirements.
  - **Additional Controls:** Can include energy storage, among other capabilities. These controls further enhance the system's ability to optimize for multiple objectives, such as reducing the carbon footprint, minimizing energy usage and costs, and potentially extending to water usage.

The figure below shows the |F| framework using two main strategies to optimize data center operations and reduce carbon emissions: geographic and temporal load shifting strategies.

.. image:: ../images/GreenDCCv3.png
   :scale: 15 %
   :alt: Green-DCC Framework demonstrating Geographic and Temporal Load Shifting Strategies
   :align: center

Geographic Load Shifting Strategy 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This strategy involves dynamically moving workloads between different data centers (DC1, DC2, and DC3) based on decisions made by the Top-Level Agent. The goal is to distribute tasks across various locations, taking advantage of regional differences in energy costs, carbon intensity (CI) of the grid, and external temperature. For example, if the top-level agent determines that DC1 has a high CI and a high external temperature, it can transfer some tasks to DC2 or DC3, which may have a lower CI and external temperature at that time. In the figure above on the left, it can be seen that some tasks move from DC1 to DC3 in the first dashed grid arrow, reducing the utilization (workload) of DC1 and increasing the utilization of DC3.

Temporal Load Shifting Strategy 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This strategy involves deferring non-critical/shiftable tasks to future time periods when conditions are more favorable for energy-efficient operation. Tasks are temporarily stored in a Deferred Task Queue (DTQ) and executed during periods of lower CI, lower external temperatures, or lower overall data center utilization. This strategy helps optimize energy consumption and reduce carbon emissions by scheduling workloads during off-peak hours or times when renewable energy sources are more abundant. In the figure above on the right, it can be observed how some tasks are added to the DTQ when a high utilization is detected (or due to other external variables such as CI, temperature, etc), and after some time, those tasks are extracted from the DTQ and computed, increasing the utilization.


Data Center Model 
-----------------------
|F| model consist of cluster of data centers. Each of those data centers has a separate lower level model. High-level overview of an individual Data Center model of a data center is given in the figure below.

.. _sustaindc_model:

.. figure:: ../images/SustainDC.png
   :scale: 25 %
   :alt: Overview of the individual data center
   :align: center


Workloads are uploaded to the DC from a proxy client. A fraction of these jobs can be flexible or delayed to different time periods. The servers that process these jobs generate heat that needs to be removed from the DC. This is done by a complex HVAC system that ensures optimal temperature in the DC. As shown in the figure below, the warm air leaves the servers and is moved to the Computer Room Air Handler (CRAH) by the forced draft of the HVAC fan. Next, the hot air is cooled down to optimal setpoint using a chilled water loop and then send back to the IT room. Parallely, a second water loop transfers the removed heat to a cooling tower, where it is rejected to the outside environment. 

.. _sustaindc_hvac:

.. image:: ../images/Data_center_modelled.png
   :scale: 60 %
   :alt: Overview of the SustainDC HVAC system
   :align: center

Big data centers also incorporate battery banks. Batteries can be charged from the grid during low Carbon Intensity (CI) periods. During higher CI periods, they provide auxiliary energy to the DC.  


|F| connects core environments and external input data sources to model data center operations. 


Core Environments 
-----------------------

|F| consist of three interconnected environments that simulate various aspects of data center operations:

  - **Workload Envronment** - model and control the execution and scheduling of delay-tolerant workloads within the DC 
  - **Data Center Environment** - model and manage the servers in the IT room cabinets that process workloads and the HVAC system and components 
  - **Battery Environment** - simulates the DC battery charging behavior during off-peak hours and provides auxiliary energy to the DC during peak grid carbon intensity periods


These environments work together to provide a comprehensive platform for benchmarking MARL algorithms aimed at optimizing energy consumption and reducing the carbon footprint of DCs.

|F| enables a comprehensive set of customizations for each of the three environments developed in Python. A high-level overview that highlights their individual components, customization capabilities, and associated control problems is given in the figure below.

.. _sustaindc_envs:

.. image:: ../images/schematic.png
   :scale: 60 %
   :alt: Overview of the SustainDC components and parameters
   :align: center


Input Data Sources 
--------------------------

|F| uses few types of external input data to provide realistic simulation environment:

  - **Workload data** - the computational demand placed on the DC
  - **Weather data** - the ambient environmental conditions impacting the DC cooling requirements 
  - **Carbon Intensity data** - the carbon emissions associated with electricity consumption


For more detals on each individual environment, the reinforcement learning algorithms implementeded, the reward functions provided to train the agents, and explanation of external data sources, all needed for succesful |F| model, check the links below. 

.. toctree::
   :maxdepth: 1

   environments
   interconnected
   externaldata
   algorithms
   reward_function
   custom



   