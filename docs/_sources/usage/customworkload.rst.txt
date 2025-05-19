.. _custom-workload-data:

Custom Workload Data
====================

By default, Sustain-Cluster includes workload traces from `Alibaba <https://github.com/alibaba/clusterdata>`_ and `Google <https://github.com/google/cluster-data>`_ data centers. These traces are used to simulate the tasks that the datacenter needs to process, providing a realistic and dynamic workload for benchmarking purposes.

Data Source
-----------

The default workload traces are extracted from:

- **Alibaba 2020 GPU Cluster Trace** (`LINK <https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md>`_)

Processed Dataset Format & Content
----------------------------------

After preprocessing, the Alibaba trace is stored as a Pandas DataFrame in a binary pickle file:

.. code-block:: text

   data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl

Each row in this DataFrame represents a **15-minute arrival interval** (UTC) and contains:

- **tasks_matrix** (NumPy array of shape N×M): detailed per-task features for all tasks arriving in that interval.  
  Columns (in order):
  
  1. ``job_id``  
  2. ``start_time`` (Unix timestamp)  
  3. ``end_time`` (Unix timestamp)  
  4. ``start_dt`` (Python datetime)  
  5. ``duration_min`` (float)  
  6. ``cpu_usage`` (%)  
  7. ``gpu_wrk_util`` (%)  
  8. ``avg_mem`` (GB)  
  9. ``avg_gpu_wrk_mem`` (GB)  
  10. ``bandwidth_gb``  
  11. ``weekday_name`` (e.g., “Monday”)  
  12. ``weekday_num`` (0 = Monday … 6 = Sunday)

Preprocessing Steps
-------------------

To adapt the raw two-month trace for year-long, continuous simulation, we apply:

1. **Duration filtering:** drop all tasks shorter than 15 minutes.  
2. **Temporal extension:** replicate and blend daily/weekly patterns to expand two months → full year.  
3. **Origin assignment:** probabilistically assign each task to a datacenter region based on population weights and local time-of-day activity.  
   (See :any:`utils/workload_utils.assign_task_origins` and main paper § 7.3 for details.)  
4. **Interval grouping:** bucket tasks into 15-minute UTC intervals.

Resource Normalization
----------------------

During simulation, percentage-based resource requests (``cpu_usage``, ``gpu_wrk_util``) and memory percentages are converted into actual resource units. This is implemented in  
:any:`utils/workload_utils.extract_tasks_from_row`.

Usage in Simulation
-------------------

- The simulation loop reads each 15-minute row from the DataFrame.  
- It queries the embedded ``tasks_matrix`` for that interval, converts percentages → units, and enqueues jobs into the cluster model.

Access & Distribution
---------------------

- The full pickle file is distributed alongside a ZIP archive in  
  ``data/workload/alibaba_2020_dataset/``.  
- On first simulation run, if ``result_df_full_year_2020.pkl`` is missing but the ZIP is present, the code automatically extracts the pickle.  
- To swap in your own workload, place your processed ``.pkl`` file (same schema) into the same folder and update the path in your config:

  .. code-block:: python

     DEFAULT_CONFIG["workload_file"] = "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"
