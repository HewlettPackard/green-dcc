# Workload Data Folder

This folder contains datasets used for simulating datacenter workloads. The structure separates raw data, processed datasets, and intermediate files.

---

## Folder structure

```
data/workload/
├── input_raw/                       
│   ├── Alibaba_CPU_Data_Hourly_1.csv
│   ├── Alibaba_CPU_Data_Hourly_2.csv
│   ├── GoogleClusteData_CPU_Data_Hourly_1.csv

├── alibaba_2020_dataset/           
│   ├── result_df_full_year_2020.pkl              # ✅ Main file used in simulation
│   ├── result_df_cropped_with_bandwidth.pkl      # Processed subset (older version)
│   ├── result_df_cropped.pkl                     # Another cropped version (not used)
│   ├── extracted_dfas.csv                        
│   ├── extracted_dfas_with_bandwidth.csv
│   ├── pai_*                                     # Raw Alibaba trace files

├── __init__.py                  # Empty init file (for Python package compatibility)
```

---

## File descriptions

- **`result_df_full_year_2020.pkl`**  
  Main DataFrame used for simulations. Contains all tasks for 2020, already preprocessed with intervals and bandwidth.

- **`result_df_cropped*.pkl`**  
  Partial datasets used in earlier experiments (can be archived if unused).

- **`extracted_dfas*.csv`**  
  Intermediate output from parsing the original Alibaba trace tables.

- **`pai_*.csv`, `*.header`, `*.tar.gz`**  
  Raw trace files from the Alibaba cluster dataset (original source files). These are not accessed during the simulation.

- **`input_raw/`**  
  Additional CSVs from earlier workloads or other datasets (e.g., Google traces). Currently not used.

---

## Usage Note

The simulation pipeline only reads from:

```
alibaba_2020_dataset/result_df_full_year_2020.pkl
```

All other files are optional or legacy.
