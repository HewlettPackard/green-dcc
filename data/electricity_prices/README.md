
# Electricity Prices Dataset

This repository contains standardized **hourly electricity price data** for multiple countries and regions.  
All files are stored as CSVs organized by region and year.
Scripts used to extract the data are also included.

---

## 📁 Folder Structure

```
electricity_prices/
├── processed/                  # Structured CSVs in local time + local currency
│   └── REGION-CODE/
│       └── YEAR/
│           └── REGION-CODE_electricity_prices_YEAR.csv
│
├── standardized/              # Cleaned data in UTC + USD (with local time column)
│   └── REGION-CODE/
│       └── YEAR/
│           └── REGION-CODE_electricity_prices_YEAR.csv
│
├── raw/                       # Raw source data (e.g. ZIPs or API downloads)
│   └── SG/
│       └── zips/
│           └── USEP_from_01-Jan-2020_to_31-Dec-2020.zip
│
├── scripts/                   # Python scripts for extraction + standardization
│   └── extract_prices_<REGION>.py
│   └── standardize_prices_to_utc_usd.py
|   └── plot_prices_standardized.py
│
└── README.md                  # This file

```

---

## 📄 Data Format

### `standardized/REGION/YEAR/REGION_electricity_prices_YEAR.csv`
This is the standardized format for all regions. The files contain the following columns:
```csv
Datetime (UTC),Datetime (Local),Price (USD/MWh)
2024-01-01 00:00:00+00:00,2023-12-31 19:00:00-05:00,32.12
2024-01-01 01:00:00+00:00,2023-12-31 20:00:00-05:00,30.80
...
```

- `Datetime (UTC)`: Timestamps in UTC, used for simulations and cross-region comparison
- `Datetime (Local)`: Local timezone-aware timestamp for human readability
- `Price (USD/MWh)`: Price converted to USD using fixed exchange rates

---

## 🌍 Regions Included

| Code         | Region / Market                          |
|--------------|------------------------------------------|
| AT           | Austria (ENTSO-E)                        |
| AU-NSW       | Australia - New South Wales (AEMO)       |
| AU-VIC       | Australia - Victoria (AEMO)              |
| BE           | Belgium (ENTSO-E)                        |
| BR           | Brazil (ONS)                             |
| CA-ON        | Canada - Ontario (IESO)                  |
| CH           | Switzerland (ENTSO-E)                    |
| CL-SIC       | Chile - Central (CDEC-SIC)               |
| DE-LU        | Germany + Luxembourg (ENTSO-E)           |
| ES           | Spain (OMIE)                             |
| FR           | France (ENTSO-E)                         |
| IN           | India (POSOCO)                           |
| JP-TK        | Japan - Tokyo Area (JEPX)                |
| KR           | Korea (KPX)                              |
| NL           | Netherlands (ENTSO-E)                    |
| PT           | Portugal (OMIE)                          |
| SG           | Singapore (USEP - EMC)                   |
| US-CAL-CISO  | California (CAISO - OASIS / GridStatus)  |
| US-MIDA-PJM  | PJM (Mid-Atlantic, e.g., Virginia)       |
| US-NY-NYIS   | New York State (NYISO)                   |
| US-TEX-ERCO  | Texas (ERCOT)                            |
| ZA           | South Africa (Eskom)                     |

---

## ⚙️ Scripts

### Extraction
Each region has a dedicated script in `scripts/` to extract raw data into the `processed/` folder:

Example:

```bash
python extract_prices_<REGION>.py
```

This will generate files like:

```
processed/<REGION>/<YEAR>/<REGION>_electricity_prices_<YEAR>.csv
```

### Standardization
This script converts local time + currency into unified UTC + USD format:

```bash
python standardize_prices_to_utc_usd.py
```
This will generate files like:

```
standardized/<REGION>/<YEAR>/<REGION>_electricity_prices_<YEAR>.csv
```

---

## 📌 Notes

- **Datetime Standard**: All timestamps in `standardized/` are in `Datetime (UTC)` format, with local time also provided.
- **Datetime format**: `YYYY-MM-DD HH:MM:SS`
- **Currency**: All prices are converted to `USD/MWh`
- **Missing data**: Some hours or days may be unavailable
- **Sources**: All from official or public APIs and datasets
- **Exchange rates**: Fixed exchange rates are used for conversion to USD


---

## 📬 Contact

If you want to contribute or suggest improvements, open an issue or PR.
