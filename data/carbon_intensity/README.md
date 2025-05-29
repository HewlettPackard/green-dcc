# README: Carbon Intensity Data Organization

## Folder Structure
The dataset is organized by **location and year** to facilitate easy access and analysis.

### **Structure:**
```
ci_data_organized/
│── [LOCATION]/
│   ├── [YEAR]/
│   │   ├── [LOCATION]_[YEAR]_hourly.csv
```

### **Example:**
```
ci_data_organized/
│── US-NY-NYIS/
│   ├── 2021/
│   │   ├── US-NY-NYIS_2021_hourly.csv
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│
│── DE/
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
```

## **Locations and Naming Convention**
Each folder represents a specific **geographical region (grid zone)**. The CSV files inside follow the pattern:
```
[LOCATION]_[YEAR]_hourly.csv
```
where:
- **[LOCATION]** is the regional grid ID (e.g., `US-NY-NYIS`, `DE`, `SG`).
- **[YEAR]** is the year of data collection (e.g., `2021`, `2022`).
- **hourly** indicates that data is recorded at an **hourly** resolution.

### **Selected Locations and Their Acronyms**
| Acronym | Region |
|---------|--------|
| **US-NY-NYIS** | New York, USA (NYISO Grid) |
| **US-MIDA-PJM** | Mid-Atlantic, USA (PJM Grid) |
| **US-TEX-ERCO** | Texas, USA (ERCOT Grid) |
| **US-CAL-CISO** | California, USA (CAISO Grid) |
| **DE** | Germany (National Grid) |
| **CA-ON** | Ontario, Canada (IESO Grid) |
| **SG** | Singapore (National Grid) |
| **AU-VIC** | Victoria, Australia (AEMO Grid) |
| **AU-NSW** | New South Wales, Australia (AEMO Grid) |
| **CL-SEN** | Chile (Sistema Eléctrico Nacional) |
| **BR-CS** | Brazil (Central-South Region Grid) |
| **ZA** | South Africa (Eskom Grid) |
| **JP-TK** | Tokyo, Japan (TEPCO Grid) |
| **IN-WE** | Western India (Indian National Grid) |
| **KR** | South Korea (KEPCO Grid) |
| **GB** | Great Britain (National Grid ESO) |

## **Data Format (CSV Files)**
Each CSV file contains hourly carbon intensity and energy mix information. The columns are as follows:

### **Columns Explanation:**
| Column Name | Description |
|-------------|-------------|
| **Datetime (UTC)** | Timestamp in Coordinated Universal Time (UTC) for the recorded data. |
| **Country** | Name of the country for the electricity grid. |
| **Zone Name** | Name of the electricity zone within the country. |
| **Zone Id** | Unique identifier for the electricity zone. |
| **Carbon Intensity gCO₂eq/kWh (direct)** | Carbon emissions from electricity production **without lifecycle emissions**. |
| **Carbon Intensity gCO₂eq/kWh (LCA)** | Lifecycle carbon emissions, including fuel extraction and infrastructure. |
| **Low Carbon Percentage** | Percentage of energy from **low-carbon** sources (e.g., nuclear, hydro, wind, solar). |
| **Renewable Percentage** | Percentage of energy from **renewable** sources (e.g., wind, solar, hydro, biomass). |
| **Data Source** | Source from which the data is obtained. |
| **Data Estimated** | `True` if data is **estimated**, `False` if measured. |
| **Data Estimation Method** | Description of how missing or estimated data was calculated. |

## **Usage**
This dataset is intended for **simulating workload transfers between global data centers** to optimize energy consumption and reduce global carbon emissions.

### **Potential Use Cases:**
- Analyzing historical carbon intensity trends.
- Simulating dynamic workload shifting to reduce emissions.
- Comparing regional carbon efficiency of data centers.
- Evaluating the impact of renewable energy integration in different regions.

## **Additional Notes**
- Some files may have missing data in certain time periods due to unavailability.
- Carbon intensity can vary significantly by season and demand levels.
- Data estimation methods vary by region and should be considered when analyzing trends.

## **Data Source**
The carbon intensity data is sourced from **Electricity Maps** ([https://portal.electricitymaps.com/datasets](https://portal.electricitymaps.com/datasets)), a platform that provides historical and real-time electricity grid carbon intensity data for multiple regions worldwide.

## **Contact**
For any questions or data sources, refer to the Electricity Maps API documentation or the original dataset providers.

