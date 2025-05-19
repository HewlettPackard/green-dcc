.. _custom-carbon-intensity-data:

Custom Carbon Intensity Data
============================

.. _app:carbon_details:

Minimizing the environmental impact requires considering the carbon intensity of the grid powering each data center.

- **Source:** We use historical grid carbon intensity data from the Electricity Maps API.
- **Content & Units:** Provides time-series data representing the grams of CO₂ equivalent emitted per kWh of electricity consumed (gCO₂eq/kWh) for different grid regions.
- **Coverage & Resolution:** Covers supported regions from 2021–2024, typically at hourly or finer resolution, aligned to the 15-minute simulation step.
- **Usage:** This data allows the simulation to calculate the carbon emissions associated with both the operational energy (compute + cooling) consumed at each DC and the energy estimated for data transmission. It provides a critical signal for carbon-aware scheduling.
- **Storage:** Stored in yearly CSV files under ``data/carbon_intensity/REGION/YEAR/``.
