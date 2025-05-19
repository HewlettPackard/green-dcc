.. _custom-weather-data:

Custom Weather Data
===================

.. _app:weather_details:

Ambient weather conditions significantly impact data center cooling efficiency.

- **Source:** Historical weather data is obtained via the Open-Meteo API.
- **Content:** Primarily uses ambient air temperature (°C) as input, though other parameters like wet-bulb temperature could be incorporated for more advanced cooling models.
- **Coverage & Resolution:** Covers supported DC locations from 2021–2024, aligned to the 15-minute simulation step.
- **Usage:** The ambient temperature directly influences the performance (Coefficient of Performance – COP) and energy consumption of the simulated HVAC system, particularly the chiller model.
- **Storage:** Stored in yearly JSON files under ``data/weather/REGION/YEAR/``.
