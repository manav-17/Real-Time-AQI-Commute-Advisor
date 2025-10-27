Data Collection - AQI Commute Advisor

This section handles the collection, cleaning, and merging of historical air quality (PM2.5) and weather data for multiple Indian metro cities.

Goal

To create a single, clean dataset containing hourly PM2.5 concentrations and relevant weather variables (temperature, humidity, precipitation, wind speed) for Mumbai, Delhi, Chennai, and Bangalore, spanning from 2022-01-01 to the present (up to yesterday).

Data Sources

Air Quality (PM2.5): Open-Meteo Air Quality API (using CAMS global reanalysis data).

Weather: Open-Meteo Historical Weather API (using ERA5 reanalysis data).
