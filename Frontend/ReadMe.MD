Frontend & API - AQI Commute Advisor

This section contains the Flask web application that serves the user interface and provides real-time AQI predictions and commute recommendations.

Goal

To provide a simple web interface where users can select a city and view:

The predicted AQI (numerical PM2.5 and quirky category) for the next 24 hours.

Associated health alerts.

Specific commute mode recommendations for morning and evening rush hours.

Basic weather forecast information for each hour.

Technologies Used

Backend: Flask (Python web framework)

Frontend: HTML, Tailwind CSS (for styling), Vanilla JavaScript (for API calls and DOM manipulation)

Prediction: Loads pre-trained models (.joblib files) using joblib.
