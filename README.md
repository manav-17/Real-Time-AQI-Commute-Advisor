AQI Commute Advisor: 

Project Goal

To predict the hourly Air Quality Index (AQI) based on PM2.5 concentration for the next 24 hours in major Indian metro cities (Mumbai, Delhi, Chennai, Bangalore) and provide actionable commute mode recommendations to minimize users' exposure to air pollution during typical commute times.

Features

Multi-City Support: Provides forecasts for Mumbai, Delhi, Chennai, and Bangalore.

24-Hour Hourly Forecast: Displays predicted PM2.5 (contextual value), a user-friendly AQI category ("quirky names"), health alert level, and basic weather information for each of the next 24 hours.

Commute Recommendations: Offers specific advice for morning (8-10 AM) and evening (5-7 PM) commutes based on the predicted average AQI during those periods.

Machine Learning Powered: Utilizes RandomForest models trained on historical weather and air quality data.

Regressor: Predicts the numerical PM2.5 value.

Classifier: Predicts the AQI category directly.

Web Interface: Simple, responsive frontend built with Flask, HTML, Tailwind CSS, and JavaScript.

Technologies Used

Python: Core programming language.

Pandas & NumPy: Data manipulation and numerical operations.

Scikit-learn: Machine learning framework (RandomForestRegressor, RandomForestClassifier, LabelEncoder).

Joblib: Saving and loading trained models.

Requests: Fetching data from external APIs.

Flask: Backend web framework for serving the API and frontend.

Open-Meteo APIs: Data source for historical/forecast air quality and weather.

HTML: Frontend structure.

Tailwind CSS: Frontend styling.

JavaScript: Frontend logic and API calls.

Potential Future Work

Train models for additional pollutants (PM10, O3, NO2).

Incorporate feature importance analysis into the frontend.

Add user accounts or location saving.

Deploy the application to a cloud platform (e.g., Heroku, AWS, Google Cloud).

Improve error handling and logging.

Perform more rigorous hyperparameter tuning if needed.

Explore more advanced time-series models (LSTM, Prophet)

Output: 
<img width="2878" height="1626" alt="image" src="https://github.com/user-attachments/assets/e536c3e6-8edb-450c-8c3e-f28e413a6e58" />
<img width="2868" height="1250" alt="image" src="https://github.com/user-attachments/assets/b424778d-74eb-4813-b4ae-2afe14e4cd47" />

