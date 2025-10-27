import pandas as pd
import numpy as np
import requests
import joblib
import os
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CLASSIFIER_MODEL_FILE = 'classification_model.joblib' # Primary model for output category
MAPPING_FILE = 'quirky_category_mapping.joblib'      # Mapping for classifier output
REGRESSOR_MODEL_FILE = 'regression_model.joblib'      # Used ONLY for lag updates
HISTORICAL_DATA_FILE = 'final_dataset.csv'          # Your historical data

FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Kolkata"


CITIES_COORDS = {
    "mumbai": {"lat": 19.0760, "lon": 72.8777},
    "delhi": {"lat": 28.6139, "lon": 77.2090},
    "chennai": {"lat": 13.0827, "lon": 80.2707},
    "bangalore": {"lat": 12.9716, "lon": 77.5946}
}
BASELINE_CITY = 'bangalore'
FEATURE_NAMES_CLF = None
FEATURE_NAMES_REG = None
LAGS_TO_CREATE = []
MAX_LAG = 0

def infer_model_details():
    global FEATURE_NAMES_CLF, FEATURE_NAMES_REG, LAGS_TO_CREATE, MAX_LAG
    try:
        if not os.path.exists(CLASSIFIER_MODEL_FILE) or not os.path.exists(REGRESSOR_MODEL_FILE):
             raise FileNotFoundError("Model file(s) not found.")

        temp_clf = joblib.load(CLASSIFIER_MODEL_FILE)
        FEATURE_NAMES_CLF = temp_clf.feature_names_in_.tolist()
        temp_reg = joblib.load(REGRESSOR_MODEL_FILE)
        FEATURE_NAMES_REG = temp_reg.feature_names_in_.tolist()

        LAGS_TO_CREATE = sorted(list(set([
            int(col.split('_')[-1][:-1]) for col in FEATURE_NAMES_CLF
            if col.startswith('pm2_5_lag_') and col.endswith('h')
        ])))

        if not LAGS_TO_CREATE:
            logging.warning("No lag features detected in classifier model names. Assuming none.")
            MAX_LAG = 0
        else:
            MAX_LAG = max(LAGS_TO_CREATE)

        logging.info(f"Successfully inferred model details. Using Classifier features.")
        logging.info(f"Required Features: {len(FEATURE_NAMES_CLF)} features.")
        logging.info(f"Inferred Lags: {LAGS_TO_CREATE}, Max Lag: {MAX_LAG}")
        del temp_clf, temp_reg # Clean up

    except FileNotFoundError as e:
         logging.error(f"Startup Error: {e}. Cannot initialize predictor.")
         FEATURE_NAMES_CLF = None # Indicate failure
         LAGS_TO_CREATE = []
         MAX_LAG = 0
    except AttributeError:
         logging.error("Startup Error: Models loaded but do not have 'feature_names_in_'. Ensure models were saved correctly.")
         FEATURE_NAMES_CLF = None
         LAGS_TO_CREATE = []
         MAX_LAG = 0
    except Exception as e:
        logging.error(f"Unexpected error during model detail inference: {e}")
        FEATURE_NAMES_CLF = None
        LAGS_TO_CREATE = []
        MAX_LAG = 0


def get_health_alert(quirky_category_name):
    if quirky_category_name in ["Crystal Clear Skies ☀️", "Light Haze ☁️"]: return {"level": "Minimal", "advice": "All outdoor activity is fine."}
    elif quirky_category_name in ["Urban Fog 🏙️", "Smog Alert 🏭"]: return {"level": "Caution", "advice": "Sensitive groups limit prolonged outdoor exertion."}
    elif quirky_category_name in ["Pea Soup Air 🍲", "Code Red Atmosphere 🚨"]: return {"level": "High Alert", "advice": "Minimize time outdoors. Use enclosed transport."}
    return {"level": "Unknown", "advice": "Check local advisories."}

def get_commute_advice(quirky_category_name):
    if quirky_category_name in ["Crystal Clear Skies ☀️", "Light Haze ☁️"]: return "All modes recommended."
    elif quirky_category_name == "Urban Fog 🏙️": return "Cycling/walking fine. Sensitive individuals consider bus/Metro."
    elif quirky_category_name == "Smog Alert 🏭": return "Limit open-air travel. Prefer enclosed AC Bus or Metro."
    elif quirky_category_name == "Pea Soup Air 🍲": return "High exposure risk. Metro/AC Bus strongly recommended."
    elif quirky_category_name == "Code Red Atmosphere 🚨": return "Severe exposure risk. Avoid non-essential outdoor travel."
    return "Check local advisories."

class AQIPredictor:
    def __init__(self):
        self.model_classifier = self._load_model(CLASSIFIER_MODEL_FILE)
        self.model_regressor = self._load_model(REGRESSOR_MODEL_FILE)
        self.category_mapping = self._load_mapping(MAPPING_FILE)
        self.inverse_mapping = {v: k for k, v in self.category_mapping.items()} if self.category_mapping else None
        self.historical_data = self._load_historical_data()
        self.feature_names = FEATURE_NAMES_CLF # Use globally inferred names

        self.is_initialized = all([
            self.model_classifier is not None,
            self.model_regressor is not None,
            self.category_mapping is not None,
            self.historical_data is not None, # Explicit check
            self.inverse_mapping is not None, # Check derived mapping too
            self.feature_names is not None
        ])


        if not self.is_initialized:
            logging.error("AQIPredictor initialization failed: One or more components could not be loaded.")
            if self.model_classifier is None: logging.error("-> Classifier model failed to load.")
            if self.model_regressor is None: logging.error("-> Regressor model failed to load.")
            if self.category_mapping is None: logging.error("-> Category mapping failed to load.")
            if self.historical_data is None: logging.error("-> Historical data failed to load.")
            if self.feature_names is None: logging.error("-> Could not determine feature names.")

    def _load_model(self, file_path):
        if os.path.exists(file_path):
            try:
                model = joblib.load(file_path)
                logging.info(f"Successfully loaded model: {file_path}")
                return model
            except Exception as e:
                logging.error(f"Error loading model {file_path}: {e}")
                return None
        else:
            logging.error(f"Model file not found: {file_path}")
            return None

    def _load_mapping(self, file_path):
        if os.path.exists(file_path):
            try:
                mapping = joblib.load(file_path)
                logging.info(f"Successfully loaded mapping: {file_path}, Content: {mapping}")
                if not isinstance(mapping, dict):
                    logging.error(f"Mapping file {file_path} did not contain a dictionary.")
                    return None
                return mapping
            except Exception as e:
                logging.error(f"Error loading mapping {file_path}: {e}")
                return None
        else:
            logging.error(f"Mapping file not found: {file_path}")
            return None

    def _load_historical_data(self):
        if os.path.exists(HISTORICAL_DATA_FILE):
            try:
                df = pd.read_csv(HISTORICAL_DATA_FILE, index_col='time', parse_dates=True, usecols=['time', 'pm2_5', 'city'])
                logging.info(f"Successfully loaded historical data. Shape: {df.shape}")
                return df
            except Exception as e:
                logging.error(f"Error loading historical data {HISTORICAL_DATA_FILE}: {e}")
                return None
        else:
            logging.error(f"Historical data file not found: {HISTORICAL_DATA_FILE}")
            return None

    def get_historical_lags(self, city):
        if self.historical_data is None: return None
        df_city = self.historical_data[self.historical_data['city'] == city]
        if MAX_LAG == 0: return []
        pm25_lags = df_city['pm2_5'].tail(MAX_LAG)
        if len(pm25_lags) < MAX_LAG:
            logging.warning(f"Need {MAX_LAG} hrs history for {city}, got {len(pm25_lags)}. Padding.")
            median_pm25 = df_city['pm2_5'].tail(MAX_LAG * 2).median() if not df_city.empty else 60
            full_lags = ([median_pm25] * (MAX_LAG - len(pm25_lags))) + pm25_lags.tolist()
            return full_lags
        else:
            return pm25_lags.tolist()

    def _engineer_features(self, df_fc, city, pm25_lags_hist):
        df_engineered = df_fc.copy()
        expects_sin_cos = any(col.endswith('_sin') for col in self.feature_names) if self.feature_names else False

        if expects_sin_cos:
            df_engineered['hour'] = df_engineered.index.hour
            df_engineered['day_of_week'] = df_engineered.index.dayofweek
            df_engineered['month'] = df_engineered.index.month
            df_engineered['hour_sin'] = np.sin(2 * np.pi * df_engineered['hour'] / 24.0); df_engineered['hour_cos'] = np.cos(2 * np.pi * df_engineered['hour'] / 24.0)
            df_engineered['day_of_week_sin'] = np.sin(2 * np.pi * df_engineered['day_of_week'] / 7.0); df_engineered['day_of_week_cos'] = np.cos(2 * np.pi * df_engineered['day_of_week'] / 7.0)
            df_engineered['month_sin'] = np.sin(2 * np.pi * (df_engineered['month'] - 1) / 12.0); df_engineered['month_cos'] = np.cos(2 * np.pi * (df_engineered['month'] - 1) / 12.0)
            df_engineered.drop(['hour', 'day_of_week', 'month'], axis=1, inplace=True)
        else:
            df_engineered['hour'] = df_engineered.index.hour
            df_engineered['day_of_week'] = df_engineered.index.dayofweek
            df_engineered['month'] = df_engineered.index.month

        dummy_cities = [c for c in CITIES_COORDS if c != BASELINE_CITY]
        for c in dummy_cities: df_engineered[f'city_{c}'] = 1 if c == city else 0

        if MAX_LAG > 0:
            if len(pm25_lags_hist) != MAX_LAG:
                logging.error(f"Historical lags length mismatch: Expected {MAX_LAG}, got {len(pm25_lags_hist)}")
                return None
            for lag in LAGS_TO_CREATE:
                col_name = f'pm2_5_lag_{lag}h'; initial_lag_values = []
                for i in range(len(df_engineered)):
                    hist_idx = MAX_LAG - lag + i
                    initial_lag_values.append(pm25_lags_hist[hist_idx] if hist_idx < MAX_LAG else np.nan)
                df_engineered[col_name] = initial_lag_values


        if self.feature_names:
            missing_cols = set(self.feature_names) - set(df_engineered.columns)
            for col in missing_cols: df_engineered[col] = 0 # Add missing columns as 0
            try:
                df_engineered = df_engineered[self.feature_names] # Reorder to match model
            except KeyError as e:
                logging.error(f"Feature engineering column mismatch: {e}. Available cols: {df_engineered.columns.tolist()}")
                return None
        else:
            logging.error("Cannot proceed without defined feature names.")
            return None

        return df_engineered

    def predict_24h_forecast(self, city):
        """Fetches forecast, engineers features, runs iterative prediction."""

        if not self.is_initialized:
            error_msg = "Predictor not fully initialized (missing components)."
            logging.error(error_msg); return None, error_msg

        pm25_lags_hist = self.get_historical_lags(city)
        if pm25_lags_hist is None and MAX_LAG > 0: return None, "Failed to retrieve historical lags."
        pm25_lags_hist = pm25_lags_hist if pm25_lags_hist is not None else []
        coords = CITIES_COORDS.get(city)
        if not coords: return None, "Invalid city name."
        forecast_params = {
            "latitude": coords["lat"], "longitude": coords["lon"],
            "hourly": ["temperature_2m", "relativehumidity_2m", "precipitation", "windspeed_10m"],
            "timezone": TIMEZONE, "forecast_days": 2
        }
        try:
            response_fc = requests.get(FORECAST_API_URL, params=forecast_params, timeout=15)
            response_fc.raise_for_status(); data_fc = response_fc.json()
            df_fc_raw = pd.DataFrame(data_fc['hourly']); df_fc_raw['time'] = pd.to_datetime(df_fc_raw['time'])
            df_fc_raw.set_index('time', inplace=True)
            if df_fc_raw.index.tz is not None: df_fc_raw.index = df_fc_raw.index.tz_localize(None)
            last_hist_time = self.historical_data[self.historical_data['city'] == city].index.max()
            df_fc = df_fc_raw[df_fc_raw.index > last_hist_time].head(24)
            if df_fc.empty: return None, "No future forecast data found."
            if len(df_fc) < 24: logging.warning(f"Only got {len(df_fc)} forecast hours.")
        except Exception as e: logging.error(f"Forecast API error: {e}"); return None, "API fetch/process failed."

        try:
            df_features = self._engineer_features(df_fc, city, pm25_lags_hist)
            if df_features is None: raise ValueError("Feature engineering returned None")
        except Exception as e: logging.error(f"Feature engineering failed: {e}"); return None, "Feature engineering process failed."


        predicted_categories_encoded = []
        predicted_pm25_for_lags = []
        recent_pm25 = deque(pm25_lags_hist, maxlen=MAX_LAG) if MAX_LAG > 0 else deque(maxlen=MAX_LAG)

        for i in range(len(df_features)):
            current_index = df_features.index[i]
            X_single_row = df_features.loc[[current_index]].copy()

            if MAX_LAG > 0:
                for lag in LAGS_TO_CREATE:
                    col_name = f'pm2_5_lag_{lag}h'
                    if lag <= len(recent_pm25):
                        X_single_row[col_name] = recent_pm25[-lag]
                    else:
                        X_single_row[col_name] = np.nan # Use NaN for missing lags
                        logging.warning(f"Insufficient history/predictions for lag {lag} at step {i}. Setting NaN.")


            X_input = X_single_row[self.feature_names]
            if X_input.isnull().values.any():
                logging.warning(f"NaNs in features step {i}. Filling with 0.")
                X_input = X_input.fillna(0) # Simple fillna - consider mean/median if issues persist


            pred_category_encoded = self.model_classifier.predict(X_input)[0]
            predicted_categories_encoded.append(pred_category_encoded)


            pred_pm25_for_lag = self.model_regressor.predict(X_input)[0]
            pred_pm25_for_lag = max(0, pred_pm25_for_lag)
            predicted_pm25_for_lags.append(pred_pm25_for_lag)

            if MAX_LAG > 0:
                recent_pm25.append(pred_pm25_for_lag)

        df_fc['predicted_category_encoded'] = pd.Series(predicted_categories_encoded, index=df_features.index)
        df_fc['predicted_category_name'] = df_fc['predicted_category_encoded'].map(self.inverse_mapping).fillna("Unknown")
        df_fc['predicted_pm25_context'] = pd.Series(predicted_pm25_for_lags, index=df_features.index)

        return df_fc, None

app = Flask(__name__)

infer_model_details()
predictor = AQIPredictor()


@app.route('/api/predict_aqi/<city_name>', methods=['GET'])
def predict_aqi_route(city_name):
    if not predictor.is_initialized:
         return jsonify({"error": "Predictor service not ready. Check server logs."}), 503 # Service Unavailable

    city = city_name.lower()
    if city not in CITIES_COORDS:
        return jsonify({"error": "Invalid city name."}), 400

    df_result, error = predictor.predict_24h_forecast(city)

    if error:
        logging.error(f"Prediction failed for {city}: {error}")
        return jsonify({"error": error}), 500
    if df_result is None or 'predicted_category_name' not in df_result.columns:
        return jsonify({"error": "Prediction failed internally (missing category)."}), 500

    hourly_data = []
    df_result_valid = df_result.dropna(subset=['predicted_category_name']) # Check category name exists

    for index, row in df_result_valid.iterrows():
        category_name = row['predicted_category_name']
        alert = get_health_alert(category_name)['level']
        pm25_context = row.get('predicted_pm25_context', np.nan)
        weather_info = {
            "temp_c": round(row.get('temperature_2m', float('nan')), 1),
            "humidity_pct": int(row.get('relativehumidity_2m', float('nan'))),
            "precip_mm": round(row.get('precipitation', float('nan')), 1),
            "wind_kmh": round(row.get('windspeed_10m', float('nan')), 1)
        }
        weather_info = {k: v if not pd.isna(v) else 'N/A' for k, v in weather_info.items()}

        hourly_data.append({
            "time": index.strftime('%Y-%m-%d %H:%M'),
            "pm25_context": round(pm25_context, 2) if not pd.isna(pm25_context) else "N/A",
            "category": category_name,
            "alert_level": alert,
            "weather": weather_info
        })

    commute_recommendations = []
    # Morning (8-10 AM)
    morning_hours = df_result_valid[(df_result_valid.index.hour >= 8) & (df_result_valid.index.hour <= 10)]
    if not morning_hours.empty:
        morning_cat_mode = morning_hours['predicted_category_name'].mode()
        morning_cat = morning_cat_mode[0] if not morning_cat_mode.empty else "Unknown"
        morning_advice = get_commute_advice(morning_cat); morning_alert = get_health_alert(morning_cat)
        avg_morning_pm25_ctx = morning_hours['predicted_pm25_context'].mean()
        commute_recommendations.append({
            "commute_time": "Morning (8-10 AM)",
            "avg_pm25_context": round(avg_morning_pm25_ctx, 2) if not pd.isna(avg_morning_pm25_ctx) else "N/A",
            "category": morning_cat, "risk": morning_alert['level'], "recommendation": morning_advice
        })
    # Evening (5-7 PM)
    evening_hours = df_result_valid[(df_result_valid.index.hour >= 17) & (df_result_valid.index.hour <= 19)]
    if not evening_hours.empty:
        evening_cat_mode = evening_hours['predicted_category_name'].mode()
        evening_cat = evening_cat_mode[0] if not evening_cat_mode.empty else "Unknown"
        evening_advice = get_commute_advice(evening_cat); evening_alert = get_health_alert(evening_cat)
        avg_evening_pm25_ctx = evening_hours['predicted_pm25_context'].mean()
        commute_recommendations.append({
            "commute_time": "Evening (5-7 PM)",
            "avg_pm25_context": round(avg_evening_pm25_ctx, 2) if not pd.isna(avg_evening_pm25_ctx) else "N/A",
            "category": evening_cat, "risk": evening_alert['level'], "recommendation": evening_advice
        })

    return jsonify({
        "city": city_name.title(), "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "hourly_forecast": hourly_data, "commute_recommendations": commute_recommendations
    })


@app.route('/')
def index():
    frontend_file = 'index.html'
    if os.path.exists(frontend_file):
        with open(frontend_file, 'r') as f: return f.read()
    else: return "Frontend file not found.", 404

if __name__ == '__main__':
    predictor = AQIPredictor()
    app.run(debug=True)