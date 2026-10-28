# Smart Energy Microgrid - Demand Forecasting Engine (HackSphere 3.0)

Forecasts hourly electricity demand (kW) for the next 24 hours from weather data, decides Main Grid vs
Renewable Storage for each hour, simulates battery charge, and raises Critical Overload alerts.
Shown on a live dashboard hosted on Zoho Catalyst AppSail.

**Live app:** https://energy-forecast-50047130408.development.catalystappsail.in
(health check: `/api/health`, prediction API: `POST /api/predict`)

## Important note about the model source (honest disclosure)
* We trained and **published** a Gradient Boosting Regression model in Zoho Catalyst QuickML
  (pipeline `energy_demand_forecast`, endpoint `energy_forecast_endpoint`). On our local benchmark it gave the best RMSE
  (baseline 72.81, Linear Regression 41.04, Random Forest 13.55, **Gradient Boosting 11.88**).
* The QuickML free tier allows only 500 prediction API calls. **Our quota was used up during testing**
  (HTTP 403 `PREDICTION_USAGE_LIMIT_EXCEEDED`), so the QuickML endpoint cannot serve new predictions right now.
* Therefore predictions are served by `model.json`: the **same Gradient Boosting algorithm trained on the same training data**,
  stored locally. The dashboard badge and the system log always show which engine produced the numbers
  ("Local backup model" or "QuickML endpoint"). We never label local results as QuickML.
* If QuickML credentials are set as environment variables (see below) and quota is available, the code uses QuickML first
  automatically.

## Architecture
```
Browser dashboard (static/index.html, Chart.js)
        |  CSV upload / manual form / demo day
        v
app.py (AppSail web server, standard library only)
        |--> controller.py : cleaning, switching rules, battery simulation
        |--> predict.py    : forecasting engine
                 1. Direct QuickML call (only if env vars are set)
                 2. Public app endpoint QUICKML_ENDPOINT_URL (/api/predict)
                 3. Local model.json (Gradient Boosting, same data)
                 4. Starter-kit formula (so nothing ever crashes)
```
* **Features (7):** ambient_temp_c, humidity_pct, solar_irradiance_wm2, hour, day_of_week, is_weekend, is_evening_peak
  (built identically in training and inference).
* **Switching rules:** Renewable Storage Mode if demand > 160 kW OR hour 18-22 inclusive; otherwise Main Grid Mode.
  Critical Overload Alert if demand >= 200 kW (independent of the mode).
* **Battery assumptions (ours, the brief gives none):** 400 kWh capacity, starts at 80%, covers 40% of demand in
  Storage Mode (never below 10%), recharges 60 kWh per Main Grid hour.
* **Data cleaning:** accepts several timestamp formats (DD-MM-YYYY HH:MM and YYYY-MM-DD HH:MM among others), fills
  missing/invalid numbers with the column median, clips humidity to 0-100 and irradiance to >= 0, skips unreadable rows,
  and reports every fix on the dashboard and in the system log.
* **Dashboard:** predicted load curve with 160/200 kW lines, grid state badge, battery state of charge, overload alert
  banner, hourly table, system log.

## API
* `GET /api/health` -> `{"status": "ok"}`
* `GET /api/demo` -> demo CSV
* `POST /api/forecast` with `{"csv": "..."}` or `{"rows": [...]}` -> full forecast, switching, battery, alerts, log
* `POST /api/predict` with `{"data": [rows]}` -> `{"predictions": [...], "source": "..."}`

## Run locally
```
python app.py          # open http://localhost:5000
python test_eval.py    # prints SUCCESS
python predict.py --input sample_test_input.csv --output predictions.csv
```
The app needs no packages. `pandas` and `numpy` are needed only for `predict()` / `test_eval.py`;
`scikit-learn` only for `train_model.py`.

## Environment variables (all OPTIONAL, used only for the direct QuickML call; never commit real values)
| Name | Meaning |
|---|---|
| QUICKML_API_URL | Endpoint URL from QuickML > Endpoints |
| QUICKML_ENDPOINT_KEY | Endpoint key from the same page |
| CATALYST_ORG | Catalyst organisation id |
| QUICKML_ENVIRONMENT | Development or Production (default Development) |
| ZOHO_CLIENT_ID / ZOHO_CLIENT_SECRET / ZOHO_REFRESH_TOKEN | OAuth Self Client credentials (`python get_refresh_token.py` creates the refresh token) |
| ZOHO_ACCOUNTS_URL | https://accounts.zoho.in (or .com to match your Catalyst console) |
| QUICKML_ENDPOINT_URL | Public app prediction URL used by `predict.py` (default is set inside the file) |
| X_ZOHO_CATALYST_LISTEN_PORT | Port set by AppSail (default 5000) |

The deployed app has no secrets configured; it runs on the local model. No credentials are stored in this repository.

## Deploy on Catalyst AppSail
1. Install Node.js LTS, then `npm install -g zcatalyst-cli`.
2. `catalyst login` (choose the data centre of your console, we used India).
3. `catalyst init` and link the project; `catalyst appsail:add` with Python 3.12, build path `./`.
4. In `app-config.json` the start command is `python3 -u app.py`.
5. `catalyst deploy`. The AppSail URL is shown in the console under Serverless > AppSail.

## predict.py for judges
`python predict.py --input sample_test_input.csv --output predictions.csv` writes `timestamp,target_demand_kw`
with the original timestamps. `predict(input_df)` returns one number per row. It works offline using `model.json`,
and also works through the public app URL set in `QUICKML_ENDPOINT_URL`.

## Files
`app.py` server, `controller.py` rules and battery, `predict.py` forecasting, `train_model.py` training and benchmark,
`model.json` exported Gradient Boosting model, `static/index.html` dashboard, `demo_day.csv` demo input,
`train_telemetry.csv` training data, `sample_test_input.csv` sample test input, `test_eval.py` self-check.
