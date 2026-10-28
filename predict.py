"""
predict.py - forecasting for HackSphere 3.0

How it works
  * Anyone (e.g. the judges) running this file only needs QUICKML_ENDPOINT_URL below.
    It is the public URL of our deployed Catalyst app (/api/predict). That app calls
    the published QuickML model with the secret credentials stored on the SERVER,
    so no secrets live in this file.
  * On the server itself, the credentials come from environment variables (see README)
    and QuickML is called directly.

Order of attempts for every call:
  1. Direct QuickML call (server only - needs the secret environment variables)
  2. Our public app endpoint QUICKML_ENDPOINT_URL (which calls QuickML for us)
  3. Local backup model (model.json, same algorithm, same training data)
  4. The simple formula from the starter kit (so the script never crashes)

Environment variables used by the server for the direct QuickML call:
  QUICKML_API_URL        endpoint URL from QuickML > Endpoints
  QUICKML_ENDPOINT_KEY   endpoint key from the same page
  CATALYST_ORG           your Catalyst org id
  QUICKML_ENVIRONMENT    Development or Production (default Development)
  ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN
  ZOHO_ACCOUNTS_URL      default https://accounts.zoho.in  (use .com if your console is .com)
"""
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

# >>> After deploying, paste your public app address here, for example:
# "https://yourapp-1234.development.catalystappsail.in/api/predict"
QUICKML_ENDPOINT_URL = "https://energy-forecast-50047130408.development.catalystappsail.in/api/predict"
PLACEHOLDER = "YOUR_PUBLISHED_QUICKML_ENDPOINT_URL_HERE"

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2",
            "hour", "day_of_week", "is_weekend", "is_evening_peak"]
TS_FORMATS = ["%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
              "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"]

last_source = "unknown"  # which engine produced the latest predictions (single-user use)


def parse_timestamp(value):
    text = str(value).strip()
    for fmt in TS_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised timestamp: {value!r}")


def build_features(row):
    """row has timestamp + 3 weather values -> dict with the 7 model features."""
    ts = parse_timestamp(row["timestamp"])
    return {
        "ambient_temp_c": float(row["ambient_temp_c"]),
        "humidity_pct": float(row["humidity_pct"]),
        "solar_irradiance_wm2": float(row["solar_irradiance_wm2"]),
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "is_weekend": 1 if ts.weekday() >= 5 else 0,
        "is_evening_peak": 1 if 18 <= ts.hour <= 22 else 0,
    }


# ---------------------------------------------------------------- QuickML
_token = {"value": None, "expires": 0.0}
_quickml_pause = {"until": 0.0}  # after a QuickML failure (e.g. free quota used up) skip it for a while


def _quickml_configured():
    return all(os.environ.get(k) for k in
               ("QUICKML_API_URL", "QUICKML_ENDPOINT_KEY", "CATALYST_ORG",
                "ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN"))


def _access_token():
    if _token["value"] and time.time() < _token["expires"]:
        return _token["value"]
    accounts = os.environ.get("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.in").rstrip("/")
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": os.environ["ZOHO_CLIENT_ID"],
        "client_secret": os.environ["ZOHO_CLIENT_SECRET"],
        "refresh_token": os.environ["ZOHO_REFRESH_TOKEN"],
    }).encode()
    req = urllib.request.Request(accounts + "/oauth/v2/token", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode())
    if "access_token" not in body:
        raise RuntimeError(f"Token refresh failed: {body}")
    _token["value"] = body["access_token"]
    _token["expires"] = time.time() + 50 * 60
    return _token["value"]


def _find_number(obj):
    """Dig a single prediction number out of whatever shape the endpoint returns."""
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return None
    if isinstance(obj, dict):
        for key in ("target_demand_kw", "prediction", "predictions", "result", "output", "data"):
            if key in obj:
                found = _find_number(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = _find_number(value)
            if found is not None:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = _find_number(item)
            if found is not None:
                return found
    return None


def _quickml_one(features):
    body = json.dumps({"data": features}).encode()
    headers = {
        "Content-Type": "application/json",
        "X-QUICKML-ENDPOINT-KEY": os.environ["QUICKML_ENDPOINT_KEY"],
        "Authorization": "Zoho-oauthtoken " + _access_token(),
        "CATALYST-ORG": os.environ["CATALYST_ORG"],
        "Environment": os.environ.get("QUICKML_ENVIRONMENT", "Development"),
    }
    req = urllib.request.Request(os.environ["QUICKML_API_URL"], data=body,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as err:
        detail = err.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {err.code}: {detail}")
    number = _find_number(payload)
    if number is None or math.isnan(number):
        raise RuntimeError(f"Could not read a prediction from: {payload}")
    return number


# ------------------------------------------------------------- local model
_model = None


def _load_model():
    global _model
    if _model is None:
        with open(os.path.join(HERE, "model.json")) as fh:
            _model = json.load(fh)
    return _model


def _local_one(features):
    model = _load_model()
    x = [features[name] for name in model["features"]]
    total = model["init"]
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            if x[tree["feature"][node]] <= tree["threshold"][node]:
                node = tree["left"][node]
            else:
                node = tree["right"][node]
        total += model["learning_rate"] * tree["value"][node]
    return total


def _formula_one(row):
    ts = parse_timestamp(row["timestamp"])
    base = 120.0 + 30.0 * math.sin(2 * math.pi * (ts.hour - 6) / 24)
    return base + max(0.0, float(row["ambient_temp_c"]) - 24.0) * 8.5


# ---------------------------------------------------------------- app endpoint
def _proxy_predict(rows, url):
    """Starter-kit style call: POST {"data": [...]} -> {"predictions": [...]}. Sent in chunks."""
    preds, source = [], "QuickML endpoint"
    for i in range(0, len(rows), 100):
        chunk = rows[i:i + 100]
        body = json.dumps({"data": chunk}, default=str).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        got = [float(x) for x in result["predictions"]]
        if len(got) != len(chunk):
            raise RuntimeError("Endpoint returned the wrong number of predictions")
        preds.extend(got)
        source = result.get("source", source)
    return preds, source + " (via app API)"


# ---------------------------------------------------------------- main API
def forecast_rows(rows, allow_proxy=True):
    """rows: list of dicts (timestamp + 3 weather values) -> (list of kW floats, source name)."""
    feats = [build_features(r) for r in rows]

    if _quickml_configured() and time.time() >= _quickml_pause["until"]:
        try:
            _access_token()  # fetch the login token once before using threads
            first = _quickml_one(feats[0])  # test ONE row first: a failure must not cost one call per row
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as pool:
                rest = list(pool.map(_quickml_one, feats[1:]))
            return [first] + rest, "QuickML endpoint"
        except Exception as exc:
            _quickml_pause["until"] = time.time() + 600  # skip QuickML for 10 minutes
            print(f"Warning: direct QuickML call failed ({exc}). Using the next option.",
                  file=sys.stderr)

    url = os.environ.get("QUICKML_ENDPOINT_URL", QUICKML_ENDPOINT_URL)
    if allow_proxy and url and url != PLACEHOLDER:
        try:
            return _proxy_predict(rows, url)
        except Exception as exc:
            print(f"Warning: QuickML endpoint call failed ({exc}). Using backup model.",
                  file=sys.stderr)

    try:
        return [_local_one(f) for f in feats], "Local backup model"
    except Exception as exc:
        print(f"Warning: backup model failed ({exc}). Using starter formula.", file=sys.stderr)
    return [_formula_one(r) for r in rows], "Starter formula"


def predict_rows(rows, allow_proxy=True):
    global last_source
    preds, last_source = forecast_rows(rows, allow_proxy)
    return preds


def predict(input_df):
    """Required contract: DataFrame in -> numpy array of predicted kW out (one per row)."""
    import numpy as np
    rows = input_df[["timestamp", "ambient_temp_c", "humidity_pct",
                     "solar_irradiance_wm2"]].to_dict(orient="records")
    return np.array(predict_rows(rows), dtype=float)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_test_input.csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    preds = predict(df)
    out = pd.DataFrame({"timestamp": df["timestamp"], "target_demand_kw": preds.round(2)})
    out.to_csv(args.output, index=False)
    print(f"Source: {last_source}. Wrote {len(out)} predictions to {args.output}")
    print(out.head())
