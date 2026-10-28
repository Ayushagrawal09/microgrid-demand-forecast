"""Data cleaning, grid switching rules and battery simulation (pure Python)."""
import csv
import io
import statistics

from predict import parse_timestamp, forecast_rows

REQUIRED = ["timestamp", "ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]
NUMERIC = REQUIRED[1:]
DEFAULTS = {"ambient_temp_c": 25.0, "humidity_pct": 60.0, "solar_irradiance_wm2": 0.0}
MAX_ROWS = 168

# Rules from the problem statement
STORAGE_THRESHOLD_KW = 160
OVERLOAD_KW = 200
PEAK_START, PEAK_END = 18, 22  # inclusive

# Battery assumptions (our own - the problem statement gives none)
CAPACITY_KWH = 400.0
START_SOC_PCT = 80.0
MIN_SOC_PCT = 10.0
DISCHARGE_SHARE = 0.40   # battery covers 40% of demand during Storage Mode
CHARGE_KWH_PER_HOUR = 60.0  # recharge rate during Main Grid Mode


def parse_csv_text(text):
    reader = csv.DictReader(io.StringIO(text.strip()))
    rows = [dict(r) for r in reader]
    if reader.fieldnames is None:
        raise ValueError("The CSV is empty.")
    names = [n.strip() for n in reader.fieldnames]
    missing = [c for c in REQUIRED if c not in names]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing))
    return [{k.strip(): v for k, v in r.items()} for r in rows]


def clean_rows(rows):
    """Validate and repair rows. Returns (clean_rows, warnings)."""
    warnings = []
    if not rows:
        raise ValueError("No data rows found.")
    if len(rows) > MAX_ROWS:
        warnings.append(f"Only the first {MAX_ROWS} rows were used.")
        rows = rows[:MAX_ROWS]

    medians = {}
    for col in NUMERIC:
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col)))
            except (TypeError, ValueError):
                pass
        medians[col] = statistics.median(vals) if vals else DEFAULTS[col]

    cleaned, filled, dropped = [], 0, 0
    for r in rows:
        try:
            ts = parse_timestamp(r.get("timestamp"))
        except ValueError:
            dropped += 1
            continue
        out = {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")}
        for col in NUMERIC:
            try:
                val = float(r.get(col))
                if val != val:  # NaN
                    raise ValueError
            except (TypeError, ValueError):
                val = medians[col]
                filled += 1
            out[col] = val
        out["humidity_pct"] = min(100.0, max(0.0, out["humidity_pct"]))
        out["solar_irradiance_wm2"] = max(0.0, out["solar_irradiance_wm2"])
        cleaned.append(out)

    if dropped:
        warnings.append(f"{dropped} row(s) skipped because the timestamp could not be read.")
    if filled:
        warnings.append(f"{filled} missing/invalid value(s) were filled with the column median.")
    if not cleaned:
        raise ValueError("No valid rows after cleaning.")
    cleaned.sort(key=lambda r: r["timestamp"])
    return cleaned, warnings


def grid_mode(demand_kw, hour):
    if demand_kw > STORAGE_THRESHOLD_KW or PEAK_START <= hour <= PEAK_END:
        return "Renewable Storage Mode"
    return "Main Grid Mode"


def is_overload(demand_kw):
    return demand_kw >= OVERLOAD_KW


def prepare_rows_keep_order(rows, max_rows=2000):
    """For /api/predict: fill gaps but never drop, sort or reorder (one output per input row)."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("'data' must be a non-empty list of rows.")
    if len(rows) > max_rows:
        raise ValueError(f"Send at most {max_rows} rows per request.")
    medians = {}
    for col in NUMERIC:
        vals = []
        for r in rows:
            try:
                v = float(r.get(col))
                if v == v:
                    vals.append(v)
            except (TypeError, ValueError):
                pass
        medians[col] = statistics.median(vals) if vals else DEFAULTS[col]
    out = []
    for i, r in enumerate(rows):
        try:
            parse_timestamp(r.get("timestamp"))
        except ValueError:
            raise ValueError(f"Row {i}: unreadable timestamp")
        row = {"timestamp": r["timestamp"]}
        for col in NUMERIC:
            try:
                v = float(r.get(col))
                if v != v:
                    raise ValueError
            except (TypeError, ValueError):
                v = medians[col]
            row[col] = v
        row["humidity_pct"] = min(100.0, max(0.0, row["humidity_pct"]))
        row["solar_irradiance_wm2"] = max(0.0, row["solar_irradiance_wm2"])
        out.append(row)
    return out


def run_forecast(rows):
    clean, warnings = clean_rows(rows)
    demands, source = forecast_rows(clean, allow_proxy=False)

    soc_kwh = CAPACITY_KWH * START_SOC_PCT / 100.0
    min_kwh = CAPACITY_KWH * MIN_SOC_PCT / 100.0
    hours, battery_used = [], 0.0

    for r, demand in zip(clean, demands):
        hour = int(r["timestamp"][11:13])
        mode = grid_mode(demand, hour)
        if mode == "Renewable Storage Mode":
            used = min(DISCHARGE_SHARE * demand, max(0.0, soc_kwh - min_kwh))
            soc_kwh -= used
            battery_used += used
        else:
            soc_kwh = min(CAPACITY_KWH, soc_kwh + CHARGE_KWH_PER_HOUR)
        soc_pct = soc_kwh / CAPACITY_KWH * 100.0
        hours.append({
            "timestamp": r["timestamp"], "hour": hour,
            "temp": round(r["ambient_temp_c"], 1),
            "demand_kw": round(demand, 1),
            "mode": mode,
            "overload": is_overload(demand),
            "soc_pct": round(soc_pct, 1),
            "battery_low": soc_pct <= MIN_SOC_PCT + 0.05,
        })

    summary = {
        "hours": len(hours),
        "peak_kw": max(h["demand_kw"] for h in hours),
        "avg_kw": round(sum(h["demand_kw"] for h in hours) / len(hours), 1),
        "storage_hours": sum(h["mode"] == "Renewable Storage Mode" for h in hours),
        "grid_hours": sum(h["mode"] == "Main Grid Mode" for h in hours),
        "overload_hours": sum(h["overload"] for h in hours),
        "final_soc_pct": hours[-1]["soc_pct"],
        "battery_energy_kwh": round(battery_used, 1),
    }
    return {
        "source": source,
        "warnings": warnings,
        "summary": summary,
        "hours": hours,
        "rules": {"storage_threshold_kw": STORAGE_THRESHOLD_KW, "overload_kw": OVERLOAD_KW,
                  "peak_window": [PEAK_START, PEAK_END],
                  "battery": {"capacity_kwh": CAPACITY_KWH, "start_soc_pct": START_SOC_PCT,
                              "min_soc_pct": MIN_SOC_PCT}},
    }
