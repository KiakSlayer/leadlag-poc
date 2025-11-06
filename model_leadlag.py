# model_leadlag.py (fixed & hardened)
import os, json, time, math
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

# --- Tunables (env overrides help during testing) ---
WINDOW  = int(os.getenv("LL_WINDOW", 60))      # e.g. set LL_WINDOW=20 for quick bring-up
Z_ENTRY = float(os.getenv("LL_Z_ENTRY", 2.0))
Z_EXIT  = float(os.getenv("LL_Z_EXIT", 0.5))
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

# --- Kafka clients ---
consumer = KafkaConsumer(
    "features.1m",
    bootstrap_servers=[BOOTSTRAP],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",     # easier to fill the window when developing
    enable_auto_commit=True,
    group_id="leadlag-model-v2",      # stable offsets across restarts
)
producer = KafkaProducer(
    bootstrap_servers=[BOOTSTRAP],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# --- Rolling buffer (latest WINDOW+5 rows) ---
buf = deque(maxlen=WINDOW + 5)

def nan_beta(rL, rG):
    """OLS beta = cov(y,x)/var(x), NaN-safe."""
    x = np.asarray(rG, dtype=float)
    y = np.asarray(rL, dtype=float)
    # mask NaNs pairwise so we use only aligned valid points
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return np.nan
    mx = x.mean(); my = y.mean()
    vx = ((x - mx) ** 2).mean()
    if not np.isfinite(vx) or vx == 0.0:
        return np.nan
    cov = ((x - mx) * (y - my)).mean()
    return cov / vx

def compute_z(spread_series):
    """Return mean, std, z for the last point of series; NaN-safe with guards."""
    s = np.asarray(spread_series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return np.nan, np.nan, np.nan
    mu = s.mean()
    sd = s.std(ddof=0)
    if not (np.isfinite(mu) and np.isfinite(sd)) or sd == 0.0:
        return mu, sd, np.nan
    z = (s[-1] - mu) / sd
    return mu, sd, z

def write_parquet_daily(out_row, out_dir="phase1_output"):
    """Write one parquet per day; keep it simple & crash-safe for Phase-1."""
    # Using pandas + fastparquet/pyarrow via pandas engine selection
    try:
        d = out_row["ts_utc"][:10]  # 'YYYY-MM-DD'
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fn = out_path / f"signals_1m_{d}.parquet"
        df = pd.DataFrame([out_row])
        if fn.exists():
            # simple read-append-rewrite (fine for small Phase-1 files)
            old = pd.read_parquet(fn)
            pd.concat([old, df], ignore_index=True).to_parquet(fn, index=False)
        else:
            df.to_parquet(fn, index=False)
    except Exception as e:
        # Don't let storage issues kill the stream
        print(f"[model] parquet write error: {e}")

def maybe_log(status_every_s=10):
    """Light heartbeat so you know it's alive."""
    now = time.time()
    if not hasattr(maybe_log, "_t"):
        maybe_log._t = 0.0
    if now - maybe_log._t >= status_every_s:
        maybe_log._t = now
        print(f"[model] WINDOW={WINDOW} Z_ENTRY={Z_ENTRY} Z_EXIT={Z_EXIT} | buf={len(buf)}")

if __name__ == "__main__":
    print("[model] lead-lag model starting…")
    for msg in consumer:
        d = msg.value
        # Guard: skip rows until both returns exist
        if d.get("r_leader") is None or d.get("r_lagger") is None:
            maybe_log()
            continue

        buf.append(d)
        maybe_log()

        if len(buf) < WINDOW:
            # not enough data for a stable window yet
            continue

        DF = pd.DataFrame(list(buf)[-WINDOW:])
        rL = DF["r_leader"].to_numpy(dtype=float)
        rG = DF["r_lagger"].to_numpy(dtype=float)

        beta = nan_beta(rL, rG)
        if np.isfinite(beta):
            spread_now = rL[-1] - beta * rG[-1]
            # rolling spread over window
            spr_series = rL - beta * rG
            mu, sd, z = compute_z(spr_series)
        else:
            spread_now, mu, sd, z = (np.nan, np.nan, np.nan, np.nan)

        # naive signal
        signal = "HOLD"
        if np.isfinite(z):
            if z > Z_ENTRY:
                signal = "SHORT_leader_LONG_lagger"
            elif z < -Z_ENTRY:
                signal = "LONG_leader_SHORT_lagger"
            elif abs(z) < Z_EXIT:
                signal = "FLAT"

        out = {
            "ts_utc": d["ts_utc"],
            "pair": d["pair"],
            "beta": None if not np.isfinite(beta) else float(beta),
            "spread": None if not np.isfinite(spread_now) else float(spread_now),
            "z": None if not np.isfinite(z) else float(z),
            "signal": signal
        }

        # publish
        producer.send("signals.1m", value=out)

        # persist (best-effort)
        write_parquet_daily(out)
