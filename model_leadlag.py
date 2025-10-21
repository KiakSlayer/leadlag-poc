import json, numpy as np, pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from pathlib import Path

WINDOW = 60
Z_ENTRY = 2.0
Z_EXIT  = 0.5

consumer = KafkaConsumer(
    "features.1m",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True
)
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

buf = []
parquet_dir = Path("phase1_output")
parquet_dir.mkdir(exist_ok=True)

def rolling_beta(rL, rG):
    x = np.asarray(rG); y = np.asarray(rL)
    if len(x) < 2 or np.nanvar(x) == 0:
        return np.nan
    return np.nancov(y, x)[0,1] / np.nanvar(x)

if __name__ == "__main__":
    for msg in consumer:
        d = msg.value
        if d["r_leader"] is None or d["r_lagger"] is None:
            continue
        buf.append(d)
        buf = buf[-(WINDOW+5):]

        DF = pd.DataFrame(buf)
        if len(DF) < WINDOW:
            continue

        rL = DF["r_leader"].tail(WINDOW).to_numpy()
        rG = DF["r_lagger"].tail(WINDOW).to_numpy()

        beta = rolling_beta(rL, rG)
        spread = rL[-1] - beta * rG[-1] if np.isfinite(beta) else np.nan

        spr_series = DF["r_leader"].tail(WINDOW) - beta * DF["r_lagger"].tail(WINDOW)
        mu, sd = np.nanmean(spr_series), np.nanstd(spr_series)
        z = (spread - mu) / sd if (sd and np.isfinite(sd)) else np.nan

        signal = "HOLD"
        if np.isfinite(z):
            if z > Z_ENTRY:         signal = "SHORT_leader_LONG_lagger"
            elif z < -Z_ENTRY:      signal = "LONG_leader_SHORT_lagger"
            elif abs(z) < Z_EXIT:   signal = "FLAT"

        out = {
            "ts_utc": d["ts_utc"],
            "pair": d["pair"],
            "beta": None if not np.isfinite(beta) else float(beta),
            "spread": None if not np.isfinite(spread) else float(spread),
            "z": None if not np.isfinite(z) else float(z),
            "signal": signal
        }
        producer.send("signals.1m", value=out)

        pd.DataFrame([out]).to_parquet(
            parquet_dir / "signals_1m.parquet",
            engine="fastparquet",
            append=True
        )