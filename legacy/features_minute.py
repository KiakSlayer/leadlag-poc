import json, pandas as pd
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer(
    "bars.1m.raw",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True
)
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

frames = {}

def upsert_bar(row):
    asset = row["asset"]
    df = frames.get(asset)
    r = pd.DataFrame([row])
    r["ts_utc"] = pd.to_datetime(r["ts_utc"])
    if df is None:
        frames[asset] = r
    else:
        frames[asset] = (pd.concat([df, r], ignore_index=True)
                           .drop_duplicates(subset=["ts_utc"])
                           .sort_values("ts_utc")
                           .tail(2000))

def try_emit_pair_features(leader="BTCUSDT", lagger="ETHUSDT"):
    if leader not in frames or lagger not in frames:
        return
    L = frames[leader][["ts_utc","close"]].rename(columns={"close":"close_L"})
    G = frames[lagger][["ts_utc","close"]].rename(columns={"close":"close_G"})
    M = pd.merge_asof(L.sort_values("ts_utc"), G.sort_values("ts_utc"),
                      on="ts_utc", direction="backward", tolerance=pd.Timedelta("2min"))
    if len(M) < 5:
        return
    M["r_L"] = M["close_L"].pct_change()
    M["r_G"] = M["close_G"].pct_change()
    last = M.iloc[-1]
    out = {
        "ts_utc": last["ts_utc"].strftime("%Y-%m-%d %H:%M:%S"),
        "pair": f"{leader}->{lagger}",
        "r_leader": None if pd.isna(last["r_L"]) else float(last["r_L"]),
        "r_lagger": None if pd.isna(last["r_G"]) else float(last["r_G"])
    }
    producer.send("features.1m", value=out)

if __name__ == "__main__":
    for msg in consumer:
        upsert_bar(msg.value)
        try_emit_pair_features()