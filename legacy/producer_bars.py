import time, json, ccxt, pandas as pd
from kafka import KafkaProducer

EXCHANGE = ccxt.binance()
PAIRS    = ["BTC/USDT","ETH/USDT"]
TF       = '1m'

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=50
)

def fetch_latest_ohlcv(symbol):
    o = EXCHANGE.fetch_ohlcv(symbol, TF, limit=200)
    for ts, o_, h, l, c, v in o:
        yield {
            "asset": symbol.replace("/",""),
            "ts_utc": pd.to_datetime(ts, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            "open": o_, "high": h, "low": l, "close": c, "volume": v,
            "venue": "binance"
        }

if __name__ == "__main__":
    seen = set()
    while True:
        for sym in PAIRS:
            for row in fetch_latest_ohlcv(sym):
                key = (row["asset"], row["ts_utc"])
                if key in seen:
                    continue
                seen.add(key)
                producer.send("bars.1m.raw", value=row)
        producer.flush()
        time.sleep(30)