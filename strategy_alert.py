"""Alert per la strategia EUR/USD su H1: trend EMA 200 + rientro dell'RSI + stop con ATR.

Viene eseguito UNA volta all'ora da GitHub Actions. Controlla l'ultima candela
CHIUSA e, se c'e' un segnale, manda un messaggio Telegram con stop loss e
take profit gia' calcolati. Non apre nessun ordine.

Regole (ipotesi da verificare con backtest.py):
  LONG : prezzo sopra EMA 200 e RSI risale sopra RSI_LOW (40)
  SHORT: prezzo sotto EMA 200 e RSI scende sotto RSI_HIGH (60)
  Stop = ATR_MULT x ATR(14); take profit = RR x distanza dello stop.
"""
import os
import sys

import pandas as pd
import requests
import yfinance as yf

# --- Configurazione (modificabile con variabili d'ambiente) ---
PAIRS = [p.strip() for p in os.getenv("PAIRS", "EURUSD=X").split(",") if p.strip()]
EMA_LEN = int(os.getenv("EMA_LEN", "200"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
RSI_LOW = float(os.getenv("RSI_LOW", "40"))
RSI_HIGH = float(os.getenv("RSI_HIGH", "60"))
ATR_MULT = float(os.getenv("ATR_MULT", "1.5"))
RR = float(os.getenv("RR", "1.5"))
MAX_AGE_MIN = int(os.getenv("MAX_AGE_MIN", "45"))  # ignora segnali piu' vecchi di cosi'
DRY_RUN = os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")

TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send(text: str) -> None:
    if DRY_RUN or not TOKEN or not CHAT_ID:
        print("[messaggio]\n" + text + "\n")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=15,
        )
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print("Errore Telegram:", e)


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Rende il DataFrame di yfinance uniforme: colonne semplici, indice in UTC."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def rsi(close: pd.Series, n: int) -> pd.Series:
    """RSI con smoothing di Wilder (come TradingView)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """ATR con smoothing di Wilder."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def find_signal(pair: str, df: pd.DataFrame, now: pd.Timestamp):
    """Cerca un segnale sull'ultima candela chiusa. Restituisce un dict o None."""
    # Candela chiusa = inizio + 1 ora <= adesso (esclude quella ancora in formazione)
    closed = df[df.index + pd.Timedelta(hours=1) <= now]
    if len(closed) < EMA_LEN + 5:
        print(f"{pair}: dati insufficienti ({len(closed)} candele)")
        return None

    close = closed["Close"]
    ema = close.ewm(span=EMA_LEN, adjust=False).mean()
    r = rsi(close, RSI_LEN)
    a = atr(closed, ATR_LEN)

    candle_start = closed.index[-1]
    age_min = (now - (candle_start + pd.Timedelta(hours=1))).total_seconds() / 60
    if age_min > MAX_AGE_MIN:
        print(f"{pair}: ultima candela chiusa {age_min:.0f} min fa, nessun segnale nuovo")
        return None

    price = float(close.iloc[-1])
    prev_rsi, cur_rsi = float(r.iloc[-2]), float(r.iloc[-1])
    cur_ema, cur_atr = float(ema.iloc[-1]), float(a.iloc[-1])
    if pd.isna(prev_rsi) or pd.isna(cur_rsi) or pd.isna(cur_atr):
        return None

    side = None
    if price > cur_ema and prev_rsi < RSI_LOW <= cur_rsi:
        side = "LONG"
    elif price < cur_ema and prev_rsi > RSI_HIGH >= cur_rsi:
        side = "SHORT"
    if side is None:
        print(f"{pair}: nessun segnale (RSI {cur_rsi:.1f}, prezzo {'sopra' if price > cur_ema else 'sotto'} EMA)")
        return None

    sign = 1 if side == "LONG" else -1
    dist = ATR_MULT * cur_atr
    pip = pip_size(pair)
    return {
        "side": side,
        "price": price,
        "sl": price - sign * dist,
        "tp": price + sign * RR * dist,
        "sl_pips": dist / pip,
        "tp_pips": RR * dist / pip,
        "rsi": cur_rsi,
        "ema": cur_ema,
        "candle_start": candle_start,
    }


def format_message(pair: str, s: dict) -> str:
    name = pair.replace("=X", "")
    name = name[:3] + "/" + name[3:] if len(name) == 6 else name
    dec = 3 if "JPY" in pair else 5
    if s["side"] == "LONG":
        head = f"📈 SEGNALE LONG {name} H1"
        why = f"RSI({RSI_LEN}) risalito sopra {RSI_LOW:g} (ora {s['rsi']:.1f}), prezzo sopra EMA {EMA_LEN}."
    else:
        head = f"📉 SEGNALE SHORT {name} H1"
        why = f"RSI({RSI_LEN}) sceso sotto {RSI_HIGH:g} (ora {s['rsi']:.1f}), prezzo sotto EMA {EMA_LEN}."
    return (
        f"{head}\n{why}\n\n"
        f"Prezzo di riferimento: {s['price']:.{dec}f}\n"
        f"Stop loss: {s['sl']:.{dec}f} ({s['sl_pips']:.1f} pip)\n"
        f"Take profit: {s['tp']:.{dec}f} ({s['tp_pips']:.1f} pip)\n\n"
        "Ordina all'apertura della prossima candela, tieni conto dello spread, "
        "rischia al massimo l'1% e scrivi il trade nel diario."
    )


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    if os.getenv("SEND_START", "").lower() == "true":
        send(f"✅ Alert strategia H1 attivo. Strumenti: {', '.join(PAIRS)}. "
             "Il controllo gira ogni ora; ti scrivo solo quando c'è un segnale.")

    for pair in PAIRS:
        try:
            df = yf.download(pair, period="90d", interval="1h", progress=False, auto_adjust=True)
            if df is None or df.empty:
                print(f"{pair}: nessun dato")
                continue
            signal = find_signal(pair, prepare(df), now)
            if signal:
                send(format_message(pair, signal))
        except Exception as e:  # noqa: BLE001
            print(f"Errore su {pair}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
