"""
Telegram bot that scans Binance USDT-margined futures (perpetuals) — filtered
down to only the coins that are also listed as futures on Toobit — on the
5-minute timeframe, and alerts you when a Double Top or Double Bottom pattern
appears to be forming, using a very tight tolerance between the two peaks/troughs.

Setup:
1. pip install -r requirements.txt
2. Set environment variables:
   TELEGRAM_BOT_TOKEN=xxxx   (from @BotFather)
   TELEGRAM_CHAT_ID=xxxx     (your chat id, see README)
3. python main.py

Commands (send to the bot in Telegram):
   /start                - show help
   /mode all             - watch every Binance USDT future also listed on Toobit (default)
   /mode custom          - watch only symbols you add manually
   /add BTC/USDT:USDT    - add a symbol (custom mode only)
   /remove BTC/USDT:USDT - remove a symbol (custom mode only)
   /list                 - show current settings
   /settimeframe 5m      - change timeframe
   /settolerance 0.3     - max % difference allowed between the two peaks/troughs
   /refresh              - reload the futures symbol list
   /scan                 - run a scan immediately
"""

import os
import json
import asyncio
import logging
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema

import ccxt  # sync client — avoids needing aiohttp/multidict/yarl (hard to build on Termux)

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

STATE_FILE = Path(__file__).parent / "state.json"

DEFAULT_STATE = {
    "mode": "all",              # "all" = every Binance USDT future also on Toobit, "custom" = manual list
    "watchlist": [],            # used only when mode == "custom"
    "timeframe": "5m",
    "tolerance_pct": 0.3,       # max % difference between the two peaks/troughs
}

CANDLE_LIMIT = 150
PEAK_ORDER = 3                  # candles on each side to confirm a local extreme (tighter for 5m noise)
MIN_PEAK_DISTANCE = 4           # min candles between the two peaks/troughs
LOOKBACK_FOR_RECENT_PEAK = 6    # 2nd peak/trough must be within this many candles of "now"
MIN_DIP_PCT = 0.15              # min % move between the two extremes and the point between them
SCAN_INTERVAL_SECONDS = 30 * 60  # check every 30 minutes (chart stays on the 5m timeframe)
MAX_CONCURRENT_REQUESTS = 8     # how many symbols to fetch in parallel (be gentle with rate limits)
SYMBOL_CACHE_SECONDS = 60 * 60  # refresh the futures symbol list once an hour

# Binance is the data source (fast, reliable REST API).
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})
# Toobit is only used to fetch its symbol list, to filter which Binance
# futures we actually scan (only coins listed as futures on both).
toobit_exchange = ccxt.toobit({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},
})

_symbol_cache = {"symbols": [], "loaded_at": 0}


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            merged = dict(DEFAULT_STATE)
            merged.update(data)
            return merged
        except Exception:
            logger.exception("Failed to read state file, using defaults")
    return dict(DEFAULT_STATE)


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


STATE = load_state()

# Avoid re-alerting the same pattern on every scan cycle
ALERTED = set()


# ---------------------------------------------------------------------------
# Symbol list (Binance USDT futures, filtered to coins also on Toobit)
# ---------------------------------------------------------------------------
async def get_futures_symbols(force: bool = False):
    import time

    now = time.time()
    if not force and _symbol_cache["symbols"] and (now - _symbol_cache["loaded_at"] < SYMBOL_CACHE_SECONDS):
        return _symbol_cache["symbols"]

    binance_markets = await asyncio.to_thread(exchange.load_markets, force)
    toobit_markets = await asyncio.to_thread(toobit_exchange.load_markets, force)

    toobit_bases = {
        m["base"]
        for m in toobit_markets.values()
        if m.get("contract") and m.get("linear") and m.get("quote") == "USDT" and m.get("active", True)
    }

    symbols = [
        m["symbol"]
        for m in binance_markets.values()
        if m.get("contract")
        and m.get("linear")
        and m.get("quote") == "USDT"
        and m.get("active", True)
        and m.get("base") in toobit_bases
    ]
    symbols.sort()
    _symbol_cache["symbols"] = symbols
    _symbol_cache["loaded_at"] = now
    return symbols


async def get_active_watchlist():
    if STATE["mode"] == "custom":
        return list(STATE["watchlist"])
    return await get_futures_symbols()


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------
def _find_extrema(values: np.ndarray, mode: str):
    """mode: 'high' for peaks (double top), 'low' for troughs (double bottom)."""
    comparator = np.greater_equal if mode == "high" else np.less_equal
    idx = argrelextrema(values, comparator, order=PEAK_ORDER)[0]

    filtered = []
    for i in idx:
        if filtered and i - filtered[-1] <= PEAK_ORDER:
            better = values[i] >= values[filtered[-1]] if mode == "high" else values[i] <= values[filtered[-1]]
            if better:
                filtered[-1] = i
        else:
            filtered.append(i)
    return filtered


def detect_pattern(ohlcv, tolerance_pct: float):
    """
    Returns a dict describing a forming double-top or double-bottom, or None.
    Checks both patterns and returns whichever is found (top takes priority
    if somehow both match, which is rare).
    """
    if len(ohlcv) < 20:
        return None

    highs = np.array([c[2] for c in ohlcv])
    lows = np.array([c[3] for c in ohlcv])
    closes = np.array([c[4] for c in ohlcv])
    n = len(ohlcv)
    tolerance = tolerance_pct / 100.0

    for mode, series in (("top", highs), ("bottom", lows)):
        pts = _find_extrema(series, "high" if mode == "top" else "low")
        if len(pts) < 2:
            continue

        p2 = pts[-1]
        p1 = pts[-2]

        if (n - 1 - p2) > LOOKBACK_FOR_RECENT_PEAK:
            continue
        if p2 - p1 < MIN_PEAK_DISTANCE:
            continue

        v1, v2 = series[p1], series[p2]
        if v1 == 0:
            continue
        diff_pct = abs(v1 - v2) / v1
        if diff_pct > tolerance:
            continue

        if mode == "top":
            segment = highs[p1:p2 + 1]
            between = segment.min()
            move_pct = (min(v1, v2) - between) / min(v1, v2)
        else:
            segment = lows[p1:p2 + 1]
            between = segment.max()
            move_pct = (between - max(v1, v2)) / max(v1, v2)

        if move_pct * 100 < MIN_DIP_PCT:
            continue

        current_price = closes[-1]
        confirmed = current_price < between if mode == "top" else current_price > between

        return {
            "pattern": "double_top" if mode == "top" else "double_bottom",
            "p1_idx": int(p1),
            "p2_idx": int(p2),
            "p1_price": float(v1),
            "p2_price": float(v2),
            "mid_price": float(between),
            "diff_pct": float(diff_pct * 100),
            "current_price": float(current_price),
            "confirmed": bool(confirmed),
        }

    return None


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def fetch_ohlcv_safe(symbol: str, timeframe: str):
    async with _sem:
        try:
            return await asyncio.to_thread(
                exchange.fetch_ohlcv, symbol, timeframe, None, CANDLE_LIMIT
            )
        except Exception as e:
            logger.warning(f"fetch_ohlcv failed for {symbol}: {e}")
            return None


def format_alert(symbol: str, tf: str, result: dict) -> str:
    label = "دو قله" if result["pattern"] == "double_top" else "دو دره"
    status = "تایید شده (خط گردن شکسته)" if result["confirmed"] else "در حال شکل‌گیری"
    p_label = "قله" if result["pattern"] == "double_top" else "دره"
    lines = [
        f"🔔 الگوی {label} در {symbol} ({tf})",
        f"وضعیت: {status}",
        f"{p_label} ۱: {result['p1_price']:.6g}",
        f"{p_label} ۲: {result['p2_price']:.6g}",
        f"اختلاف دو نقطه: {result['diff_pct']:.3f}%",
        f"خط گردن: {result['mid_price']:.6g}",
        f"قیمت فعلی: {result['current_price']:.6g}",
    ]
    return "\n".join(lines)


async def run_scan(context: ContextTypes.DEFAULT_TYPE, chat_id: str, notify_if_empty: bool = False):
    tf = STATE["timeframe"]
    tolerance = STATE["tolerance_pct"]

    try:
        symbols = await get_active_watchlist()
    except Exception as e:
        logger.exception("Failed to load symbol list")
        await context.bot.send_message(chat_id=chat_id, text=f"خطا در گرفتن لیست نمادها: {e}")
        return

    if not symbols:
        if notify_if_empty:
            await context.bot.send_message(chat_id=chat_id, text="لیست نمادها خالیه.")
        return

    tasks = [fetch_ohlcv_safe(s, tf) for s in symbols]
    results = await asyncio.gather(*tasks)

    found_any = False
    for symbol, ohlcv in zip(symbols, results):
        if not ohlcv:
            continue
        result = detect_pattern(ohlcv, tolerance)
        key = (symbol, tf)
        if result:
            found_any = True
            fingerprint = (key, result["pattern"], result["p1_idx"], result["p2_idx"])
            if fingerprint not in ALERTED:
                ALERTED.add(fingerprint)
                await context.bot.send_message(chat_id=chat_id, text=format_alert(symbol, tf, result))
        else:
            stale = {fp for fp in ALERTED if fp[0] == key}
            for fp in stale:
                ALERTED.discard(fp)

    if notify_if_empty and not found_any:
        await context.bot.send_message(chat_id=chat_id, text="در حال حاضر الگوی دو قله/دو دره‌ای پیدا نشد.")


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! این ربات قیمت رو از بایننس می‌گیره (فقط ارزهایی که فیوچرزشون توی توبیت هم "
        f"هست) و تو تایم‌فریم {STATE['timeframe']} زیر نظر می‌گیره؛ هر ۳۰ دقیقه چک می‌کنه و "
        "وقتی الگوی دو قله یا دو دره با اختلاف خیلی کم در حال شکل‌گیریه بهت خبر می‌ده.\n\n"
        "دستورات:\n"
        "/mode all - پایش همه‌ی نمادهای مشترک بایننس/توبیت (پیش‌فرض)\n"
        "/mode custom - پایش فقط نمادهایی که خودت اضافه می‌کنی\n"
        "/add BTC/USDT - اضافه کردن نماد (فقط حالت custom)\n"
        "/remove BTC/USDT - حذف نماد (فقط حالت custom)\n"
        "/list - نمایش تنظیمات فعلی\n"
        "/settimeframe 5m - تغییر تایم‌فریم\n"
        "/settolerance 0.3 - حداکثر اختلاف مجاز بین دو قله/دره (درصد)\n"
        "/refresh - به‌روزرسانی لیست نمادها\n"
        "/scan - اسکن فوری"
    )


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or context.args[0] not in ("all", "custom"):
        await update.message.reply_text("مثال: /mode all یا /mode custom")
        return
    STATE["mode"] = context.args[0]
    save_state(STATE)
    await update.message.reply_text(f"حالت پایش روی «{STATE['mode']}» تنظیم شد.")


async def add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if STATE["mode"] != "custom":
        await update.message.reply_text("اول با /mode custom به حالت دستی برو.")
        return
    if not context.args:
        await update.message.reply_text("مثال: /add BTC/USDT:USDT")
        return
    symbol = context.args[0].upper()
    if symbol not in STATE["watchlist"]:
        STATE["watchlist"].append(symbol)
        save_state(STATE)
        await update.message.reply_text(f"✅ {symbol} اضافه شد.")
    else:
        await update.message.reply_text("این نماد از قبل توی لیست هست.")


async def remove_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("مثال: /remove BTC/USDT:USDT")
        return
    symbol = context.args[0].upper()
    if symbol in STATE["watchlist"]:
        STATE["watchlist"].remove(symbol)
        save_state(STATE)
        await update.message.reply_text(f"❌ {symbol} حذف شد.")
    else:
        await update.message.reply_text("این نماد توی لیست نبود.")


async def list_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if STATE["mode"] == "all":
        try:
            symbols = await get_active_watchlist()
            count_text = f"{len(symbols)} نماد (مشترک بین بایننس و توبیت)"
        except Exception as e:
            count_text = f"خطا در گرفتن لیست: {e}"
    else:
        count_text = f"{len(STATE['watchlist'])} نماد دستی"

    await update.message.reply_text(
        f"حالت: {STATE['mode']}\n"
        f"تایم‌فریم: {STATE['timeframe']}\n"
        f"حداکثر اختلاف مجاز بین دو نقطه: {STATE['tolerance_pct']}%\n"
        f"تعداد نمادهای تحت پایش: {count_text}"
    )


async def set_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("مثال: /settimeframe 5m")
        return
    STATE["timeframe"] = context.args[0]
    save_state(STATE)
    await update.message.reply_text(f"تایم‌فریم به {STATE['timeframe']} تغییر کرد.")


async def set_tolerance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("مثال: /settolerance 0.3")
        return
    try:
        value = float(context.args[0])
    except ValueError:
        await update.message.reply_text("عدد معتبر بده، مثلاً 0.3")
        return
    STATE["tolerance_pct"] = value
    save_state(STATE)
    await update.message.reply_text(f"حداکثر اختلاف مجاز روی {value}% تنظیم شد.")


async def refresh_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("در حال به‌روزرسانی لیست نمادها از بایننس/توبیت...")
    try:
        symbols = await get_futures_symbols(force=True)
        await update.message.reply_text(f"لیست به‌روزرسانی شد. {len(symbols)} نماد مشترک پیدا شد.")
    except Exception as e:
        await update.message.reply_text(f"خطا: {e}")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("در حال اسکن... (بسته به تعداد نمادها ممکنه کمی طول بکشه)")
    await run_scan(context, chat_id=update.effective_chat.id, notify_if_empty=True)


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    if CHAT_ID:
        await run_scan(context, chat_id=CHAT_ID, notify_if_empty=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def on_shutdown(app):
    exchange.close()
    toobit_exchange.close()


def main():
    if not BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN environment variable is not set.")

    app = Application.builder().token(BOT_TOKEN).post_shutdown(on_shutdown).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mode", set_mode))
    app.add_handler(CommandHandler("add", add_symbol))
    app.add_handler(CommandHandler("remove", remove_symbol))
    app.add_handler(CommandHandler("list", list_settings))
    app.add_handler(CommandHandler("settimeframe", set_timeframe))
    app.add_handler(CommandHandler("settolerance", set_tolerance))
    app.add_handler(CommandHandler("refresh", refresh_symbols))
    app.add_handler(CommandHandler("scan", scan_command))

    if app.job_queue is not None:
        app.job_queue.run_repeating(scheduled_scan, interval=SCAN_INTERVAL_SECONDS, first=15)

    logger.info("Bot started.")
    app.run_polling()


if __name__ == "__main__":
    main()
