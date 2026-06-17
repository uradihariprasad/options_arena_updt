from gevent import monkey
monkey.patch_all()

import os
import time
import json
import traceback
import threading
import math
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ─────────────────────────────────────────────
#  App bootstrap
# ─────────────────────────────────────────────
import os as _os
_BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
app = Flask(__name__, static_folder=_BASE_DIR)
app.config['SECRET_KEY'] = 'trading-intelligence-secret'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# ─────────────────────────────────────────────
#  Global state
# ─────────────────────────────────────────────
ACCESS_TOKEN = ""
THREAD_RUNNING = False
data_thread = None

INSTRUMENTS = {
    'NIFTY':  {'index_key': 'NSE_INDEX|Nifty 50',  'option_prefix': 'NSE_FO', 'strike_gap': 50,  'lot_size': 50},
    'SENSEX': {'index_key': 'BSE_INDEX|SENSEX',     'option_prefix': 'BSE_FO', 'strike_gap': 100, 'lot_size': 10}
}

def _empty_index_data():
    return {
        'ltp': 0, 'open': 0, 'high': 0, 'low': 0, 'close': 0,
        'prev_close': 0, 'prev_high': 0, 'prev_low': 0,
        'volume': 0, 'change': 0, 'change_pct': 0,
        'vwap': 0, 'candles_1m': [], 'candles_3m': [], 'candles_5m': [],
        'option_chain': [], 'supports': [], 'resistances': [],
        'alerts': [], 'chart_alerts': [], 'trade_suggestion': {},
        'smc_zones': [], 'oi_analysis': {}, 'vix': 0,
        'institutional': {}, 'dashboard': {}
    }

market_data = {
    'NIFTY':  _empty_index_data(),
    'SENSEX': _empty_index_data()
}

alert_history       = {'NIFTY': deque(maxlen=50), 'SENSEX': deque(maxlen=50)}
chart_alert_history = {'NIFTY': deque(maxlen=30), 'SENSEX': deque(maxlen=30)}

# ─────────────────────────────────────────────
#  Upstox API helpers
# ─────────────────────────────────────────────
UPSTOX_BASE = "https://api.upstox.com/v2"

def upstox_headers():
    return {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

def _get(url, params=None, timeout=10):
    try:
        r = requests.get(url, headers=upstox_headers(), params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        print(f"[API] {url} → {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[API ERROR] {url}: {e}")
    return None

def fetch_market_quote(instrument_key):
    url = f"{UPSTOX_BASE}/market-quote/quotes"
    data = _get(url, params={'instrument_key': instrument_key})
    if data and data.get('status') == 'success':
        key = instrument_key.replace('|', ':')
        return data.get('data', {}).get(key)
    return None

def fetch_intraday_candles(instrument_key, interval='1minute'):
    enc = requests.utils.quote(instrument_key, safe='')
    url = f"{UPSTOX_BASE}/historical-candle/intraday/{enc}/{interval}"
    data = _get(url)
    if data and data.get('status') == 'success':
        return data.get('data', {}).get('candles', [])
    return []

def fetch_historical_candles(instrument_key, interval='day', days_back=5):
    from_dt = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_dt   = datetime.now().strftime('%Y-%m-%d')
    enc = requests.utils.quote(instrument_key, safe='')
    url = f"{UPSTOX_BASE}/historical-candle/{enc}/{interval}/{to_dt}/{from_dt}"
    data = _get(url)
    if data and data.get('status') == 'success':
        return data.get('data', {}).get('candles', [])
    return []

def fetch_option_expiries(instrument_key):
    url = f"{UPSTOX_BASE}/option/contract"
    data = _get(url, params={'instrument_key': instrument_key})
    if data and data.get('status') == 'success':
        expiries = sorted(set(
            c.get('expiry') for c in data.get('data', [])
            if c.get('expiry') and c.get('expiry') >= datetime.now().strftime('%Y-%m-%d')
        ))
        return expiries[0] if expiries else None
    return None

def fetch_option_chain(instrument_key, expiry_date):
    url = f"{UPSTOX_BASE}/option/chain"
    data = _get(url, params={'instrument_key': instrument_key, 'expiry_date': expiry_date})
    if data and data.get('status') == 'success':
        return data.get('data', [])
    return []

def fetch_india_vix():
    q = fetch_market_quote('NSE_INDEX|India VIX')
    if q:
        return q.get('last_price', 0)
    return 0

# ─────────────────────────────────────────────
#  Candle aggregation
# ─────────────────────────────────────────────
def aggregate_candles(minute_candles, period_minutes):
    """Group 1-min candles into N-min OHLCV candles."""
    if not minute_candles:
        return []
    # Sort ascending by time
    sorted_candles = sorted(minute_candles, key=lambda c: c[0])
    result = []
    chunk = []
    for candle in sorted_candles:
        chunk.append(candle)
        if len(chunk) == period_minutes:
            ts   = chunk[0][0]
            o    = chunk[0][1]
            h    = max(c[2] for c in chunk)
            lo   = min(c[3] for c in chunk)
            cl   = chunk[-1][4]
            vol  = sum(c[5] if len(c) > 5 else 0 for c in chunk)
            result.append([ts, o, h, lo, cl, vol])
            chunk = []
    if chunk:
        ts   = chunk[0][0]
        o    = chunk[0][1]
        h    = max(c[2] for c in chunk)
        lo   = min(c[3] for c in chunk)
        cl   = chunk[-1][4]
        vol  = sum(c[5] if len(c) > 5 else 0 for c in chunk)
        result.append([ts, o, h, lo, cl, vol])
    return result

# ─────────────────────────────────────────────
#  VWAP
# ─────────────────────────────────────────────
def calculate_vwap(candles):
    """Rolling VWAP from list of [ts, o, h, l, c, vol] candles."""
    if not candles:
        return 0, {}
    cum_tp_vol = 0.0
    cum_vol    = 0.0
    vwap_points = []
    for c in candles:
        if len(c) < 6:
            continue
        o, h, lo, cl, vol = c[1], c[2], c[3], c[4], c[5]
        tp  = (h + lo + cl) / 3.0
        vol = max(vol, 1)
        cum_tp_vol += tp * vol
        cum_vol    += vol
        vwap_points.append({'time': c[0], 'value': round(cum_tp_vol / cum_vol, 2)})
    vwap = round(cum_tp_vol / cum_vol, 2) if cum_vol > 0 else 0
    return vwap, {'points': vwap_points}

# ─────────────────────────────────────────────
#  Support / Resistance detection
# ─────────────────────────────────────────────
def detect_support_resistance(candles, ltp, prev_high=0, prev_low=0, num_levels=3):
    if len(candles) < 6:
        supports, resistances = [], []
        if prev_high > ltp:
            resistances.append({'price': prev_high, 'volume': 0, 'tests': 1,
                                 'strength': 'prev_day_high', 'score': 65, 'factors': ['Prev Day High']})
        if prev_low > 0 and prev_low < ltp:
            supports.append({'price': prev_low, 'volume': 0, 'tests': 1,
                              'strength': 'prev_day_low', 'score': 65, 'factors': ['Prev Day Low']})
        return supports, resistances

    highs  = [c[2] for c in candles]
    lows   = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    vols   = [c[5] if len(c) > 5 else 1 for c in candles]
    avg_vol = max(np.mean(vols), 1)

    pivot_highs = []
    pivot_lows  = []
    lookback = 2
    for i in range(lookback, len(candles) - lookback):
        is_pivot_high = all(highs[i] >= highs[i-j] and highs[i] >= highs[i+j] for j in range(1, lookback+1))
        is_pivot_low  = all(lows[i]  <= lows[i-j]  and lows[i]  <= lows[i+j]  for j in range(1, lookback+1))
        if is_pivot_high:
            pivot_highs.append({'price': highs[i], 'volume': vols[i], 'idx': i})
        if is_pivot_low:
            pivot_lows.append({'price': lows[i],  'volume': vols[i], 'idx': i})

    def cluster_levels(pivots, threshold_pct=0.002):
        if not pivots:
            return []
        pivots_sorted = sorted(pivots, key=lambda x: x['price'])
        clusters = []
        current = [pivots_sorted[0]]
        for p in pivots_sorted[1:]:
            if abs(p['price'] - current[-1]['price']) / max(current[-1]['price'], 1) < threshold_pct:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)
        result = []
        for cl in clusters:
            avg_price  = np.mean([x['price']  for x in cl])
            total_vol  = sum(x['volume'] for x in cl)
            tests      = len(cl)
            avg_vf     = total_vol / avg_vol if avg_vol > 0 else 1
            score      = min(100, tests * 20 + int(avg_vf * 15))
            factors    = []
            if tests >= 3: factors.append(f'Tested {tests}×')
            if avg_vf > 1.5: factors.append('High Volume')
            result.append({'price': round(avg_price, 2), 'volume': round(total_vol, 2),
                           'tests': tests, 'avg_vol_factor': round(avg_vf, 2), 'score': score, 'factors': factors})
        return result

    raw_res = cluster_levels(pivot_highs)
    raw_sup = cluster_levels(pivot_lows)

    def assign_strength(lvl):
        s = lvl['score']
        t = lvl['tests']
        if s >= 60 or t >= 3: return 'strong'
        if s >= 35 or t >= 2: return 'moderate'
        return 'weak'

    for lvl in raw_res: lvl['strength'] = assign_strength(lvl)
    for lvl in raw_sup: lvl['strength'] = assign_strength(lvl)

    resistances = sorted([l for l in raw_res if l['price'] > ltp], key=lambda x: x['price'])[:num_levels]
    supports    = sorted([l for l in raw_sup if l['price'] < ltp], key=lambda x: x['price'], reverse=True)[:num_levels]

    # Pad if fewer than 2 levels
    gap = ltp * 0.005
    while len(resistances) < 2:
        base = resistances[-1]['price'] if resistances else ltp
        resistances.append({'price': round(base + gap, 2), 'volume': 0, 'tests': 1,
                            'strength': 'weak', 'score': 20, 'factors': ['Calculated']})
    while len(supports) < 2:
        base = supports[-1]['price'] if supports else ltp
        supports.append({'price': round(base - gap, 2), 'volume': 0, 'tests': 1,
                         'strength': 'weak', 'score': 20, 'factors': ['Calculated']})

    # Append prev day levels
    if prev_high and prev_high > ltp:
        if not any(abs(r['price'] - prev_high) / max(prev_high, 1) < 0.001 for r in resistances):
            resistances.append({'price': prev_high, 'volume': 0, 'tests': 1,
                                 'strength': 'prev_day_high', 'score': 65, 'factors': ['Prev Day High']})
    if prev_low and prev_low > 0 and prev_low < ltp:
        if not any(abs(s['price'] - prev_low) / max(prev_low, 1) < 0.001 for s in supports):
            supports.append({'price': prev_low, 'volume': 0, 'tests': 1,
                              'strength': 'prev_day_low', 'score': 65, 'factors': ['Prev Day Low']})

    return supports, resistances

# ─────────────────────────────────────────────
#  Option chain analysis
# ─────────────────────────────────────────────
def analyze_option_chain_data(raw_chain, ltp, strike_gap):
    if not raw_chain:
        return {'chain': [], 'analysis': {'pcr': 1.0, 'max_ce_oi_strike': 0,
                'max_pe_oi_strike': 0, 'signals': [], 'oi_unwinding': False,
                'short_buildup': False, 'long_buildup': False}}

    rows = []
    total_ce_oi = 0
    total_pe_oi = 0
    max_ce_oi   = 0
    max_pe_oi   = 0
    max_ce_strike = 0
    max_pe_strike = 0

    for item in raw_chain:
        strike = item.get('strike_price', 0)
        ce = item.get('call_options', {}) or {}
        pe = item.get('put_options',  {}) or {}
        ce_md = ce.get('market_data', {}) or {}
        pe_md = pe.get('market_data', {}) or {}

        ce_oi  = ce_md.get('oi', 0) or 0
        pe_oi  = pe_md.get('oi', 0) or 0
        ce_ltp = ce_md.get('ltp', 0) or 0
        pe_ltp = pe_md.get('ltp', 0) or 0
        ce_vol = ce_md.get('volume', 0) or 0
        pe_vol = pe_md.get('volume', 0) or 0

        total_ce_oi += ce_oi
        total_pe_oi += pe_oi

        if ce_oi > max_ce_oi:
            max_ce_oi = ce_oi
            max_ce_strike = strike
        if pe_oi > max_pe_oi:
            max_pe_oi = pe_oi
            max_pe_strike = strike

        rows.append({
            'strike': strike, 'ce_oi': ce_oi, 'pe_oi': pe_oi,
            'ce_ltp': ce_ltp, 'pe_ltp': pe_ltp,
            'ce_volume': ce_vol, 'pe_volume': pe_vol,
            'itm': strike <= ltp
        })

    pcr = round(total_pe_oi / max(total_ce_oi, 1), 3)

    signals = []
    if pcr > 1.3:
        signals.append({'signal': 'High PCR — Bullish Sentiment', 'bias': 'bullish', 'strength': 'strong'})
    elif pcr > 1.0:
        signals.append({'signal': 'PCR above 1 — Mild Bullish', 'bias': 'bullish', 'strength': 'moderate'})
    elif pcr < 0.7:
        signals.append({'signal': 'Low PCR — Bearish Sentiment', 'bias': 'bearish', 'strength': 'strong'})
    elif pcr < 1.0:
        signals.append({'signal': 'PCR below 1 — Mild Bearish', 'bias': 'bearish', 'strength': 'moderate'})

    long_buildup  = total_pe_oi > total_ce_oi * 1.2
    short_buildup = total_ce_oi > total_pe_oi * 1.2

    if long_buildup:
        signals.append({'signal': 'Long Buildup in PE', 'bias': 'bullish', 'strength': 'strong'})
    if short_buildup:
        signals.append({'signal': 'Short Buildup in CE', 'bias': 'bearish', 'strength': 'moderate'})

    # Sort rows closest to ATM first
    rows_sorted = sorted(rows, key=lambda x: abs(x['strike'] - ltp))

    return {
        'chain': rows_sorted[:20],
        'analysis': {
            'pcr': pcr,
            'max_ce_oi_strike': max_ce_strike,
            'max_pe_oi_strike': max_pe_strike,
            'signals': signals,
            'oi_unwinding': False,
            'short_buildup': short_buildup,
            'long_buildup': long_buildup,
            'total_ce_oi': total_ce_oi,
            'total_pe_oi': total_pe_oi
        }
    }

# ─────────────────────────────────────────────
#  SMC pattern detection
# ─────────────────────────────────────────────
def detect_smc_patterns(candles, ltp):
    if len(candles) < 10:
        return []

    highs  = [c[2] for c in candles]
    lows   = [c[3] for c in candles]
    times  = [c[0] for c in candles]
    zones  = []

    # Find swing highs/lows
    for i in range(3, len(candles) - 3):
        # Bullish BOS: new high above previous swing high
        local_prev_high = max(highs[max(0, i-5):i])
        if highs[i] > local_prev_high * 1.001:
            zones.append({'type': 'BULL_BOS', 'price': round(highs[i], 2),
                          'time': str(times[i]), 'description': 'Break of Structure ↑'})

        # Bearish BOS: new low below previous swing low
        local_prev_low = min(lows[max(0, i-5):i])
        if lows[i] < local_prev_low * 0.999:
            zones.append({'type': 'BEAR_BOS', 'price': round(lows[i], 2),
                          'time': str(times[i]), 'description': 'Break of Structure ↓'})

    # CHOCH detection: look for reversal after sustained trend
    if len(candles) >= 15:
        recent_highs = highs[-10:]
        recent_lows  = lows[-10:]
        # If we had a series of lower highs and now broke above last high → BULL CHOCH
        if (recent_highs[-1] > max(recent_highs[-5:-1]) and
                recent_lows[-3] < recent_lows[-6]):
            zones.append({'type': 'BULL_CHOCH', 'price': round(recent_highs[-1], 2),
                          'time': str(times[-1]), 'description': 'Change of Character ↑'})
        # Series of higher lows then new low → BEAR CHOCH
        if (recent_lows[-1] < min(recent_lows[-5:-1]) and
                recent_highs[-3] > recent_highs[-6]):
            zones.append({'type': 'BEAR_CHOCH', 'price': round(recent_lows[-1], 2),
                          'time': str(times[-1]), 'description': 'Change of Character ↓'})

    # Deduplicate and keep latest 10
    seen_types = {}
    deduped = []
    for z in reversed(zones):
        key = (z['type'], round(z['price'] / 50) * 50)
        if key not in seen_types:
            seen_types[key] = True
            deduped.append(z)
    return deduped[:10]

# ─────────────────────────────────────────────
#  VWAP price action analysis
# ─────────────────────────────────────────────
def analyze_vwap_price_action(candles, vwap, ltp):
    if not candles or vwap == 0:
        return {'status': 'neutral', 'signals': [], 'vwap_slope': 0}

    signals = []
    if ltp > vwap * 1.002:
        status = 'above_vwap'
        signals.append(f'Price {round((ltp/vwap - 1)*100, 2)}% above VWAP')
    elif ltp < vwap * 0.998:
        status = 'below_vwap'
        signals.append(f'Price {round((1 - ltp/vwap)*100, 2)}% below VWAP')
    else:
        status = 'at_vwap'
        signals.append('Price at VWAP — decision zone')

    # VWAP slope from last 5 candles
    vwap_slope = 0
    if len(candles) >= 5:
        closes = [c[4] for c in candles[-5:]]
        if closes[0] != 0:
            vwap_slope = round((closes[-1] - closes[0]) / closes[0] * 100, 3)

    return {'status': status, 'signals': signals, 'vwap_slope': vwap_slope}

# ─────────────────────────────────────────────
#  Chart-level momentum alerts
# ─────────────────────────────────────────────
def generate_chart_alerts(index_name, candles, ltp, vwap_analysis, supports, resistances):
    if len(candles) < 6:
        return []

    alerts = []
    now_str = datetime.now().isoformat()
    vols = [c[5] if len(c) > 5 else 1 for c in candles]
    avg_vol = max(np.mean(vols), 1)

    for i in range(4, len(candles)):
        c  = candles[i]
        ts = str(c[0])
        o, h, lo, cl = c[1], c[2], c[3], c[4]
        vol = c[5] if len(c) > 5 else 0
        rng  = max(h - lo, 0.001)
        body = abs(cl - o)
        body_ratio = body / rng
        upper_wick = h - max(o, cl)
        lower_wick = min(o, cl) - lo
        vol_ratio  = vol / avg_vol

        # ── MOMENTUM_CONTINUATION ─────────────────
        last3 = candles[i-2:i+1]
        bull_cont = all(c2[4] > c2[1] for c2 in last3)
        bear_cont = all(c2[4] < c2[1] for c2 in last3)
        if bull_cont and body_ratio > 0.6 and vol_ratio > 1.2:
            reasons = ['3 consecutive bullish closes',
                       f'Body ratio {round(body_ratio, 2)}',
                       f'Volume {round(vol_ratio, 1)}× avg']
            if vwap_analysis.get('status') == 'above_vwap':
                reasons.append('Above VWAP')
            alerts.append({
                'type': 'MOMENTUM_CONTINUATION', 'direction': 'bullish',
                'time': ts, 'price': round(cl, 2), 'score': min(80, 55 + int(vol_ratio*5)),
                'label': '⚡ Bull Continuation', 'short_label': '↑C',
                'color': '#10b981', 'reasons': reasons
            })
        elif bear_cont and body_ratio > 0.6 and vol_ratio > 1.2:
            reasons = ['3 consecutive bearish closes',
                       f'Body ratio {round(body_ratio, 2)}',
                       f'Volume {round(vol_ratio, 1)}× avg']
            alerts.append({
                'type': 'MOMENTUM_CONTINUATION', 'direction': 'bearish',
                'time': ts, 'price': round(cl, 2), 'score': min(80, 55 + int(vol_ratio*5)),
                'label': '⚡ Bear Continuation', 'short_label': '↓C',
                'color': '#ef4444', 'reasons': reasons
            })

        # ── MOMENTUM_SLOWING ──────────────────────
        if i >= 6:
            prev5_moves = [abs(candles[j][4] - candles[j-1][4]) for j in range(i-4, i)]
            avg_prev_move = max(np.mean(prev5_moves[:-1]), 0.001)
            last_move = prev5_moves[-1]
            if last_move < avg_prev_move * 0.4 and vol_ratio < 0.9:
                direction = 'bullish_slowing' if cl > o else 'bearish_slowing'
                alerts.append({
                    'type': 'MOMENTUM_SLOWING', 'direction': direction,
                    'time': ts, 'price': round(cl, 2), 'score': 60,
                    'label': '⚠ Momentum Slowing', 'short_label': '~S',
                    'color': '#f59e0b',
                    'reasons': [f'Move {round(last_move, 1)} vs avg {round(avg_prev_move, 1)}',
                                f'Volume {round(vol_ratio, 1)}× avg']
                })

        # ── MOMENTUM_EXHAUSTION ───────────────────
        if body_ratio < 0.2 and vol_ratio > 1.5:
            has_long_upper = upper_wick > 0.6 * rng
            has_long_lower = lower_wick > 0.6 * rng
            if has_long_upper or has_long_lower:
                direction = 'bearish' if has_long_upper else 'bullish'
                alerts.append({
                    'type': 'MOMENTUM_EXHAUSTION', 'direction': direction,
                    'time': ts, 'price': round(cl, 2), 'score': 70,
                    'label': '🔥 Exhaustion Signal', 'short_label': '!E',
                    'color': '#f97316',
                    'reasons': ['Doji/Spinning top pattern',
                                f'Volume {round(vol_ratio, 1)}× avg',
                                'Long wick detected']
                })

    # Sort ascending, keep latest 15
    alerts_sorted = sorted(alerts, key=lambda x: x['time'])[-15:]
    return alerts_sorted

# ─────────────────────────────────────────────
#  High-level alerts
# ─────────────────────────────────────────────
def generate_alerts(index_name, data, oi_analysis, vwap_analysis, smc_zones, supports, resistances):
    alerts = []
    now   = datetime.now().isoformat()
    ltp   = data.get('ltp', 0)
    vix   = data.get('vix', 0)
    pcr   = oi_analysis.get('pcr', 1.0)

    if vix > 20:
        alerts.append({'type': 'WARNING', 'message': f'High VIX {vix:.1f} — Elevated volatility', 'time': now})
    if vix > 25:
        alerts.append({'type': 'DANGER', 'message': f'VIX SPIKE {vix:.1f} — Extreme fear', 'time': now})
    if pcr > 1.5:
        alerts.append({'type': 'INFO', 'message': f'PCR extreme {pcr:.2f} — Very bullish sentiment', 'time': now})
    elif pcr < 0.6:
        alerts.append({'type': 'WARNING', 'message': f'PCR extreme low {pcr:.2f} — Very bearish sentiment', 'time': now})

    status = vwap_analysis.get('status', 'neutral')
    if status == 'above_vwap':
        alerts.append({'type': 'INFO', 'message': 'Price trading above VWAP — Bullish', 'time': now})
    elif status == 'below_vwap':
        alerts.append({'type': 'WARNING', 'message': 'Price trading below VWAP — Bearish', 'time': now})

    for s in supports[:2]:
        dist_pct = abs(ltp - s['price']) / max(ltp, 1) * 100
        if dist_pct < 0.3:
            alerts.append({'type': 'INFO', 'message': f'Near support {s["price"]} ({dist_pct:.2f}% away)', 'time': now})
    for r in resistances[:2]:
        dist_pct = abs(r['price'] - ltp) / max(ltp, 1) * 100
        if dist_pct < 0.3:
            alerts.append({'type': 'WARNING', 'message': f'Near resistance {r["price"]} ({dist_pct:.2f}% away)', 'time': now})

    for zone in smc_zones[:3]:
        alerts.append({'type': 'SMC', 'message': f'{zone["type"]}: {zone["description"]} @ {zone["price"]}', 'time': now})

    return alerts[:15]

# ─────────────────────────────────────────────
#  Trade suggestion
# ─────────────────────────────────────────────
def generate_trade_suggestion(index_name, data, oi_analysis, vwap_analysis, smc_zones, supports, resistances, vix):
    ltp     = data.get('ltp', 0)
    pcr     = oi_analysis.get('pcr', 1.0)
    signals = oi_analysis.get('signals', [])
    reasons = []
    score   = 0

    # VWAP
    if vwap_analysis.get('status') == 'above_vwap':
        score += 2; reasons.append('Price above VWAP')
    elif vwap_analysis.get('status') == 'below_vwap':
        score -= 2; reasons.append('Price below VWAP')

    # PCR
    if pcr > 1.2:
        score += 2; reasons.append(f'Bullish PCR {pcr:.2f}')
    elif pcr < 0.8:
        score -= 2; reasons.append(f'Bearish PCR {pcr:.2f}')

    # OI signals
    for sig in signals:
        if sig['bias'] == 'bullish':
            score += 1; reasons.append(sig['signal'])
        elif sig['bias'] == 'bearish':
            score -= 1; reasons.append(sig['signal'])

    # SMC
    bull_smc = sum(1 for z in smc_zones if 'BULL' in z['type'])
    bear_smc = sum(1 for z in smc_zones if 'BEAR' in z['type'])
    if bull_smc > bear_smc:
        score += 1; reasons.append('Bullish SMC structure')
    elif bear_smc > bull_smc:
        score -= 1; reasons.append('Bearish SMC structure')

    # VIX penalty
    if vix > 20:
        score = int(score * 0.7)
        reasons.append(f'High VIX {vix:.1f} reduces confidence')

    # Determine bias
    if score >= 2:
        bias = 'BUY'
    elif score <= -2:
        bias = 'SELL'
    else:
        bias = 'NEUTRAL'

    # Entry / target / SL
    if bias == 'BUY' and supports:
        sl     = supports[0]['price']
        target = resistances[0]['price'] if resistances else round(ltp * 1.005, 2)
        entry  = ltp
    elif bias == 'SELL' and resistances:
        sl     = resistances[0]['price']
        target = supports[0]['price'] if supports else round(ltp * 0.995, 2)
        entry  = ltp
    else:
        entry  = ltp
        target = ltp
        sl     = ltp
    
    risk   = abs(entry - sl)
    reward = abs(target - entry)
    rr     = round(reward / max(risk, 0.01), 2)
    conf   = min(90, max(20, 50 + abs(score) * 10))

    # Option suggestion
    instr = INSTRUMENTS.get(index_name, {})
    sg    = instr.get('strike_gap', 50)
    if bias == 'BUY':
        ce_strike = int(round(ltp / sg) * sg)
        opt_suggestion = {'type': 'CE', 'strike': ce_strike, 'action': f'Buy {ce_strike} CE'}
    elif bias == 'SELL':
        pe_strike = int(round(ltp / sg) * sg)
        opt_suggestion = {'type': 'PE', 'strike': pe_strike, 'action': f'Buy {pe_strike} PE'}
    else:
        opt_suggestion = {'type': '-', 'strike': 0, 'action': 'Wait for signal'}

    return {
        'bias': bias, 'entry': round(entry, 2), 'target': round(target, 2),
        'stop_loss': round(sl, 2), 'risk_reward': rr, 'confidence': conf,
        'reasons': reasons[:6], 'option_suggestion': opt_suggestion
    }

# ─────────────────────────────────────────────
#  RSI helper
# ─────────────────────────────────────────────
def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g  = np.mean(gains[:period])
    avg_l  = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100
    rs  = avg_g / avg_l
    return round(100 - 100 / (1 + rs), 2)

# ─────────────────────────────────────────────
#  Dashboard metrics
# ─────────────────────────────────────────────
def calculate_dashboard(data, oi_analysis, vwap_analysis, vix):
    ltp    = data.get('ltp', 0)
    op     = data.get('open', ltp)
    candles = data.get('candles_5m', [])
    closes  = [c[4] for c in candles] if candles else [ltp]
    rsi     = calculate_rsi(closes)
    pcr     = oi_analysis.get('pcr', 1.0)
    vwap    = data.get('vwap', ltp)
    change_pct = data.get('change_pct', 0)

    # Trend
    if change_pct > 0.5:   trend_v, trend_c = 'BULLISH', 'bullish'
    elif change_pct < -0.5: trend_v, trend_c = 'BEARISH', 'bearish'
    else:                   trend_v, trend_c = 'SIDEWAYS', 'neutral'

    # Signal
    vs  = vwap_analysis.get('status', 'neutral')
    ois = oi_analysis.get('signals', [])
    bull_sig = sum(1 for s in ois if s['bias'] == 'bullish')
    bear_sig = sum(1 for s in ois if s['bias'] == 'bearish')
    if vs == 'above_vwap' and bull_sig >= bear_sig: sig_v, sig_c = 'BUY',  'bullish'
    elif vs == 'below_vwap' and bear_sig > bull_sig: sig_v, sig_c = 'SELL', 'bearish'
    else:                                            sig_v, sig_c = 'HOLD', 'neutral'

    # RSI class
    if rsi > 70:   rsi_c = 'bearish'
    elif rsi < 30: rsi_c = 'bullish'
    else:          rsi_c = 'neutral'

    # VWAP position
    if vs == 'above_vwap': vwap_v, vwap_c = 'Above', 'bullish'
    elif vs == 'below_vwap': vwap_v, vwap_c = 'Below', 'bearish'
    else: vwap_v, vwap_c = 'At VWAP', 'neutral'

    # VIX
    if vix > 20:     vix_v, vix_c = f'{vix:.1f} HIGH', 'bearish'
    elif vix > 14:   vix_v, vix_c = f'{vix:.1f} MED',  'neutral'
    elif vix > 0:    vix_v, vix_c = f'{vix:.1f} LOW',  'bullish'
    else:            vix_v, vix_c = 'N/A', 'neutral'

    # PCR
    if pcr > 1.2:    pcr_v, pcr_c = f'{pcr:.2f} ↑',  'bullish'
    elif pcr < 0.8:  pcr_v, pcr_c = f'{pcr:.2f} ↓',  'bearish'
    else:            pcr_v, pcr_c = f'{pcr:.2f}',     'neutral'

    # Momentum
    if len(closes) >= 3:
        recent = closes[-3:]
        if recent[-1] > recent[0]: mom_v, mom_c = 'Rising', 'bullish'
        elif recent[-1] < recent[0]: mom_v, mom_c = 'Falling', 'bearish'
        else: mom_v, mom_c = 'Flat', 'neutral'
    else:
        mom_v, mom_c = 'N/A', 'neutral'

    # Risk
    if vix > 20 or abs(change_pct) > 1.5: risk_v, risk_c = 'HIGH',   'bearish'
    elif vix > 15 or abs(change_pct) > 0.8: risk_v, risk_c = 'MEDIUM', 'neutral'
    else:                                    risk_v, risk_c = 'LOW',    'bullish'

    return {
        'trend':         {'value': trend_v, 'class': trend_c},
        'signal':        {'value': sig_v,   'class': sig_c},
        'rsi':           {'value': str(rsi), 'class': rsi_c},
        'vwap_position': {'value': vwap_v,  'class': vwap_c},
        'vix_level':     {'value': vix_v,   'class': vix_c},
        'pcr':           {'value': pcr_v,   'class': pcr_c},
        'momentum':      {'value': mom_v,   'class': mom_c},
        'risk_level':    {'value': risk_v,  'class': risk_c}
    }

# ─────────────────────────────────────────────
#  Institutional flow analysis
# ─────────────────────────────────────────────
def analyze_institutional(oi_analysis, smc_zones):
    signals  = oi_analysis.get('signals', [])
    bull_cnt = sum(1 for s in signals if s['bias'] == 'bullish')
    bear_cnt = sum(1 for s in signals if s['bias'] == 'bearish')
    bull_smc = sum(1 for z in smc_zones if 'BULL' in z['type'])
    bear_smc = sum(1 for z in smc_zones if 'BEAR' in z['type'])

    total_bull = bull_cnt + bull_smc
    total_bear = bear_cnt + bear_smc

    if total_bull > total_bear:
        fii_bias = 'Long'
        d_bias   = 'Bullish'
        f_buildup = 'Long Buildup'
    elif total_bear > total_bull:
        fii_bias = 'Short'
        d_bias   = 'Bearish'
        f_buildup = 'Short Buildup'
    else:
        fii_bias = 'Neutral'
        d_bias   = 'Sideways'
        f_buildup = 'No Clear Buildup'

    return {'fii_bias': fii_bias, 'futures_buildup': f_buildup, 'directional_bias': d_bias}

# ─────────────────────────────────────────────
#  Serialize market data for SocketIO
# ─────────────────────────────────────────────
def serialize_market_data(d, idx_name):
    def safe(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0
        return v

    candles_out = []
    source_candles = d.get('candles_5m') or d.get('candles_3m') or d.get('candles_1m', [])
    for c in source_candles[-200:]:
        if len(c) >= 5:
            try:
                candles_out.append([c[0], safe(c[1]), safe(c[2]), safe(c[3]), safe(c[4]),
                                    safe(c[5]) if len(c) > 5 else 0])
            except Exception:
                pass

    return {
        'ltp':        safe(d.get('ltp', 0)),
        'open':       safe(d.get('open', 0)),
        'high':       safe(d.get('high', 0)),
        'low':        safe(d.get('low', 0)),
        'prev_close': safe(d.get('prev_close', 0)),
        'prev_high':  safe(d.get('prev_high', 0)),
        'prev_low':   safe(d.get('prev_low', 0)),
        'vwap':       safe(d.get('vwap', 0)),
        'change':     safe(d.get('change', 0)),
        'change_pct': safe(d.get('change_pct', 0)),
        'volume':     safe(d.get('volume', 0)),
        'candles':    candles_out,
        'supports':   d.get('supports', []),
        'resistances':d.get('resistances', []),
        'chart_alerts': list(d.get('chart_alerts', [])),
        'oi_analysis': d.get('oi_analysis', {}),
        'vix':        safe(d.get('vix', 0)),
        'institutional': d.get('institutional', {}),
        'smc_zones':  d.get('smc_zones', []),
        'trade_suggestion': d.get('trade_suggestion', {}),
        'dashboard':  d.get('dashboard', {}),
        'alerts':     d.get('alerts', [])
    }

# ─────────────────────────────────────────────
#  Main data fetch loop
# ─────────────────────────────────────────────
def data_fetch_loop():
    global THREAD_RUNNING
    iteration    = 0
    expiry_cache = {}

    while THREAD_RUNNING:
        for idx_name, idx_info in INSTRUMENTS.items():
            try:
                ik = idx_info['index_key']
                md = market_data[idx_name]

                # 1. Fetch LTP + OHLC
                quote = fetch_market_quote(ik)
                if quote:
                    ohlc = quote.get('ohlc', {})
                    md['ltp']       = quote.get('last_price', md['ltp'])
                    md['open']      = ohlc.get('open',  md['open'])
                    md['high']      = ohlc.get('high',  md['high'])
                    md['low']       = ohlc.get('low',   md['low'])
                    md['close']     = md['ltp']
                    md['prev_close']= ohlc.get('close', md['prev_close'])
                    md['volume']    = quote.get('volume', md['volume'])
                    prev_c          = md['prev_close'] if md['prev_close'] else 1
                    md['change']    = round(md['ltp'] - prev_c, 2)
                    md['change_pct']= round(md['change'] / prev_c * 100, 3)

                ltp = md.get('ltp', 0)
                if ltp == 0:
                    continue

                # 2. Fetch intraday 1m candles → aggregate
                raw_1m = fetch_intraday_candles(ik, '1minute')
                if raw_1m:
                    md['candles_1m'] = raw_1m
                    md['candles_3m'] = aggregate_candles(raw_1m, 3)
                    md['candles_5m'] = aggregate_candles(raw_1m, 5)

                minute_candles = md.get('candles_1m', [])
                candles_5m     = md.get('candles_5m', [])
                all_candles    = candles_5m if candles_5m else minute_candles

                # 3. Calculate VWAP  ← BEFORE chart_alerts
                vwap_val = 0
                if minute_candles:
                    vwap_val, _ = calculate_vwap(minute_candles)
                    md['vwap'] = vwap_val

                # 4. Prev day H/L every 60 iterations
                if iteration % 60 == 0:
                    hist = fetch_historical_candles(ik, 'day', days_back=3)
                    if len(hist) >= 2:
                        prev_day = hist[-2]  # Upstox returns newest first
                        md['prev_high'] = prev_day[2]
                        md['prev_low']  = prev_day[3]

                prev_h = md.get('prev_high', 0)
                prev_l = md.get('prev_low',  0)

                # 5. Detect S/R
                supports, resistances = detect_support_resistance(
                    all_candles, ltp, prev_h, prev_l)
                md['supports']    = supports
                md['resistances'] = resistances

                # 6. Detect SMC
                smc_zones = detect_smc_patterns(candles_5m or all_candles, ltp)
                md['smc_zones'] = smc_zones

                # 7. Analyze VWAP price action  ← MUST come before chart_alerts
                if all_candles:
                    vwap_analysis = analyze_vwap_price_action(all_candles, vwap_val, ltp)
                else:
                    vwap_analysis = {'status': 'neutral', 'signals': [], 'vwap_slope': 0}

                # 8. Generate chart alerts → merge into history
                new_chart_alerts = generate_chart_alerts(
                    idx_name, all_candles, ltp, vwap_analysis, supports, resistances)
                for ca in new_chart_alerts:
                    existing_times = {x['time'] for x in chart_alert_history[idx_name]}
                    if ca['time'] not in existing_times:
                        chart_alert_history[idx_name].appendleft(ca)
                md['chart_alerts'] = list(chart_alert_history[idx_name])

                # 9. Option chain (cache expiry, refresh every 300 iterations)
                if iteration % 300 == 0 or idx_name not in expiry_cache:
                    expiry = fetch_option_expiries(ik)
                    if expiry:
                        expiry_cache[idx_name] = expiry

                expiry = expiry_cache.get(idx_name)
                if expiry and iteration % 5 == 0:
                    raw_chain = fetch_option_chain(ik, expiry)
                    if raw_chain:
                        oc_result = analyze_option_chain_data(raw_chain, ltp, idx_info['strike_gap'])
                        md['option_chain'] = oc_result['chain']
                        md['oi_analysis']  = oc_result['analysis']

                oi_analysis = md.get('oi_analysis', {'pcr': 1.0, 'signals': []})

                # 10. Fetch VIX every 5 iterations
                if iteration % 5 == 0:
                    vix = fetch_india_vix()
                    if vix:
                        md['vix'] = vix
                vix = md.get('vix', 0)

                # 11. Generate alerts → merge into history
                new_alerts = generate_alerts(
                    idx_name, md, oi_analysis, vwap_analysis, smc_zones, supports, resistances)
                for a in new_alerts:
                    alert_history[idx_name].appendleft(a)
                md['alerts'] = list(alert_history[idx_name])[:20]

                # 12. Trade suggestion
                md['trade_suggestion'] = generate_trade_suggestion(
                    idx_name, md, oi_analysis, vwap_analysis, smc_zones, supports, resistances, vix)

                # 13. Dashboard
                md['dashboard'] = calculate_dashboard(md, oi_analysis, vwap_analysis, vix)

                # 14. Institutional
                md['institutional'] = analyze_institutional(oi_analysis, smc_zones)

            except Exception as e:
                print(f"[ERROR] {idx_name} loop error: {e}")
                traceback.print_exc()

        # Emit to all connected clients
        try:
            payload = {
                'NIFTY':  serialize_market_data(market_data['NIFTY'],  'NIFTY'),
                'SENSEX': serialize_market_data(market_data['SENSEX'], 'SENSEX')
            }
            socketio.emit('market_update', payload)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Emitted market_update iter={iteration}")
        except Exception as e:
            print(f"[EMIT ERROR] {e}")

        iteration += 1
        socketio.sleep(15)  # gevent-friendly sleep

# ─────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(_BASE_DIR, 'index.html')

@app.route('/api/set_token', methods=['POST'])
def set_token():
    global ACCESS_TOKEN, THREAD_RUNNING, data_thread
    body  = request.get_json(force=True)
    token = body.get('token', '').strip()
    if not token:
        return jsonify({'success': False, 'error': 'Empty token'}), 400
    ACCESS_TOKEN = token
    if not THREAD_RUNNING:
        THREAD_RUNNING = True
        socketio.start_background_task(data_fetch_loop)
        print("[INFO] Data fetch background task started")
    return jsonify({'success': True})

@app.route('/api/status')
def status():
    return jsonify({'connected': THREAD_RUNNING, 'token_set': bool(ACCESS_TOKEN)})

# ─────────────────────────────────────────────
#  SocketIO events
# ─────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    print(f"[WS] Client connected: {request.sid}")
    try:
        emit('market_update', {
            'NIFTY':  serialize_market_data(market_data['NIFTY'],  'NIFTY'),
            'SENSEX': serialize_market_data(market_data['SENSEX'], 'SENSEX')
        })
    except Exception as e:
        print(f"[WS CONNECT ERROR] {e}")

@socketio.on('disconnect')
def on_disconnect():
    print(f"[WS] Client disconnected: {request.sid}")

@socketio.on('request_update')
def on_request_update():
    try:
        emit('market_update', {
            'NIFTY':  serialize_market_data(market_data['NIFTY'],  'NIFTY'),
            'SENSEX': serialize_market_data(market_data['SENSEX'], 'SENSEX')
        })
    except Exception as e:
        print(f"[WS UPDATE ERROR] {e}")

# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print(" Trading Intelligence System")
    print(f" Open http://localhost:{port}")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

# gunicorn uses `app` directly (see Procfile)