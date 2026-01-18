"""
Student Trading Algorithm Template
===================================
Connect to the exchange simulator, receive market data, and submit orders.

    python student_algorithm.py --host ip:host --scenario normal_market --name your_name --password your_password --secure

YOUR TASK:
    Modify the `decide_order()` method to implement your trading strategy.
"""

import json
import uuid

import websocket
import threading
import argparse
import time
import requests
import ssl
import urllib3
from typing import Dict, Optional

# Suppress SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import numpy as np
from collections import deque
import time

class MarketRegimeDetector:
    def __init__(self, window=200):
        self.mid_prices = deque(maxlen=window)
        self.spreads = deque(maxlen=window)

        # timestamps for the mid/spread series
        self.timestamps = deque(maxlen=window)

        # message receive timestamps (optional, for tick-rate)
        self.recv_times = deque(maxlen=window)

        # (timestamp, 0/1 did mid or spread change)
        self.change_flags = deque(maxlen=window)

        self._last_mid = None
        self._last_spread = None

    def update(self, bid, ask, recv_time=None):
        # Ignore only totally empty quotes
        if bid <= 0 and ask <= 0:
            return

        now = time.time()

        # Robust mid/spread with partial quotes
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            spread = ask - bid
        else:
            mid = bid if bid > 0 else ask
            spread = self._last_spread if self._last_spread is not None else 0.0

        changed = 0
        if self._last_mid is not None:
            if mid != self._last_mid or spread != self._last_spread:
                changed = 1

        self._last_mid = mid
        self._last_spread = spread

        self.mid_prices.append(mid)
        self.spreads.append(spread)
        self.timestamps.append(now)
        self.change_flags.append((now, changed))

        if recv_time is not None:
            self.recv_times.append(recv_time)

    def classify(self):
        if len(self.mid_prices) < 30:
            return "normal_market"

        prices = np.array(self.mid_prices, dtype=float)
        spreads = np.array(self.spreads, dtype=float)
        times = np.array(self.timestamps, dtype=float)

        window_seconds = max(1e-6, times[-1] - times[0])

        # === Volatility
        rets = np.diff(prices) / (prices[:-1] + 1e-9)
        vol = float(np.std(rets[-50:])) if len(rets) else 0.0

        # === Spread stats
        recent_spreads = spreads[-50:] if len(spreads) >= 50 else spreads
        spread_med = float(np.median(recent_spreads)) if len(recent_spreads) else 0.0
        spread_cv = np.std(recent_spreads) / (spread_med + 1e-9) if len(recent_spreads) else 0.0
        last_spread = float(recent_spreads[-1]) if len(recent_spreads) else 0.0

        # === Tick + churn
        tick_rate = (len(times) - 1) / window_seconds
        changes = sum(c for (_, c) in self.change_flags)
        change_rate = changes / window_seconds

        # === Flash crash detection
        recent = prices[-50:]
        peak_idx = np.argmax(recent)
        trough_idx = np.argmin(recent)
        peak = recent[peak_idx]
        trough = recent[trough_idx]

        drawdown = 0.0 if peak == 0 else (trough - peak) / peak
        drop_speed = abs(trough_idx - peak_idx) / len(recent)  # normalized time
        rebound = 0.0
        if trough_idx < len(recent) - 1:
            rebound = (recent[-1] - trough) / (peak - trough + 1e-9)

        # === Classify crash regimes
        if drawdown < -0.015 and drop_speed < 0.7:
            if rebound < 0.25:
                new_regime = "flash_crash"
            elif rebound > 0.60:
                new_regime = "mini_flash_crash"
            else:
                new_regime = "stressed_market"
        elif (
                vol > 0.003
                or spread_cv > 0.35
                or (spread_med > 0 and last_spread > 1.5 * spread_med)
                or change_rate > 10
        ):
            new_regime = "stressed_market"
        else:
            # HFT detection
            churn = float(np.mean(np.abs(np.diff(prices[-15:])) > 1e-9)) if len(prices) > 15 else 0.0
            min_spread = float(np.min(recent_spreads)) if len(recent_spreads) else 0.0
            pct_min = float(np.mean(recent_spreads <= (min_spread + 1e-9))) if len(recent_spreads) else 0.0

            if tick_rate > 15 and churn > 0.2 and pct_min < 0.9 and vol < 0.002:
                new_regime = "hft_dominated"
            else:
                new_regime = "normal_market"

        # === Debug print
        if hasattr(self, "_last_regime"):
            if new_regime != self._last_regime:
                print(f"[Regime Shift] {self._last_regime} → {new_regime}")
            self._last_regime = new_regime
        else:
            self._last_regime = new_regime

        return new_regime


class TradingBot:
    """
    A trading bot that connects to the exchange simulator.
    
    Students should modify the `decide_order()` method to implement their strategy.
    """
    
    def __init__(self, student_id: str, host: str, scenario: str, password: str = None, secure: bool = False):
        self.student_id = student_id
        self.host = host
        self.scenario = scenario
        self.password = password
        self.secure = secure
        
        # Protocol configuration
        self.http_proto = "https" if secure else "http"
        self.ws_proto = "wss" if secure else "ws"
        
        # Session info (set after registration)
        self.token = None
        self.run_id = None
        
        # Trading state - track your position
        self.inventory = 0      # Current position (positive = long, negative = short)
        self.cash_flow = 0.0    # Cumulative cash from trades (negative when buying)
        self.pnl = 0.0          # Mark-to-market PnL (cash_flow + inventory * mid_price)
        self.current_step = 0   # Current simulation step
        self.orders_sent = 0    # Number of orders sent
        
        # Market data
        self.last_bid = 0.0
        self.last_ask = 0.0
        self.last_mid = 0.0
        
        # WebSocket connections
        self.market_ws = None
        self.order_ws = None
        self.running = True
        
        # Latency measurement
        self.last_done_time = None          # When we sent DONE
        self.step_latencies = []            # Time between DONE and next market data
        self.order_send_times = {}          # order_id -> time sent
        self.fill_latencies = []            # Time between order and fill
        self.last_order_time = 0.0
        
        self.last_mini_trade_time = 0


        # Market Regime Detector
        self.regime_detector = MarketRegimeDetector(window=100)
        self.current_market_type = "normal_market"
        self.open_orders = 0
        # Strategy state (MUST exist before trading starts)
        self.open_orders = 0
        self.last_inventory = 0
        self.has_open_order = False

        self.last_hft_trade_step = -100
        self.HFT_COOLDOWN = 15  # steps
        self.HFT_MAX_INV = 300
        self.hft_entry_price = None
        self.hft_entry_step = None
        self.HFT_HOLD_LIMIT = 120  # steps
        self.HFT_EXIT_EDGE = 0.02
        self.last_hft_trade_step = -10_000
        self.pending_buys = 0
        self.pending_sells = 0


    # =========================================================================
    # REGISTRATION - Get a token to start trading
    # =========================================================================
    
    def register(self) -> bool:
        """Register with the server and get an auth token."""
        print(f"[{self.student_id}] Registering for scenario '{self.scenario}'...")
        try:
            url = f"{self.http_proto}://{self.host}/api/replays/{self.scenario}/start"
            headers = {"Authorization": f"Bearer {self.student_id}"}
            if self.password:
                headers["X-Team-Password"] = self.password
            resp = requests.get(
                url,
                headers=headers,
                timeout=10,
                verify=not self.secure  # Disable SSL verification for self-signed certs
            )
            
            if resp.status_code != 200:
                print(f"[{self.student_id}] Registration FAILED: {resp.text}")
                return False
            
            data = resp.json()
            self.token = data.get("token")
            self.run_id = data.get("run_id")
            
            if not self.token or not self.run_id:
                print(f"[{self.student_id}] Missing token or run_id")
                return False
            
            print(f"[{self.student_id}] Registered! Run ID: {self.run_id}")
            return True
            
        except Exception as e:
            print(f"[{self.student_id}] Registration error: {e}")
            return False
    
    # =========================================================================
    # CONNECTION - Connect to WebSocket streams
    # =========================================================================
    
    def connect(self) -> bool:
        """Connect to market data and order entry WebSockets."""
        try:
            # SSL options for self-signed certificates
            sslopt = {"cert_reqs": ssl.CERT_NONE} if self.secure else None
            
            # Market Data WebSocket
            market_url = f"{self.ws_proto}://{self.host}/api/ws/market?run_id={self.run_id}"
            self.market_ws = websocket.WebSocketApp(
                market_url,
                on_message=self._on_market_data,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Market data connected")
            )
            
            # Order Entry WebSocket
            order_url = f"{self.ws_proto}://{self.host}/api/ws/orders?token={self.token}&run_id={self.run_id}"
            self.order_ws = websocket.WebSocketApp(
                order_url,
                on_message=self._on_order_response,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Order entry connected")
            )
            
            # Start WebSocket threads
            threading.Thread(
                target=lambda: self.market_ws.run_forever(sslopt=sslopt),
                daemon=True
            ).start()
            
            threading.Thread(
                target=lambda: self.order_ws.run_forever(sslopt=sslopt),
                daemon=True
            ).start()
            
            # Wait for connections
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"[{self.student_id}] Connection error: {e}")
            return False
    
    # =========================================================================
    # MARKET DATA HANDLER - Called when new market data arrives
    # =========================================================================
    
    def _on_market_data(self, ws, message: str):
        """Handle incoming market data snapshot."""
        try:
            recv_time = time.time()
            data = json.loads(message)
            
            # Skip connection confirmation messages
            if data.get("type") == "CONNECTED":
                return
            step_latency = None
            # Measure step latency (time since we sent DONE)
            if self.last_done_time is not None:
                step_latency = (recv_time - self.last_done_time) * 1000  # ms
                self.step_latencies.append(step_latency)
            
            # Extract market data
            self.current_step = data.get("step", 0)
            self.last_bid = data.get("bid", 0.0)
            self.last_ask = data.get("ask", 0.0)
            
            # Log progress every 500 steps with latency stats
            if self.current_step % 500 == 0 and self.step_latencies:
                avg_lat = sum(self.step_latencies[-100:]) / min(len(self.step_latencies), 100)
                print(f"[{self.student_id}] Step {self.current_step} | Orders: {self.orders_sent} | Inv: {self.inventory} | Avg Latency: {avg_lat:.1f}ms")
                print(f"[{self.student_id}] Step {self.current_step} | Orders: {self.orders_sent} | Inv: {self.inventory} | Avg Latency: {avg_lat:.1f}ms")

            # Calculate mid price
            if self.last_bid > 0 and self.last_ask > 0:
                self.last_mid = (self.last_bid + self.last_ask) / 2
            elif self.last_bid > 0:
                self.last_mid = self.last_bid
            elif self.last_ask > 0:
                self.last_mid = self.last_ask
            else:
                self.last_mid = 0
            # =====================================================
            # MARKET REGIME DETECTION (NEW)
            # =====================================================
            self.regime_detector.update(self.last_bid, self.last_ask, recv_time=recv_time)
            self.current_market_type = self.regime_detector.classify()

            if self.current_step % 200 == 0:
                print(f"[{self.student_id}] Step {self.current_step} | Market: {self.current_market_type}")
            # =============================================
            # YOUR STRATEGY LOGIC GOES HERE
            # =============================================
            order = self.decide_order(
                self.last_bid,
                self.last_ask,
                self.last_mid
            )

            # ===== Send order if any =====
            if order is not None and self.order_ws and self.order_ws.sock:
                self._send_order(order)

            # Signal DONE to advance to next step
            try:
                if self.order_ws and self.order_ws.sock:
                    self.order_ws.send(json.dumps({"action": "DONE"}))
                    self.last_done_time = time.time()
            except:
                pass


        except Exception as e:
            print(f"[{self.student_id}] Market data error: {e}")
    
    # =========================================================================
    # YOUR STRATEGY - MODIFY THIS METHOD!
    # =========================================================================

    def decide_order(self, bid: float, ask: float, mid: float):
        if bid <= 0 or ask <= 0:
            return None

        # === SMART POSITION RISK LAYER ===
        SOFT_LIMIT = 4500
        HARD_LIMIT = 5000
        QTY = 100

        effective_inventory = self.inventory + self.pending_buys - self.pending_sells
        self.too_close_to_limit = abs(effective_inventory) >= SOFT_LIMIT

        # Absolute hard limit: emergency flatten
        if effective_inventory >= HARD_LIMIT:
            return {"side": "SELL", "price": round(bid, 2), "qty": QTY}
        elif effective_inventory <= -HARD_LIMIT:
            return {"side": "BUY", "price": round(ask, 2), "qty": QTY}

        # ===== ABSOLUTE SAFETY =====
        if self.has_open_order and self.current_market_type != "hft_dominated":
            return None

        # ===== PARAMETER SELECTION =====
        regime = self.current_market_type
       
        if regime == "flash_crash":
            QTY = 600
            EDGE = 0.015
            MAX_POS = 1000

        elif regime == "mini_flash_crash":
            QTY = 300
            EDGE = 0.035
            MAX_POS = 2400

            now = time.time()
            if now - self.last_mini_trade_time < 0.25:
                return None

            self.last_mini_trade_time = now
            if (ask - bid) < EDGE * 2:
                    return None
        elif regime == "hft_dominated":
            BASE_QTY = 100
            MAX_QTY = 200
            EDGE = 0.01
            COOLDOWN = 5
            HOLD_LIMIT = 20
            SLIPPAGE_LIMIT = 0.004
            HARD_LIMIT = 5000
            REENTRY_DELAY = 10

            # === Effective inventory ===
            effective_pos = self.inventory + self.pending_buys - self.pending_sells

            # === Init state ===
            if not hasattr(self, "hft_state"):
                self.hft_state = {
                    "mid_history": deque(maxlen=10),
                    "last_entry_step": -1000,
                    "entry_price": None,
                    "direction": None,
                    "trend_age": 0,
                }

            if not hasattr(self, "hft_last_exit_step"):
                self.hft_last_exit_step = -1000

            # === Cooldown / reentry delay ===
            if (self.current_step - self.hft_state["last_entry_step"] < COOLDOWN or
                    self.current_step - self.hft_last_exit_step < REENTRY_DELAY):
                return None

            # === Update trend memory ===
            self.hft_state["mid_history"].append(mid)
            history = list(self.hft_state["mid_history"])
            if len(history) < 6:
                return None

            trend = history[-1] - history[0]
            trend_confidence = abs(trend)
            rand_factor = np.random.rand()
            in_position = self.inventory != 0

            # === Dynamically scale QTY based on trend confidence ===
            scaled_qty = BASE_QTY if trend_confidence < 0.005 else MAX_QTY

            # === ENTRY ===
            if not in_position:
                if trend > EDGE and rand_factor > 0.4 and effective_pos + scaled_qty <= HARD_LIMIT and not self.too_close_to_limit:
                    self.hft_state.update({
                        "last_entry_step": self.current_step,
                        "entry_price": mid,
                        "direction": "LONG",
                        "trend_age": 0
                    })
                    return {
                        "side": "BUY",
                        "price": round(bid + EDGE, 2),
                        "qty": scaled_qty
                    }

                elif trend < -EDGE and rand_factor > 0.4 and effective_pos - scaled_qty >= -HARD_LIMIT and not self.too_close_to_limit:
                    self.hft_state.update({
                        "last_entry_step": self.current_step,
                        "entry_price": mid,
                        "direction": "SHORT",
                        "trend_age": 0
                    })
                    return {
                        "side": "SELL",
                        "price": round(ask - EDGE, 2),
                        "qty": scaled_qty
                    }

            # === EXIT ===
            else:
                direction = self.hft_state["direction"]
                entry_price = self.hft_state["entry_price"]
                time_held = self.current_step - self.hft_state["last_entry_step"]
                self.hft_state["trend_age"] += 1

                exit_qty = BASE_QTY
                exit_cond = False

                if direction == "LONG":
                    gain = mid - entry_price
                    loss = entry_price - mid
                    if gain >= EDGE or time_held > HOLD_LIMIT or loss > SLIPPAGE_LIMIT:
                        exit_cond = True

                elif direction == "SHORT":
                    gain = entry_price - mid
                    loss = mid - entry_price
                    if gain >= EDGE or time_held > HOLD_LIMIT or loss > SLIPPAGE_LIMIT:
                        exit_cond = True

                if exit_cond:
                    self.hft_last_exit_step = self.current_step
                    self.hft_state["last_entry_step"] = self.current_step
                    side = "SELL" if direction == "LONG" else "BUY"
                    price = round(ask - EDGE, 2) if direction == "LONG" else round(bid + EDGE, 2)

                    if (side == "SELL" and effective_pos - exit_qty >= -HARD_LIMIT) or \
                            (side == "BUY" and effective_pos + exit_qty <= HARD_LIMIT):
                        return {"side": side, "price": price, "qty": exit_qty}

            return None
        elif regime == "stressed_market":
            BASE_QTY = 100
            EDGE = 0.02
            HARD_LIMIT = 5000
            TRADE_EVERY = 4  # throttle: only trade every 4 steps

            effective_pos = self.inventory + self.pending_buys - self.pending_sells

            # Step throttle
            if self.current_step % TRADE_EVERY != 0:
                return None

            # Init mid history
            if not hasattr(self, "stressed_mid_history"):
                self.stressed_mid_history = deque(maxlen=15)
            if not hasattr(self, "last_stressed_trade_step"):
                self.last_stressed_trade_step = -1000

            self.stressed_mid_history.append(mid)
            if len(self.stressed_mid_history) < 10:
                return None

            recent_avg = sum(self.stressed_mid_history) / len(self.stressed_mid_history)
            deviation = mid - recent_avg

            # Emergency flatten if beyond hard limits
            if effective_pos >= HARD_LIMIT:
                return {"side": "SELL", "price": round(bid, 2), "qty": BASE_QTY}
            if effective_pos <= -HARD_LIMIT:
                return {"side": "BUY", "price": round(ask, 2), "qty": BASE_QTY}

            # === Mean-reversion fade logic ===
            if deviation < -EDGE and effective_pos + BASE_QTY <= HARD_LIMIT and not self.too_close_to_limit:
                self.last_stressed_trade_step = self.current_step
                return {"side": "BUY", "price": round(bid + 0.01, 2), "qty": BASE_QTY}

            elif deviation > EDGE and effective_pos - BASE_QTY >= -HARD_LIMIT and not self.too_close_to_limit:
                self.last_stressed_trade_step = self.current_step
                return {"side": "SELL", "price": round(ask - 0.01, 2), "qty": BASE_QTY}

            # === Passive unwind occasionally ===
            if abs(self.inventory) > 0 and np.random.rand() > 0.97:
                side = "SELL" if self.inventory > 0 else "BUY"
                price = round(bid + 0.01, 2) if side == "BUY" else round(ask - 0.01, 2)
                self.last_stressed_trade_step = self.current_step
                return {"side": side, "price": price, "qty": BASE_QTY}

            return None
        elif regime == "normal_market":
            BASE_QTY = 100
            EDGE = 0.015
            HARD_LIMIT = 5000
            TRADE_EVERY = 3  # throttle frequency

            effective_pos = self.inventory + self.pending_buys - self.pending_sells

            if self.current_step % TRADE_EVERY != 0:
                return None

            # Emergency flatten
            if effective_pos >= HARD_LIMIT:
                return {"side": "SELL", "price": round(bid, 2), "qty": BASE_QTY}
            if effective_pos <= -HARD_LIMIT:
                return {"side": "BUY", "price": round(ask, 2), "qty": BASE_QTY}

            # === Trend bias helper ===
            if not hasattr(self, "trend_buffer"):
                self.trend_buffer = deque(maxlen=10)
            self.trend_buffer.append(mid)
            if len(self.trend_buffer) < self.trend_buffer.maxlen:
                return None

            avg_mid = sum(self.trend_buffer) / len(self.trend_buffer)
            trend = mid - avg_mid  # positive = upward trend

            # === Buy low logic ===
            if bid < avg_mid - EDGE and not self.too_close_to_limit and effective_pos + BASE_QTY <= HARD_LIMIT:
                return {"side": "BUY", "price": round(bid + 0.01, 2), "qty": BASE_QTY}

            # === Sell high logic ===
            if ask > avg_mid + EDGE and not self.too_close_to_limit and effective_pos - BASE_QTY >= -HARD_LIMIT:
                return {"side": "SELL", "price": round(ask - 0.01, 2), "qty": BASE_QTY}

            # === Flatten passively ===
            if abs(self.inventory) > 0 and np.random.rand() > 0.98:
                side = "SELL" if self.inventory > 0 else "BUY"
                price = round(ask - 0.01, 2) if side == "SELL" else round(bid + 0.01, 2)
                return {"side": side, "price": price, "qty": BASE_QTY}

            return None

        # ===== FORCE FLATTEN =====
        if self.inventory > MAX_POS:
            return {
                "side": "SELL",
                "price": round(bid, 2),
                "qty": QTY
            }

        if self.inventory < -MAX_POS:
            return {
                "side": "BUY",
                "price": round(ask, 2),
                "qty": QTY
            }

        # ===== PURE FADE / SPREAD CAPTURE =====
        if self.inventory <= 0:
            # Buy slightly inside bid
            return {
                "side": "BUY",
                "price": round(bid + EDGE, 2),
                "qty": QTY
            }

        # Sell slightly inside ask
        return {
            "side": "SELL",
            "price": round(ask - EDGE, 2),
            "qty": QTY
        }





    # =========================================================================
    # ORDER HANDLING
    # =========================================================================

    def _send_order(self, order):
        order_id = f"ORD_{self.student_id}_{self.current_step}_{self.orders_sent}"

        msg = {
            "order_id": order_id,
            "side": order["side"],
            "price": order["price"],
            "qty": order["qty"]
        }

        self.order_ws.send(json.dumps(msg))
        self.has_open_order = True
        self.orders_sent += 1
        self.order_send_times[order_id] = time.time()

        # === Track pending inventory ===
        if order["side"] == "BUY":
            self.pending_buys += order["qty"]
        else:
            self.pending_sells += order["qty"]

        print(f"[{self.student_id}] Sent order: {msg}")

    def _on_order_response(self, ws, message: str):
        """Handle order responses and fills."""
        try:
            recv_time = time.time()
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "AUTHENTICATED":
                print(f"[{self.student_id}] Authenticated - ready to trade!")

            elif msg_type == "FILL":
                self.has_open_order = False

                qty = data["qty"]
                price = data["price"]

                if data["side"] == "BUY":
                    self.pending_buys -= qty
                else:
                    self.pending_sells -= qty
                if data["side"] == "BUY":
                    self.inventory += qty
                    self.cash_flow -= qty * price
                else:
                    self.inventory -= qty
                    self.cash_flow += qty * price

                self.pnl = self.cash_flow + self.inventory * self.last_mid
                print(f"FILL {data['side']} {qty} @ {price} | Inv {self.inventory} | PnL {self.pnl:.2f}")
                # Measure fill latency
                order_id = data.get("order_id")     
                if order_id in self.order_send_times:
                    send_time = self.order_send_times.pop(order_id)
                    fill_latency = (recv_time - send_time) * 1000  # ms
                    self.fill_latencies.append(fill_latency)
            elif msg_type == "REJECTED":
                self.has_open_order = False
                print(f"[{self.student_id}] Order REJECTED: {data}")
        except Exception as e:
            print(f"[{self.student_id}] Order response error: {e}")



    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    def _on_error(self, ws, error):
        if self.running:
            print(f"[{self.student_id}] WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        self.running = False
        print(f"[{self.student_id}] Connection closed (status: {close_status_code})")
    
    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================
    
    def run(self):
        """Main entry point - register, connect, and run."""
        # Step 1: Register
        if not self.register():
            return
        
        # Step 2: Connect
        if not self.connect():
            return
        
        # Step 3: Run until complete
        print(f"[{self.student_id}] Running... Press Ctrl+C to stop")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n[{self.student_id}] Stopped by user")
        finally:
            self.running = False
            if self.market_ws:
                self.market_ws.close()
            if self.order_ws:
                self.order_ws.close()
            
            print(f"\n[{self.student_id}] Final Results:")
            print(f"  Orders Sent: {self.orders_sent}")
            print(f"  Inventory: {self.inventory}")
            print(f"  PnL: {self.pnl:.2f}")
            
            # Print latency statistics
            if self.step_latencies:
                print(f"\n  Step Latency (ms):")
                print(f"    Min: {min(self.step_latencies):.1f}")
                print(f"    Max: {max(self.step_latencies):.1f}")
                print(f"    Avg: {sum(self.step_latencies)/len(self.step_latencies):.1f}")
            
            if self.fill_latencies:
                print(f"\n  Fill Latency (ms):")
                print(f"    Min: {min(self.fill_latencies):.1f}")
                print(f"    Max: {max(self.fill_latencies):.1f}")
                print(f"    Avg: {sum(self.fill_latencies)/len(self.fill_latencies):.1f}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Student Trading Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local server:
    python student_algorithm.py --name team_alpha --password secret123 --scenario normal_market
    
  Deployed server (HTTPS):
    python student_algorithm.py --name team_alpha --password secret123 --scenario normal_market --host 3.98.52.120:8433 --secure
        """
    )
    
    parser.add_argument("--name", required=True, help="Your team name")
    parser.add_argument("--password", required=True, help="Your team password")
    parser.add_argument("--scenario", default="normal_market", help="Scenario to run")
    parser.add_argument("--host", default="localhost:8080", help="Server host:port")
    parser.add_argument("--secure", action="store_true", help="Use HTTPS/WSS (for deployed servers)")
    args = parser.parse_args()
    
    bot = TradingBot(
        student_id=args.name,
        host=args.host,
        scenario=args.scenario,
        password=args.password,
        secure=args.secure
    )
    
    bot.run()
