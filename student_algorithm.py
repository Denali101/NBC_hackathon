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
        spreads = np.array(self.spreads)
        times = np.array(self.timestamps)

        # Window stats (same window you already use via deques)
        window_start_time = times[0]
        window_seconds = max(1e-6, times[-1] - window_start_time)

        # tick_rate (ticks/sec)
        tick_rate = (len(times) - 1) / window_seconds

        # change_rate (changes/sec)
        changes = [c for (t, c) in self.change_flags if t >= window_start_time]
        change_rate = sum(changes) / window_seconds

        # churn: how often spread changes (0..1)
        churn = float(np.mean(np.diff(spreads) != 0.0)) if len(spreads) > 1 else 0.0

        # pct_min: % of window sitting at minimum spread (0..1)
        min_spread = float(np.min(spreads))
        pct_min = float(np.mean(spreads <= (min_spread + 1e-9)))
        # Returns / vol
        rets = np.diff(prices) / (prices[:-1] + 1e-9)
        vol = float(np.std(rets[-50:])) if len(rets) else 0.0

        # =========================
        # Crash detection: drawdown + rebound (robust)
        # =========================
        w = min(50, len(prices))
        recent = prices[-w:]
        peak = float(np.max(recent))
        trough = float(np.min(recent))

        drawdown = 0.0 if peak == 0 else (trough - peak) / peak  # negative
        rebound = 0.0 if (peak - trough) == 0 else (recent[-1] - trough) / (peak - trough)

        if drawdown < -0.03:
            if rebound < 0.20:
                return "flash_crash"
            if rebound > 0.60:
                return "mini_flash_crash"
            return "stressed_market"

        # =========================
        # Stressed: vol or spread blowout
        # =========================
        recent_spreads = spreads[-50:] if len(spreads) >= 50 else spreads
        spread_baseline = float(np.median(recent_spreads)) if len(recent_spreads) else 0.0
        last_spread = float(recent_spreads[-1]) if len(recent_spreads) else 0.0

        if vol > 0.006 or (spread_baseline > 0 and last_spread > 1.8 * spread_baseline):
            return "stressed_market"

        # =========================
        # HFT: high message rate + high quote churn + tight spread most of the time
        # =========================
        tick_rate = 0.0
        if len(self.recv_times) >= 10:
            dt = self.recv_times[-1] - self.recv_times[0]
            if dt > 0:
                tick_rate = (len(self.recv_times) - 1) / dt  # msgs per second

        # churn: how often the mid changes (proxy for quote updates)
        churn = float(np.mean(np.abs(np.diff(recent)) > 1e-9)) if len(recent) > 1 else 0.0

        min_spread = float(np.min(recent_spreads)) if len(recent_spreads) else 0.0
        pct_at_min = float(np.mean(recent_spreads <= (min_spread + 1e-9))) if len(recent_spreads) else 0.0
        spread_cv = float(np.std(recent_spreads) / (float(np.mean(recent_spreads)) + 1e-9)) if len(recent_spreads) else 0.0

        # DEBUG every so often (temporary)
        # if len(self.recv_times) >= 10 and len(self.mid_prices) % 50 == 0:
        #     print(f"[DEBUG] tick_rate={tick_rate:.1f}/s churn={churn:.2f} pct_min={pct_at_min:.2f} vol={vol:.5f}")
        # HFT: high message rate AND high change rate, with low volatility
        if churn > 0.20 and pct_min < 0.95 and vol < 0.002:
            return "hft_dominated"
        return "normal_market"


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

        # Market Regime Detector
        self.regime_detector = MarketRegimeDetector(window=100)
        self.current_market_type = "normal_market"
        self.open_orders = 0


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
            order = self.decide_order()

            # ===== Send order if any =====
            if order is not None and self.order_ws and self.order_ws.sock:
                self._send_order(order)
                self.open_orders += 1
            # Signal DONE to advance to next step
            self._send_done()
            
        except Exception as e:
            print(f"[{self.student_id}] Market data error: {e}")
    
    # =========================================================================
    # YOUR STRATEGY - MODIFY THIS METHOD!
    # =========================================================================

    def decide_order(self, bid: float, ask: float, mid: float) -> Optional[Dict]:
        """
        ╔══════════════════════════════════════════════════════════════════╗
        ║                    YOUR STRATEGY GOES HERE!                       ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  Input:                                                           ║
        ║    - bid: Best bid price                                          ║
        ║    - ask: Best ask price                                          ║
        ║    - mid: Mid price (average of bid and ask)                      ║
        ║                                                                   ║
        ║  Available state:                                                 ║
        ║    - self.inventory: Your current position                         ║
        ║    - self.pnl: Your realized PnL                                  ║
        ║    - self.current_step: Current simulation step                   ║
        ║                                                                   ║
        ║  Return:                                                          ║
        ║    - {"side": "BUY"|"SELL", "price": X, "qty": N}                 ║
        ║    - Or return None to not send an order                          ║
        ╚══════════════════════════════════════════════════════════════════╝
        """

        # Skip if no valid prices
        if mid <= 0 or bid <= 0 or ask <= 0:
            return None

        # =================================================================
        # Normal_market Strategy:
        # 1. Calculate Reservation Price (Inventory Skew):
        #    - Shift the pricing "center" to encourage inventory balancing.
        #    - Formula: Reserve_Price = Mid_Price - (Inventory * Skew_Factor)
        #    - Result: If holding Long, quotes shift down (Sell faster, Buy lower).
        #              If holding Short, quotes shift up (Buy faster, Sell higher).
        #
        # 2. Quote Dynamic Spreads:
        #    - Calculate Bid/Ask around the Reservation Price, not the Mid.
        #    - My_Bid = Reserve_Price - (Half_Spread)
        #    - My_Ask = Reserve_Price + (Half_Spread)
        #
        #
        #
        # 4. Order Execution:
        #    - If Inventory is safe (<500): Quote both sides (or alternate).
        #    - If Inventory is heavy (>500): Prioritize the order that reduces risk.
        # =================================================================

        # =================================================================
        # HARD SAFETY CAP
        # The rules say 5000 is the limit.
        # We stop buying at 4800 to account for any "in-flight" delays.
        # =================================================================

        if self.current_step % 5 != 0:
            return None

        # 1. Panic Sell (Too much inventory)
        if self.inventory >= 4800:
            # FORCE SELL: Ignore the skew, just sell below market to get out
            return {"side": "SELL", "price": round(bid, 2), "qty": 100}

        # 2. Panic Buy (Too much shorting)
        if self.inventory <= -4800:
            # FORCE BUY: Just buy at ask to cover
            return {"side": "BUY", "price": round(ask, 2), "qty": 100}

        # Only send orders every few steps to avoid overwhelming the system

        mid = round(mid, 2)
        skew = 0.001
        current_spread = 0.02

        reservation_price = mid - (self.inventory * skew)
        my_bid = round(reservation_price - (current_spread / 2), 2)
        my_ask = round(reservation_price + (current_spread / 2), 2)


        # 4 possibilities:
        # i) Low/negative inventory => buy aggressively
        # ii) Nearing limit of 5k => sell aggressively
        # otherwise alternate iii) buying 0.05 below mid and iv) selling 0.05 above mid
        # python student_algorithm.py --name Quackonomics --password Pegg3d_M1dpoint_Ordr$ --scenario normal_market  --host 3.98.52.120:8433 --secure


        if self.inventory > 200:
            return {"side": "SELL", "price": round(bid, 2), "qty": 100}

        elif self.inventory < -200:
            return {"side": "BUY", "price": round(ask, 2), "qty": 100}

        elif (self.current_step // 5) % 2 == 0:
            return {"side": "BUY","price": my_bid,"qty": 100}
        else:
            return {"side": "SELL","price": my_ask,"qty": 100}
    '''
    ELIZA ELIZA ELIZA ELIZA
    def decide_order(self, *args):
        # Support both call styles:
        # - decide_order() uses self.last_bid/self.last_ask
        # - decide_order(bid, ask) or decide_order(bid, ask, mid)
        if len(args) >= 2:
            bid = args[0]
            ask = args[1]
        else:
            bid = getattr(self, "last_bid", None)
            ask = getattr(self, "last_ask", None)

        if bid is None or ask is None:
            return None

        bid = float(bid)
        ask = float(ask)
        if bid <= 0 or ask <= 0 or ask <= bid:
            return None

        # Required state
        if not hasattr(self, "open_orders"):
            self.open_orders = 0
        if not hasattr(self, "inventory"):
            self.inventory = 0
        if not hasattr(self, "_last_order_step"):
            self._last_order_step = -10 ** 9

        spread = ask - bid
        mid = (bid + ask) / 2.0
        tick = 0.10
        qty = 100
        max_inv = 600

        mkt = getattr(self, "current_market_type", "normal_market")

        # ---------- Hard safety ----------
        # If you don't have cancel logic, spamming = guaranteed rate-limit.
        # Keep at most 1 outstanding order at a time.
        if self.open_orders >= 1:
            return None

        # Don't place orders too frequently
        # (reduces adverse selection + avoids open-order buildup)
        cooldown = 8 if mkt == "hft_dominated" else 4
        if (self.current_step - self._last_order_step) < cooldown:
            return None

        # If inventory is too large, ONLY unwind (do not "trade")
        if abs(self.inventory) >= max_inv:
            if self.inventory > 0:
                self._last_order_step = self.current_step
                return {"side": "SELL", "price": bid, "qty": qty}
            else:
                self._last_order_step = self.current_step
                return {"side": "BUY", "price": ask, "qty": qty}

        # ---------- Regime rules ----------
        # Crash/stress: stop market making, only flatten risk
        if mkt in ("stressed_market", "mini_flash_crash", "flash_crash"):
            if self.inventory > 0:
                self._last_order_step = self.current_step
                return {"side": "SELL", "price": bid, "qty": qty}
            if self.inventory < 0:
                self._last_order_step = self.current_step
                return {"side": "BUY", "price": ask, "qty": qty}
            return None

        # HFT-dominated: if spread is tight, DO NOT play (fees + adverse selection kill you)
        # Only consider trading when there's real edge (spread wide enough)
        if mkt == "hft_dominated":
            if spread < 3 * tick:
                return None

            # If flat, be picky: place ONE passive order at best price, not inside
            # (inside quotes get picked off fast)
            prev_mid = getattr(self, "_prev_mid", mid)
            mom = mid - prev_mid
            self._prev_mid = mid

            if self.inventory == 0:
                if mom > 0:
                    self._last_order_step = self.current_step
                    return {"side": "SELL", "price": ask, "qty": qty}
                elif mom < 0:
                    self._last_order_step = self.current_step
                    return {"side": "BUY", "price": bid, "qty": qty}
                else:
                    return None

            # If holding inventory, unwind at touch
            if self.inventory > 0:
                self._last_order_step = self.current_step
                return {"side": "SELL", "price": ask, "qty": qty}
            else:
                self._last_order_step = self.current_step
                return {"side": "BUY", "price": bid, "qty": qty}

        # Normal market: simple “buy bid / sell ask” inventory-leaning
        # Flat: alternate sides based on tiny momentum to avoid deadlock
        prev_mid = getattr(self, "_prev_mid", mid)
        mom = mid - prev_mid
        self._prev_mid = mid

        if self.inventory > 0:
            self._last_order_step = self.current_step
            return {"side": "SELL", "price": ask, "qty": qty}
        if self.inventory < 0:
            self._last_order_step = self.current_step
            return {"side": "BUY", "price": bid, "qty": qty}

        if mom > 0:
            self._last_order_step = self.current_step
            return {"side": "SELL", "price": ask, "qty": qty}
        if mom < 0:
            self._last_order_step = self.current_step
            return {"side": "BUY", "price": bid, "qty": qty}

        return None
    '''

    # =========================================================================
    # ORDER HANDLING
    # =========================================================================

    def _send_order(self, order: Dict):
        """Send an order to the exchange."""
        order_id = f"ORD_{self.student_id}_{self.current_step}_{self.orders_sent}"

        msg = {
            "order_id": order_id,
            "side": order["side"],
            "price": order["price"],
            "qty": order["qty"]
        }

        try:
            self.order_send_times[order_id] = time.time()  # Track send time
            self.order_ws.send(json.dumps(msg))
            self.open_orders += 1
            self.orders_sent += 1
        except Exception as e:
            print(f"[{self.student_id}] Send order error: {e}")

    def _send_done(self):
        """Signal DONE to advance to the next simulation step."""
        try:
            self.order_ws.send(json.dumps({"action": "DONE"}))
            self.last_done_time = time.time()  # Track when we sent DONE
        except:
            pass

    def _on_order_response(self, ws, message: str):
        """Handle order responses and fills."""
        try:
            recv_time = time.time()
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "AUTHENTICATED":
                print(f"[{self.student_id}] Authenticated - ready to trade!")

            elif msg_type == "FILL":
                qty = data.get("qty", 0)
                price = data.get("price", 0)
                side = data.get("side", "")
                order_id = data.get("order_id", "")

                # Measure fill latency
                if order_id in self.order_send_times:
                    fill_latency = (recv_time - self.order_send_times[order_id]) * 1000  # ms
                    self.fill_latencies.append(fill_latency)
                    del self.order_send_times[order_id]

                # Update inventory and cash flow
                if side == "BUY":
                    self.inventory += qty
                    self.cash_flow -= qty * price  # Spent cash to buy
                else:
                    self.inventory -= qty
                    self.cash_flow += qty * price  # Received cash from selling

                # Calculate mark-to-market PnL using mid price
                self.pnl = self.cash_flow + self.inventory * self.last_mid
                self.open_orders = max(0, self.open_orders - 1)
                print(f"[{self.student_id}] FILL: {side} {qty} @ {price:.2f} | Inventory: {self.inventory} | PnL: {self.pnl:.2f}")

            elif msg_type == "ERROR":
                print(f"[{self.student_id}] ERROR: {data.get('message')}")

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
