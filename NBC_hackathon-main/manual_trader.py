import websocket
import threading
import json
import requests
import time
import sys
import argparse
import ssl
import urllib3

# Suppress InsecureRequestWarning when using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ANSI Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
CYAN = '\033[96m'

class ManualTrader:
    def __init__(self, host="localhost:8080", scenario="normal_market", name="manual_trader", password=None, secure=False):
        self.host = host
        self.scenario = scenario
        self.secure = secure
        self.http_proto = "https" if secure else "http"
        self.ws_proto = "wss" if secure else "ws"
        self.student_id = name
        self.password = password
        self.token = None
        self.run_id = None
        self.ws_order = None
        self.ws_market = None
        self.running = True
        self.last_snapshot_hash = None
        
        # Tracking
        self.inventory = 0
        self.pnl = 0.0

    def register(self):
        print(f"Registering for scenario {self.scenario}...")
        try:
            url = f"{self.http_proto}://{self.host}/api/replays/{self.scenario}/start"
            headers = {"Authorization": f"Bearer {self.student_id}"}
            if self.password:
                headers["X-Team-Password"] = self.password
            resp = requests.get(
                url, 
                headers=headers,
                timeout=5,
                verify=not self.secure  # Disable SSL verification for self-signed certs
            )
            if resp.status_code != 200:
                print(f"{RED}Registration failed: {resp.text}{RESET}")
                return False
            
            data = resp.json()
            self.token = data.get("token")
            self.run_id = data.get("run_id")
            print(f"{GREEN}Registered! Run ID: {self.run_id}{RESET}")
            return True
        except Exception as e:
            print(f"{RED}Error registering: {e}{RESET}")
            return False

    def on_market_message(self, ws, message):
        data = json.loads(message)
        msg_type = data.get("type")
        
        if msg_type == "CONNECTED":
            print(f"\n{GREEN}[MARKET] Connected: {data.get('message')}{RESET}")
            print("> ", end="", flush=True)

        elif msg_type == "MARKET_DATA" or msg_type == "SNAPSHOT":
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            last_price = data.get("last_trade", 0.0)
            
            # Create a hash to check if book changed (simple string repr)
            # We want to update if ANY price/qty changes in top 10
            current_snapshot = json.dumps({
                "bids": bids[:10],
                "asks": asks[:10],
                "last": last_price,
                "step": data.get("step") 
            })
            
            # Always display if forced or changed
            if current_snapshot != self.last_snapshot_hash:
                self.display_book(bids, asks, last_price)
                self.last_snapshot_hash = current_snapshot
        
        else:
             # Debug unknown messages
             pass

    def display_book(self, bids, asks, last_price):
        # We'll just print a big block.
        
        print(f"\n{BLUE}================ MARKET SNAPSHOT ================{RESET}")
        print(f"Last Trade: {CYAN}{last_price:.2f}{RESET}")
        print(f"{'BID QTY':>10} | {'BID PRICE':>10} || {'ASK PRICE':<10} | {'ASK QTY':<10}")
        print("-" * 50)
        
        depth = 10
        for i in range(depth):
            bid_str = ""
            ask_str = ""
            
            # Bids (Green)
            if i < len(bids):
                b = bids[i]
                price = b.get('price', 0.0)
                qty = b.get('qty', 0)
                bid_str = f"{GREEN}{qty:>10} | {price:>10.2f}{RESET}"
            else:
                bid_str = f"{'':>10} | {'':>10}"
                
            # Asks (Red)
            if i < len(asks):
                a = asks[i]
                price = a.get('price', 0.0)
                qty = a.get('qty', 0)
                ask_str = f"{RED}{price:<10.2f} | {qty:<10}{RESET}"
            else:
                ask_str = f"{'':<10} | {'':<10}"
                
            print(f"{bid_str} || {ask_str}")
            
        print(f"{BLUE}================================================={RESET}")
        print(f"Inv: {self.inventory} | PnL: {self.pnl:.2f}")
        print("> ", end="", flush=True)

    def on_order_message(self, ws, message):
        data = json.loads(message)
        msg_type = data.get("type")
        
        if msg_type == "FILL":
            side = data.get("side")
            price = data.get("price")
            qty = data.get("qty")
            
            if side == "BUY":
                self.inventory += qty
                self.pnl -= qty * price 
            else:
                self.inventory -= qty
                self.pnl += qty * price
                
            color = GREEN if side == "BUY" else RED
            print(f"\n{color}*** FILL *** {side} {qty} @ {price:.2f}{RESET}")
            # Re-print prompt
            print("> ", end="", flush=True)
            
        elif msg_type == "ERROR":
            print(f"\n{RED}[ERROR] {data.get('message')}{RESET}")
            print("> ", end="", flush=True)

    def on_error(self, ws, error):
        if self.running:
            print(f"\n{RED}[WS ERROR] {error}{RESET}")

    def on_close(self, ws, status_code, msg):
        if self.running:
            print(f"\n{YELLOW}[WS CLOSED] {status_code}{RESET}")

    def start(self):
        if not self.register():
            return

        # SSL options for self-signed certs
        sslopt = {"cert_reqs": ssl.CERT_NONE} if self.secure else None

        # Market Data Connection
        market_url = f"{self.ws_proto}://{self.host}/api/ws/market?run_id={self.run_id}"
        self.ws_market = websocket.WebSocketApp(
            market_url,
            on_message=self.on_market_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Order Entry Connection
        order_url = f"{self.ws_proto}://{self.host}/api/ws/orders?token={self.token}&run_id={self.run_id}"
        self.ws_order = websocket.WebSocketApp(
            order_url,
            on_message=self.on_order_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        threading.Thread(target=lambda: self.ws_market.run_forever(sslopt=sslopt), daemon=True).start()
        threading.Thread(target=lambda: self.ws_order.run_forever(sslopt=sslopt), daemon=True).start()
        
        print("Connected.")
        print("Commands: buy/sell <qty> <price> | cancel <id> | step | quit")

        try:
            while self.running:
                try:
                    cmd_line = input() # Clean input via display_book prompt
                except EOFError:
                    break
                    
                parts = cmd_line.strip().split()
                if not parts:
                    continue
                    
                cmd = parts[0].lower()
                
                if cmd == "quit" or cmd == "exit":
                    self.running = False
                    break
                
                elif cmd in ["step", "next", "done"]:
                    msg = {"action": "DONE"}
                    self.ws_order.send(json.dumps(msg))
                    print("Sent STEP / DONE")

                elif cmd in ["buy", "sell"]:
                    if len(parts) < 3:
                        print("Usage: buy/sell <qty> <price>")
                        continue
                    try:
                        qty = int(parts[1])
                        price = float(parts[2])
                        side = "BUY" if cmd == "buy" else "SELL"
                        order_id = f"manual_{int(time.time()*1000)}"
                        msg = {
                            "type": "NEW_ORDER",
                            "order_id": order_id,
                            "ticker": "SYM",
                            "side": side,
                            "price": price,
                            "qty": qty
                        }
                        self.ws_order.send(json.dumps(msg))
                        print(f"Sent {side} {qty} @ {price} (ID: {order_id})")
                    except ValueError:
                        print("Invalid number")
                        
                elif cmd == "cancel":
                    if len(parts) < 2:
                        print("Usage: cancel <order_id>")
                        continue
                    order_id = parts[1]
                    msg = {"type": "CANCEL_ORDER", "order_id": order_id}
                    self.ws_order.send(json.dumps(msg))
                    print(f"Sent CANCEL {order_id}")
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if self.ws_market: self.ws_market.close()
            if self.ws_order: self.ws_order.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost:8080")
    parser.add_argument("--scenario", default="normal_market")
    parser.add_argument("--name", default="manual_trader", help="Your team name")
    parser.add_argument("--password", required=True, help="Your team password")
    parser.add_argument("--secure", action="store_true", help="Use HTTPS/WSS for secure connections")
    args = parser.parse_args()
    
    ManualTrader(args.host, args.scenario, args.name, args.password, args.secure).start()
