# Trading Algorithm Template

A Python template for building trading algorithms that connect to the exchange simulator.

## Quick Start
Available scenarios
- normal_market
- stressed_market
- flash_crash
- hft_dominated
- mini_flash_crash
### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your Team Credentials

ASk NBC to register your team. 
- **Team Name:** `team_alpha`
- **Password:** `secret123`

### 3. Run Your Algorithm

**Deployed server (HTTPS):**
```bash
python student_algorithm.py --name team_alpha --password secret123 --scenario normal_market --host ip:host --secure
```

### 4. Manual Trading (Optional)

Use `manual_trader.py` to manually submit orders and explore the market:

```bash
python manual_trader.py --name team_alpha --password secret123 --scenario normal_market
```

Commands: `buy <qty> <price>` | `sell <qty> <price>` | `step` | `quit`

---

## Customize Your Strategy

Edit the `decide_order()` method in `student_algorithm.py`:

```python
def decide_order(self, bid: float, ask: float, mid: float) -> Optional[Dict]:
    """
    Your strategy goes here!
    
    Available state:
        self.inventory     - Your current position
        self.pnl           - Your realized PnL
        self.current_step  - Current simulation step
    
    Return:
        {"side": "BUY"|"SELL", "price": X, "qty": N}
        Or return None to not send an order
    """
    
    # YOUR LOGIC HERE
    return {"side": "BUY", "price": mid - 0.05, "qty": 100}
```

---

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--name` | Yes | Your team name |
| `--password` | Yes | Your team password |
| `--scenario` | No | Scenario to run (default: `normal_market`) |
| `--host` | No | Server host:port (default: `localhost:8080`) |
| `--secure` | No | Use HTTPS/WSS for deployed servers |

---

## Important Rules

| Rule | Limit |
|------|-------|
| Max open orders | 50 |
| Order quantity | 100-500 per order |
| Rate limit | 1000 runs per hour |

---

## Files

| File | Description |
|------|-------------|
| `student_algorithm.py` | Main algorithm template - **edit this!** |
| `manual_trader.py` | Interactive manual trading tool |
| `requirements.txt` | Python dependencies |

