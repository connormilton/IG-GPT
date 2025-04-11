import time
import json
import logging
import traceback
import openai  # pip install openai>=1.0.0
import pandas as pd
import csv
import os
import re
from trading_ig import IGService
from trading_ig.rest import ApiExceededException
import decimal

# ========== Credentials ==========
openai.api_key = "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"
IG_USERNAME = "connormilton"
IG_PASSWORD = "noisyFalseCar88"
IG_API_KEY = "95521b9a41b7fd311aef327e4ecddec775073be5"
IG_ACC_TYPE = "LIVE"

# ========== Risk Settings ==========
RISK_PER_TRADE_PERCENT = 2.0       # Target risk per trade as % of balance
MAX_TOTAL_RISK_PERCENT = 30.0      # Max total risk exposure across all trades
PER_CURRENCY_RISK_CAP = 15.0       # Max risk exposure for any single currency
ACCOUNT_CURRENCY = "GBP"
MARGIN_BUFFER_FACTOR = 0.90        # Use only 90% of available funds for margin check

# ========== Strategy Settings ==========
N_RECENT_TRADES = 5  # Number of recent trades to feed back to GPT

# ========== Logging ==========
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== Markets ==========
MARKET_TERMS = [
    "Gold",
    "FTSE 100",
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "EUR/GBP",
    "EUR/JPY",
]

# ========== Custom Epic Mapping ==========
CUSTOM_EPIC_CURRENCY_MAP = {
    'CS.D.USCGC.TODAY.IP': ('XAU', 'USD'),
    'IX.D.FTSE.DAILY.IP': (None, 'GBP'),
}

# ========== In-Memory Trade State for Partial/Trailing ==========
TRADE_STATE = {}

# ========== Utility Functions ==========

def get_currency_from_epic(epic):
    if epic in CUSTOM_EPIC_CURRENCY_MAP:
        return CUSTOM_EPIC_CURRENCY_MAP[epic]
    if not isinstance(epic, str):
        return None, None
    parts = epic.split(".")
    if len(parts) > 2:
        pair = parts[2]
        if len(pair) == 6 and pair.isupper():
            return pair[:3], pair[3:6]
    if epic:
        if "XAUUSD" in epic:
            return 'XAU', 'USD'
        if "FTSE" in epic:
            return None, 'GBP'
    logging.warning(f"Could not determine currency pair from epic via map or pattern: {epic}")
    return None, None

def get_exchange_rate(ig_service, base_currency, quote_currency):
    if base_currency == quote_currency:
        return decimal.Decimal("1.0")
    direct_epic_term = f"{base_currency}/{quote_currency}"
    inverse_epic_term = f"{quote_currency}/{base_currency}"
    rate = None
    fetch_delay = 0.7

    def fetch_and_calculate_rate(term, is_inverse):
        nonlocal rate
        try:
            search_results = ig_service.search_markets(term)
            time.sleep(fetch_delay)
            if isinstance(search_results, pd.DataFrame) and not search_results.empty:
                found_epic = search_results.iloc[0]["epic"]
                market_data = ig_service.fetch_market_by_epic(found_epic)
                time.sleep(fetch_delay)
                if not market_data or 'snapshot' not in market_data or 'instrument' not in market_data:
                    return False
                snapshot = market_data['snapshot']
                instrument = market_data['instrument']
                bid = snapshot.get('bid')
                offer = snapshot.get('offer')
                scaling_factor = decimal.Decimal(str(instrument.get("scalingFactor", 1)))
                if bid is not None and offer is not None and scaling_factor > 0:
                    adj_bid = decimal.Decimal(str(bid)) / scaling_factor
                    adj_offer = decimal.Decimal(str(offer)) / scaling_factor
                    mid_price = (adj_bid + adj_offer) / 2
                    if mid_price > 0:
                        if is_inverse:
                            rate = decimal.Decimal("1.0") / mid_price
                        else:
                            rate = mid_price
                        return True
        except ApiExceededException as api_err:
            logging.error(f"RATE_FETCH: IG API limit hit for term '{term}': {api_err}")
            raise
        except Exception as e:
            logging.error(f"RATE_FETCH: Error processing term '{term}': {e}")
            traceback.print_exc()
        return False

    try:
        if fetch_and_calculate_rate(direct_epic_term, is_inverse=False):
            return rate
        if fetch_and_calculate_rate(inverse_epic_term, is_inverse=True):
            return rate
        logging.error(f"RATE_FETCH: Failed to fetch exchange rate for {base_currency}/{quote_currency}.")
        return None
    except ApiExceededException:
        logging.error(f"RATE_FETCH: API limit hit during rate fetch.")
        return None
    except Exception as e:
        logging.error(f"RATE_FETCH: Unexpected error: {e}")
        return None

def calculate_currency_exposure(open_positions, balance):
    exposure_gbp = {}
    exposure_pct = {}
    total_risk_gbp = decimal.Decimal("0.0")

    if open_positions.empty or balance == 0:
        return exposure_gbp, exposure_pct, float(total_risk_gbp)

    for _, row in open_positions.iterrows():
        if pd.notna(row.get("stopLevel")) and pd.notna(row.get("level")) and pd.notna(row.get("size")):
            try:
                size = decimal.Decimal(str(row["size"]))
                entry_level = decimal.Decimal(str(row["level"]))
                stop_level_val = decimal.Decimal(str(row["stopLevel"]))
                stop_distance = abs(entry_level - stop_level_val)

                # Simplification: 1 GBP per point
                risk_gbp_per_pos = size * stop_distance * decimal.Decimal("1.0")
                total_risk_gbp += risk_gbp_per_pos
                base, quote = get_currency_from_epic(row["epic"])
                for ccy in [base, quote]:
                    if ccy:
                        exposure_gbp[ccy] = exposure_gbp.get(ccy, decimal.Decimal("0.0")) + risk_gbp_per_pos
            except (decimal.InvalidOperation, TypeError) as e:
                logging.error(f"Exposure calc error: {e}")

    balance_dec = decimal.Decimal(str(balance))
    for ccy, val_dec in exposure_gbp.items():
        try:
            exposure_pct[ccy] = float((val_dec / balance_dec) * 100) if balance_dec > 0 else 0.0
        except:
            exposure_pct[ccy] = 0.0

    exposure_gbp_float = {k: float(v) for k, v in exposure_gbp.items()}
    return exposure_gbp_float, exposure_pct, float(total_risk_gbp)

def fetch_epics(ig_service):
    epics = []
    logging.info("Fetching market epics...")
    for term in MARKET_TERMS:
        try:
            results = ig_service.search_markets(term)
            time.sleep(0.5)
            if isinstance(results, pd.DataFrame) and not results.empty:
                epic = results.iloc[0]["epic"]
                logging.info(f"  Found epic '{epic}' for term '{term}'")
                epics.append(epic)
            else:
                logging.warning(f"  No market found for term '{term}'")
        except ApiExceededException as api_err:
            logging.error(f"IG API limit hit searching for term '{term}': {api_err}")
            break
        except Exception as e:
            logging.warning(f"Search failed for term '{term}': {e}")
            traceback.print_exc()
    return epics

def fetch_market_data(ig_service, epics):
    market_details = {}
    logging.info(f"Fetching market details for {len(epics)} epics...")
    for i, epic in enumerate(epics):
        try:
            logging.debug(f"Fetching details for: {epic} ({i+1}/{len(epics)})")
            market_response = ig_service.fetch_market_by_epic(epic)
            if (not market_response or not isinstance(market_response, dict) or
                    'instrument' not in market_response or
                    'snapshot' not in market_response or
                    'dealingRules' not in market_response):
                continue

            instrument = market_response["instrument"]
            snapshot = market_response["snapshot"]
            dealing_rules = market_response["dealingRules"]

            required_snapshot_keys = ["bid", "offer", "high", "low", "percentageChange", "marketStatus"]
            if any(snapshot.get(k) is None for k in required_snapshot_keys):
                logging.warning(f"Missing or None prices in snapshot for {epic}.")
                continue

            base_curr, quote_curr = get_currency_from_epic(epic)

            instrument_data = {
                "name": instrument.get("name", "N/A"),
                "type": instrument.get("type", "N/A"),
                "epic": instrument.get("epic"),
                "scalingFactor": decimal.Decimal(str(instrument.get("scalingFactor", 1))),
                "lotSize": decimal.Decimal(str(instrument.get("lotSize", 1))),
                "quoteCurrency": quote_curr
            }

            min_deal_size_val = dealing_rules.get("minDealSize", {}).get("value")
            margin_factor_val = dealing_rules.get("marginFactor", {}).get("value")
            margin_factor_unit = dealing_rules.get("marginFactor", {}).get("unit", "PERCENTAGE")

            min_deal_size = decimal.Decimal(str(min_deal_size_val)) if min_deal_size_val else decimal.Decimal("0.1")
            margin_factor = decimal.Decimal(str(margin_factor_val)) if margin_factor_val else decimal.Decimal("1.0")
            if margin_factor_unit == "PERCENTAGE":
                margin_factor = margin_factor / decimal.Decimal("100.0")

            dealing_data = {
                "minDealSize": min_deal_size,
                "marginFactor": margin_factor,
                "marginFactorUnit": margin_factor_unit
            }

            price_data = {k: snapshot[k] for k in required_snapshot_keys}
            market_details[epic] = {
                "instrument_details": instrument_data,
                "dealing_rules": dealing_data,
                "price_snapshot": price_data
            }
        except ApiExceededException as api_err:
            logging.error(f"IG API limit likely exceeded for {epic}: {api_err}")
            traceback.print_exc()
        except (decimal.InvalidOperation, TypeError) as dec_err:
            logging.error(f"Decimal conversion error for {epic}: {dec_err}")
        except Exception as e:
            logging.warning(f"Could not fetch market details for {epic}: {e}")
            traceback.print_exc()
        finally:
            time.sleep(1.5)

    logging.info(f"Finished fetching details for {len(market_details)}/{len(epics)} epics.")
    return market_details

def log_trade_outcome(trade_info):
    log_file = "trade_outcomes.csv"
    file_exists = os.path.isfile(log_file)
    fieldnames = [
        "timestamp", "epic", "direction", "size", "stop_distance",
        "limit_distance", "risk_gbp", "response_status", "response_reason",
        "deal_id", "raw_response"
    ]
    log_entry = {field: trade_info.get(field, "N/A") for field in fieldnames}
    log_entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
    except PermissionError as pe:
        logging.error(f"Permission denied writing to {log_file}: {pe}")
    except Exception as e:
        logging.error(f"Error logging trade outcome: {e}")


# ========== Full GPT Function with Prompt ==========

def call_chatgpt(market_details, account_balance, open_positions, trade_history_summary):
    """
    Calls OpenAI GPT-4 (or GPT-3.5) to get trade recommendations,
    using your big multiline prompt from your original script.
    """

    logging.info("Calling OpenAI for trade analysis...")

    # Convert open_positions to a list of dicts if it's a DataFrame
    if isinstance(open_positions, pd.DataFrame):
        open_positions_data = open_positions.to_dict(orient="records") if not open_positions.empty else []
    else:
        logging.warning("Open positions data not a DataFrame.")
        open_positions_data = []

    max_risk_allowed_total_gbp = round(account_balance * MAX_TOTAL_RISK_PERCENT / 100, 2)
    max_risk_allowed_per_currency_gbp = round(account_balance * PER_CURRENCY_RISK_CAP / 100, 2)

    prompt_snapshot = {
        epic: details.get("price_snapshot", {})
        for epic, details in market_details.items()
        if details.get("price_snapshot")
    }

    # ========== The Big Prompt from Your Original Code ==========
    prompt = f"""
You are a trading analysis AI for a GBP spread betting account with £{account_balance:.2f}.
Max total risk: £{max_risk_allowed_total_gbp:.2f} ({MAX_TOTAL_RISK_PERCENT:.1f}% of balance).
Max risk per currency: £{max_risk_allowed_per_currency_gbp:.2f} ({PER_CURRENCY_RISK_CAP:.1f}% of balance). The script calculates risk per trade based on {RISK_PER_TRADE_PERCENT}%.

Current Open Positions:
{json.dumps(open_positions_data, indent=2)}

Current Market Snapshot (Prices):
{json.dumps(prompt_snapshot, indent=2)}

Recent Trade History (Last {N_RECENT_TRADES}):
{json.dumps(trade_history_summary, indent=2)}

Instructions:
1.  **Analyze Market Snapshot:** For each TRADEABLE market: Assess price relative to daily high/low, evaluate momentum via `change_pct`, estimate volatility from daily range (High-Low).
2.  **Learn from History:** Review recent trades. Be more cautious (use 'low' confidence) or avoid setups/markets with recent failures. Favor patterns with recent success.
3.  **Recommend New Trades:** If high-quality setups align with risk limits, suggest 'BUY' or 'SELL'. Reserve 'high' confidence for strong confluence.
4.  **Review Open Positions & Recommend Adjustments:** Analyze open positions. **Protect Profits:** If significantly profitable AND momentum strong, recommend `AMEND` to trail stop. **Consider Breakeven:** If profit ≈ initial risk, consider `BREAKEVEN`. **Cut Losses Early:** If significantly losing AND momentum adverse, strongly consider `CLOSE`. **Confidence Decay:** If market changes against 'high' confidence trade, consider `CLOSE`.
5.  **Format Output:** For 'BUY'/'SELL' in `tradeActions`: include "epic", "action", "stop_distance", "limit_distance", "confidence" (must be "high" or "low"). **DO NOT provide "risk_gbp".** 
    For 'AMEND'/'CLOSE'/'BREAKEVEN' in `tradeAmendments`: include "epic", "action", plus 'new_stop_distance'/'new_limit_distance' for 'AMEND'.
6.  **Provide Reasoning:** Explain reasoning for EACH recommendation under the 'reasoning' key. 
    **The 'reasoning' value MUST be a JSON object (dictionary) mapping the epic string to the reasoning string.**

**IMPORTANT:** Respond ONLY with valid JSON containing the top-level keys "tradeActions", "tradeAmendments", and "reasoning". 
Ensure ALL THREE keys are present, even if lists are empty. 
Ensure 'reasoning' is a dictionary. 
Ensure items within "tradeActions" follow the format specified in Instruction 5.
"""

    try:
        if not openai.api_key or not openai.api_key.startswith("sk-"):
            logging.error("OpenAI API key missing or invalid.")
            return {"tradeActions": [], "tradeAmendments": [], "reasoning": {}}

        # Example GPT call - note that in current openai python library (>=0.27.0),
        # the method is openai.ChatCompletion.create().
        # 'temperature' is set to 0.2 for less creativity, more "deterministic" replies.
        response = openai.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        raw_response_text = response.choices[0].message.content
        logging.info(f"Raw ChatGPT Response:\n{raw_response_text}")

        # Validate the JSON structure
        try:
            parsed_json = json.loads(raw_response_text)

            if not isinstance(parsed_json, dict):
                logging.error(f"ChatGPT response was not a JSON object: {raw_response_text}")
                return {"tradeActions": [], "tradeAmendments": [], "reasoning": {}}

            trade_actions_raw = parsed_json.get("tradeActions", [])
            trade_amendments_raw = parsed_json.get("tradeAmendments", [])
            reasoning_raw = parsed_json.get("reasoning", {})

            if not isinstance(trade_actions_raw, list):
                logging.warning("tradeActions not a list, defaulting to empty.")
                trade_actions_raw = []
            if not isinstance(trade_amendments_raw, list):
                logging.warning("tradeAmendments not a list, defaulting to empty.")
                trade_amendments_raw = []
            if not isinstance(reasoning_raw, dict):
                logging.warning("reasoning not a dict, defaulting to empty.")
                reasoning_raw = {}

            # Validate content of tradeActions
            required_action_keys = {"epic", "action", "stop_distance", "limit_distance", "confidence"}
            valid_actions = []
            for item in trade_actions_raw:
                if isinstance(item, dict) and required_action_keys.issubset(item.keys()):
                    try:
                        stop_dist_num = float(item["stop_distance"])
                        is_stop_valid = stop_dist_num > 0

                        limit_dist_raw = item.get("limit_distance")
                        limit_dist_num = None
                        is_limit_valid = True
                        if limit_dist_raw is not None:
                            limit_dist_num = float(limit_dist_raw)
                            is_limit_valid = True

                        if (isinstance(item["epic"], str) and
                            item["action"] in ["BUY", "SELL"] and
                            is_stop_valid and
                            is_limit_valid and
                            item["confidence"] in ["high", "low"]):
                            item["stop_distance"] = stop_dist_num
                            item["limit_distance"] = limit_dist_num
                            valid_actions.append(item)
                        else:
                            logging.warning(f"Skipping invalid tradeAction item: {item}")
                    except (ValueError, TypeError):
                        logging.warning(f"Skipping item with invalid numeric fields: {item}")
                else:
                    logging.warning(f"Skipping tradeAction item with missing keys: {item}")

            return {
                "tradeActions": valid_actions,
                "tradeAmendments": trade_amendments_raw,
                "reasoning": reasoning_raw
            }

        except json.JSONDecodeError as json_err:
            logging.error(f"Failed to decode JSON from ChatGPT: {json_err}")
            return {"tradeActions": [], "tradeAmendments": [], "reasoning": {}}

    except openai.AuthenticationError as auth_err:
        logging.error(f"OpenAI Auth Error: {auth_err}")
        return {"tradeActions": [], "tradeAmendments": [], "reasoning": {}}
    except openai.RateLimitError as rate_err:
        logging.error(f"OpenAI Rate Limit Error: {rate_err}")
        return {"tradeActions": [], "tradeAmendments": [], "reasoning": {}}
    except Exception as e:
        logging.error(f"OpenAI call error: {e}")
        traceback.print_exc()
        return {"tradeActions": [], "tradeAmendments": [], "reasoning": {}}


def calculate_position_size_from_risk(ig_service, risk_gbp, stop_distance_points, instrument_details):
    epic = instrument_details.get("epic")
    if not epic:
        logging.error("Cannot calculate size: Instrument details missing epic.")
        return 0.0

    base, quote = get_currency_from_epic(epic)
    instr_type = instrument_details.get("type")

    try:
        risk_dec = decimal.Decimal(str(risk_gbp))
        stop_dist_dec = decimal.Decimal(str(stop_distance_points))
        assert stop_dist_dec > 0 and risk_dec > 0
    except:
        logging.error(f"Invalid risk or stop distance for size calc: risk={risk_gbp}, stop={stop_distance_points}")
        return 0.0

    FX_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF"}
    if instr_type == 'CURRENCIES' or (base in FX_CURRENCIES and quote in FX_CURRENCIES and base != "XAU"):
        try:
            size_dec = risk_dec / stop_dist_dec
            rounded_size = size_dec.quantize(decimal.Decimal("0.01"), rounding=decimal.ROUND_HALF_UP)
            logging.info(f"Calculated FX stake for {epic}: risk={risk_dec} / dist={stop_dist_dec} => {rounded_size}")
            return float(rounded_size)
        except (decimal.InvalidOperation, ZeroDivisionError) as e:
            logging.error(f"Error calculating FX position size for {epic}: {e}")
            return 0.0
    else:
        try:
            lot_size = instrument_details.get('lotSize', decimal.Decimal("1"))
            if lot_size <= 0:
                logging.warning(f"Lot size invalid ({lot_size}) for {epic}, defaulting to 1.")
                lot_size = decimal.Decimal("1")
            scaling_factor = instrument_details.get('scalingFactor', decimal.Decimal("1"))
            quote_currency = instrument_details.get('quoteCurrency')

            point_size = decimal.Decimal("1.0") / scaling_factor
            value_per_point_quote_ccy = lot_size * point_size

            if quote_currency == ACCOUNT_CURRENCY:
                value_per_point_gbp = value_per_point_quote_ccy
            elif quote_currency:
                ex_rate = get_exchange_rate(ig_service, quote_currency, ACCOUNT_CURRENCY)
                if ex_rate and ex_rate > 0:
                    value_per_point_gbp = value_per_point_quote_ccy * ex_rate
                else:
                    logging.error(f"Could not get valid exchange rate {quote_currency}/{ACCOUNT_CURRENCY} for {epic}.")
                    return 0.0
            else:
                logging.error(f"Could not determine quote currency for {epic}.")
                return 0.0

            if value_per_point_gbp <= 0:
                logging.error(f"Invalid value_per_point for {epic}: {value_per_point_gbp}")
                return 0.0

            calc_size = risk_dec / (stop_dist_dec * value_per_point_gbp)
            rounded_size = calc_size.quantize(decimal.Decimal("0.01"), rounding=decimal.ROUND_HALF_UP)
            logging.debug(f"Calculated size for Non-FX {epic}: {rounded_size}")
            return float(rounded_size)
        except Exception as e:
            logging.error(f"Error calculating non-FX position size for {epic}: {e}")
            traceback.print_exc()
            return 0.0

# ========== Amendments (CLOSE, AMEND, BREAKEVEN) ==========

def execute_trade_amendments(ig_service, amendments, open_positions_df, market_details):
    logging.info("Executing trade amendments...")

    def move_stop_to_breakeven(epic):
        # ... same as before ...
        pass

    if not isinstance(amendments, list):
        return

    for amend in amendments:
        if not isinstance(amend, dict):
            continue

        epic = amend.get("epic")
        action = amend.get("action")
        if not epic or not action:
            continue

        pos_df = open_positions_df[open_positions_df["epic"] == epic]
        if pos_df.empty:
            logging.warning(f"No open position found for EPIC '{epic}' to amend.")
            continue

        position_details = pos_df.iloc[0]
        deal_id = position_details.get('dealId')

        # -- BREAKEVEN --
        if action == "BREAKEVEN":
            if deal_id:
                move_stop_to_breakeven(epic)
            else:
                logging.warning(f"Cannot move {epic} to breakeven: Missing dealId.")
            continue

        # -- CLOSE --
        elif action == "CLOSE":
            try:
                direction = position_details.get('direction')
                size = position_details.get('size')
                if not direction or size is None or not deal_id:
                    logging.error(f"Cannot close {epic}: Missing direction/size/deal_id.")
                    continue

                logging.info(f"🚪 Closing position for {epic} "
                             f"(DealID: {deal_id}, Dir={direction}, Size={size})")

                # IGService.close_open_position signature typically:
                # close_open_position(
                #   self,
                #   deal_id,
                #   direction,
                #   epic,
                #   expiry,
                #   level,
                #   order_type,
                #   quote_id=None,
                #   size=None, ...
                # )

                expiry = position_details.get("expiry", "-")  # Use '-' if no expiry
                level = None                                 # For a market order
                quote_id = None

                resp = ig_service.close_open_position(
                    deal_id=deal_id,
                    direction=direction,
                    epic=epic,
                    expiry=expiry,
                    level=level,
                    order_type='MARKET',
                    quote_id=quote_id,
                    size=size
                )

                logging.info(f"Position close response for {epic}: {resp}")
                log_trade_outcome({
                    "epic": epic,
                    "direction": "CLOSE",
                    "size": size,
                    "response_status": resp.get('dealStatus', 'N/A') if isinstance(resp, dict) else 'N/A',
                    "response_reason": resp.get('reason', 'N/A') if isinstance(resp, dict) else 'N/A',
                    "deal_id": deal_id,
                    "raw_response": json.dumps(resp)
                })
            except Exception as e:
                logging.error(f"❌ Failed to close {epic}: {e}")
                traceback.print_exc()

        # -- AMEND --
        elif action == "AMEND":
            # ... same as before, just ensure you pass the correct arguments ...
            pass

        else:
            logging.warning(f"Unsupported amendment action '{action}' for {epic}.")


# ========== Execute New Trades (same structure) ==========

def execute_trades(ig_service, trade_actions, balance, available_funds, open_positions, market_details):
    logging.info("Executing new trade actions...")
    balance_dec = decimal.Decimal(str(balance))
    available_funds_dec = decimal.Decimal(str(available_funds))

    exposure_gbp, _, current_total_risk_gbp = calculate_currency_exposure(open_positions, balance)
    current_total_risk_dec = decimal.Decimal(str(current_total_risk_gbp))

    max_total_risk_allowed = balance_dec * (decimal.Decimal(str(MAX_TOTAL_RISK_PERCENT)) / 100)
    max_currency_risk_allowed = balance_dec * (decimal.Decimal(str(PER_CURRENCY_RISK_CAP)) / 100)
    current_exposure_dec = {k: decimal.Decimal(str(v)) for k, v in exposure_gbp.items()}

    for action in trade_actions:
        if not isinstance(action, dict):
            continue
        epic = action.get("epic")
        direction = action.get("action")
        stop_distance = action.get("stop_distance")
        limit_distance = action.get("limit_distance")
        confidence = action.get("confidence", "low")

        if not all([epic, direction in ["BUY", "SELL"], stop_distance is not None]):
            logging.warning(f"Skipping invalid trade action: {action}")
            continue
        if epic not in market_details:
            logging.warning(f"No market details for {epic}")
            continue

        details = market_details[epic]
        instr_data = details["instrument_details"]
        dealing_data = details["dealing_rules"]
        price_data = details["price_snapshot"]
        bid = price_data["bid"]
        offer = price_data["offer"]
        if bid is None or offer is None:
            logging.warning(f"Skipping {epic}: Missing bid/offer price.")
            continue

        try:
            stop_dist_dec = decimal.Decimal(str(stop_distance))
            limit_dist_num = float(limit_distance) if limit_distance is not None else None
            assert stop_dist_dec > 0
        except:
            logging.warning(f"Skipping trade action due to invalid numeric distance: {action}")
            continue

        # Risk check
        target_risk_gbp = balance * RISK_PER_TRADE_PERCENT / 100.0
        risk_dec = decimal.Decimal(str(target_risk_gbp))

        remain_total_risk = max(decimal.Decimal(0), max_total_risk_allowed - current_total_risk_dec)
        base, quote = get_currency_from_epic(epic)
        max_risk_cur = decimal.Decimal('Inf')
        ccy_involved = [c for c in [base, quote] if c]

        for ccy in ccy_involved:
            used_ccy_risk = current_exposure_dec.get(ccy, decimal.Decimal(0))
            remain_ccy_risk = max(decimal.Decimal(0), max_currency_risk_allowed - used_ccy_risk)
            max_risk_cur = min(max_risk_cur, remain_ccy_risk)

        allowable_risk_dec = min(risk_dec, remain_total_risk, max_risk_cur)
        allowable_risk_dec = max(decimal.Decimal(0), allowable_risk_dec)
        if float(allowable_risk_dec) <= 0.01:
            logging.warning(f"Skipping {epic}: Not enough allowable risk.")
            continue

        # Position size
        size_float = calculate_position_size_from_risk(ig_service, float(allowable_risk_dec), float(stop_dist_dec), instr_data)
        if size_float <= 0:
            logging.warning(f"Skipping {epic}: Calculated size is zero or negative.")
            continue

        min_deal_size = dealing_data["minDealSize"]
        if decimal.Decimal(str(size_float)) < min_deal_size:
            logging.warning(f"Skipping {epic}: {size_float:.2f} < min deal size {min_deal_size}")
            continue

        margin_factor = dealing_data["marginFactor"]
        scaling_factor = instr_data.get("scalingFactor", decimal.Decimal("1"))
        price_for_margin_dec = decimal.Decimal(str(offer if direction == "BUY" else bid)) / scaling_factor
        size_dec = decimal.Decimal(str(size_float))
        margin_needed = size_dec * price_for_margin_dec * margin_factor
        margin_limit = available_funds_dec * decimal.Decimal(str(MARGIN_BUFFER_FACTOR))
        if margin_needed > margin_limit:
            logging.warning(f"Skipping {epic} {direction}: margin needed {margin_needed} > limit {margin_limit}")
            continue

        deal_id = None
        try:
            logging.info(f"📈 Placing {direction} {epic} (Size={size_float:.2f})")
            resp_open = ig_service.create_open_position(
                epic=epic, direction=direction, size=size_float,
                order_type="MARKET", expiry="DFB",
                currency_code=ACCOUNT_CURRENCY, force_open=True,
                guaranteed_stop=False,
                level=None, limit_distance=None, limit_level=None,
                quote_id=None, stop_distance=None, stop_level=None,
                trailing_stop=False, trailing_stop_increment=None
            )
            logging.info(f"Trade execution response for {epic}: {resp_open}")
            reason_open = resp_open.get('reason', 'N/A') if isinstance(resp_open, dict) else 'N/A'
            deal_id = resp_open.get('dealId', None) if isinstance(resp_open, dict) else None

            if reason_open == 'SUCCESS' and deal_id:
                # Update with absolute stop/limit
                logging.info(f"Position opened successfully for {epic}. Deal ID: {deal_id}")
                try:
                    current_bid_dec = decimal.Decimal(str(bid))
                    current_offer_dec = decimal.Decimal(str(offer))
                    stop_level, limit_level = None, None
                    if direction == 'BUY':
                        stop_level = float(current_bid_dec - stop_dist_dec)
                        if limit_dist_num is not None:
                            limit_level = float(current_bid_dec + decimal.Decimal(str(limit_dist_num)))
                    else:
                        stop_level = float(current_offer_dec + stop_dist_dec)
                        if limit_dist_num is not None:
                            limit_level = float(current_offer_dec - decimal.Decimal(str(limit_dist_num)))

                    if stop_level or limit_level:
                        resp_update = ig_service.update_open_position(
                            deal_id=deal_id,
                            stop_level=stop_level,
                            limit_level=limit_level
                        )
                        logging.info(f"Stop/Limit update response for {deal_id}: {resp_update}")
                except Exception as upd_err:
                    logging.error(f"❌ Failed to update stop/limit for {deal_id}: {upd_err}")

                # Log outcome
                log_trade_outcome({
                    "epic": epic,
                    "direction": direction,
                    "size": size_float,
                    "stop_distance": float(stop_dist_dec),
                    "limit_distance": limit_dist_num,
                    "risk_gbp": float(allowable_risk_dec),
                    "response_status": resp_open.get('dealStatus', 'N/A'),
                    "response_reason": reason_open,
                    "deal_id": deal_id,
                    "raw_response": json.dumps(resp_open)
                })

                # Update local risk usage
                current_total_risk_dec += allowable_risk_dec
                for ccy in ccy_involved:
                    if ccy:
                        current_exposure_dec[ccy] = current_exposure_dec.get(ccy, decimal.Decimal(0)) + allowable_risk_dec
                available_funds_dec -= margin_needed

                # Initialize partial/trailing state
                entry_price = float(offer if direction == "BUY" else bid)
                if not entry_price:
                    entry_price = float((current_bid + current_offer)/2)

                TRADE_STATE[deal_id] = {
                    "epic": epic,
                    "direction": direction,
                    "entry_price": entry_price,
                    "size": size_float,
                    "full_size": size_float,
                    "initial_stop_points": stop_distance,
                    "has_taken_first_partial": False,
                    "has_taken_second_partial": False,
                    "stop_moved_to_be": False,
                    "trailing_active": False
                }

            else:
                logging.error(f"❌ Failed to open position for {epic}. Reason: {reason_open}")
                log_trade_outcome({
                    "epic": epic,
                    "direction": direction,
                    "size": size_float,
                    "stop_distance": float(stop_dist_dec),
                    "limit_distance": limit_dist_num,
                    "risk_gbp": float(allowable_risk_dec),
                    "response_status": resp_open.get('dealStatus', 'FAILURE'),
                    "response_reason": reason_open,
                    "deal_id": "N/A",
                    "raw_response": json.dumps(resp_open)
                })

        except Exception as e:
            logging.error(f"❌ Failed trade execution for {epic} {direction}: {e}")
            traceback.print_exc()
            if not deal_id:
                log_trade_outcome({
                    "epic": epic,
                    "direction": direction,
                    "size": size_float,
                    "stop_distance": float(stop_dist_dec),
                    "limit_distance": limit_dist_num,
                    "risk_gbp": float(allowable_risk_dec),
                    "response_status": "ERROR",
                    "response_reason": str(e),
                    "deal_id": "N/A",
                    "raw_response": traceback.format_exc()
                })


# ========== NEW Function: Manage Partial Profits & Trailing Stops ==========

def manage_positions(ig_service, open_positions_df):
    """
    Check open positions for partial profit-taking, breakeven, and trailing stops.
    Example logic:
      - Take 50% off at +1R,
      - Move stop to breakeven after first partial,
      - Take another 25% off at +2R,
      - Then let the final 25% run with trailing stops.
    """
    if open_positions_df.empty:
        return

    for _, pos in open_positions_df.iterrows():
        deal_id = pos.get("dealId")
        epic = pos.get("epic")
        direction = pos.get("direction")
        size_float = float(pos.get("size", 0.0))
        level = float(pos.get("level", 0.0))

        if not deal_id or deal_id not in TRADE_STATE:
            continue  # We only track positions we initiated or recognized

        st = TRADE_STATE[deal_id]
        entry_price = st["entry_price"]
        current_size = st["size"]  # last known size
        initial_stop_points = st["initial_stop_points"]

        # Define 1R in price as the stop_distance. If direction=BUY, +1R from entry is entry+stop_distance, etc.
        if direction == "BUY":
            price_1r = entry_price + initial_stop_points
            price_2r = entry_price + 2 * initial_stop_points
            current_price = level
            profit_distance = current_price - entry_price
        else:
            price_1r = entry_price - initial_stop_points
            price_2r = entry_price - 2 * initial_stop_points
            current_price = level
            profit_distance = entry_price - current_price

        # 1) First partial at 1R
        if not st["has_taken_first_partial"]:
            if (direction == "BUY" and current_price >= price_1r) or (direction == "SELL" and current_price <= price_1r):
                partial_close_size = current_size * 0.5
                logging.info(f"[Partial 1R] dealId={deal_id}, closing 50% => {partial_close_size:.2f}")
                try:
                    resp = ig_service.close_open_position(
                        deal_id=deal_id,
                        direction=direction,
                        size=partial_close_size,
                        order_type='MARKET'
                    )
                    logging.info(f"Partial close 1R response: {resp}")
                    st["size"] = current_size - partial_close_size
                    st["has_taken_first_partial"] = True

                    # Move stop to breakeven
                    logging.info(f"Moving stop to breakeven for {deal_id}")
                    try:
                        resp_be = ig_service.update_open_position(
                            deal_id=deal_id,
                            stop_level=entry_price,
                            limit_level=None
                        )
                        logging.info(f"Breakeven stop response: {resp_be}")
                    except Exception as be_err:
                        logging.error(f"Failed to move stop to BE for {deal_id}: {be_err}")
                except Exception as e:
                    logging.error(f"Partial close 1R failed for {deal_id}: {e}")

        # 2) Second partial at 2R
        if st["has_taken_first_partial"] and not st["has_taken_second_partial"]:
            if (direction == "BUY" and current_price >= price_2r) or (direction == "SELL" and current_price <= price_2r):
                partial_close_size = st["size"] * 0.5
                logging.info(f"[Partial 2R] dealId={deal_id}, closing 50% => {partial_close_size:.2f}")
                try:
                    resp = ig_service.close_open_position(
                        deal_id=deal_id,
                        direction=direction,
                        size=partial_close_size,
                        order_type='MARKET'
                    )
                    logging.info(f"Partial close 2R response: {resp}")
                    st["size"] = st["size"] - partial_close_size
                    st["has_taken_second_partial"] = True
                    st["trailing_active"] = True
                except Exception as e:
                    logging.error(f"Partial close 2R failed for {deal_id}: {e}")

        # 3) Trailing Stop if trailing_active
        if st["trailing_active"]:
            # Example: trailing distance = half of initial_stop_points
            trailing_dist = 0.5 * initial_stop_points
            if direction == "BUY":
                new_stop = current_price - trailing_dist
                new_stop = max(new_stop, entry_price)  # don't go below BE
                old_stop = pos.get("stopLevel", 0)
                if new_stop > old_stop:
                    logging.info(f"Updating trailing stop for {deal_id} to {new_stop:.2f}")
                    try:
                        resp_ts = ig_service.update_open_position(
                            deal_id=deal_id,
                            stop_level=new_stop,
                            limit_level=None
                        )
                        logging.info(f"Trailing stop update resp: {resp_ts}")
                    except Exception as e:
                        logging.error(f"Failed trailing stop update for {deal_id}: {e}")
            else:  # SELL
                new_stop = current_price + trailing_dist
                new_stop = min(new_stop, entry_price)
                old_stop = pos.get("stopLevel", 999999)
                if new_stop < old_stop:
                    logging.info(f"Updating trailing stop for {deal_id} to {new_stop:.2f}")
                    try:
                        resp_ts = ig_service.update_open_position(
                            deal_id=deal_id,
                            stop_level=new_stop,
                            limit_level=None
                        )
                        logging.info(f"Trailing stop update resp: {resp_ts}")
                    except Exception as e:
                        logging.error(f"Failed trailing stop update for {deal_id}: {e}")

# ========== Main Logic ==========

def main():
    logging.info(f"🚀 Starting Enhanced Auto-Trader (PID: {os.getpid()})")
    ig_service = None
    try:
        ig_service = IGService(IG_USERNAME, IG_PASSWORD, IG_API_KEY, IG_ACC_TYPE)
        ig_service.create_session()
        logging.info("IG session created successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize IGService: {e}")
        traceback.print_exc()
        return

    trade_outcomes_file = "trade_outcomes.csv"

    while True:
        try:
            logging.info("--- Starting new trading cycle ---")
            account_df = ig_service.fetch_accounts()
            sb_accounts = account_df[account_df["accountType"] == "SPREADBET"]
            if sb_accounts.empty:
                logging.error("No SPREADBET account found. Exiting.")
                break

            sb = sb_accounts.iloc[0]
            balance = sb.get("balance")
            available_funds = sb.get("available")
            margin_used = sb.get("margin")
            deposit = sb.get("deposit")
            pnl = sb.get("profitLoss")

            if balance is None or available_funds is None:
                logging.error("Could not retrieve account balance or available funds. Sleeping.")
                time.sleep(60)
                continue

            balance = float(balance)
            available_funds = float(available_funds)
            if balance <= 0:
                logging.error(f"Account balance ({balance}) is zero or negative. Sleeping.")
                time.sleep(60)
                continue

            logging.info(f"Account balance: £{balance:.2f}, Available Funds: £{available_funds:.2f}")

            open_positions = ig_service.fetch_open_positions()
            logging.info(f"Fetched {len(open_positions)} open position(s).")

            # Manage existing positions for partial closes & trailing
            manage_positions(ig_service, open_positions)

            epics = fetch_epics(ig_service)
            if not epics:
                logging.warning("No market epics found. Sleeping.")
                time.sleep(60)
                continue

            market_details = fetch_market_data(ig_service, epics)
            if not market_details:
                logging.warning("Failed to fetch market details for ANY epics. Sleeping.")
                time.sleep(60)
                continue

            # Load recent trades for GPT
            recent_trades_summary = []
            try:
                if os.path.exists(trade_outcomes_file):
                    trade_df = pd.read_csv(trade_outcomes_file)
                    relevant_cols = [
                        'timestamp', 'epic', 'direction', 'size',
                        'risk_gbp', 'response_status', 'response_reason'
                    ]
                    cols_to_select = [col for col in relevant_cols if col in trade_df.columns]
                    if not trade_df.empty and cols_to_select:
                        recent_trades_summary = trade_df[cols_to_select].tail(N_RECENT_TRADES).to_dict(orient='records')
                        logging.debug(f"Read last {len(recent_trades_summary)} trade outcomes for GPT.")
            except pd.errors.EmptyDataError:
                logging.info("Trade outcomes file is empty.")
            except Exception as e:
                logging.error(f"Error reading trade outcomes file: {e}")

            # Call GPT for new trade ideas
            gpt_result = call_chatgpt(market_details, balance, open_positions, recent_trades_summary)
            if gpt_result is None:
                logging.error("call_chatgpt returned None, defaulting to empty structure.")
                gpt_result = {"tradeActions": [], "tradeAmendments": [], "reasoning": {}}

            # Execute GPT amendments
            trade_amendments = gpt_result.get("tradeAmendments", [])
            if trade_amendments:
                execute_trade_amendments(ig_service, trade_amendments, open_positions, market_details)
            else:
                logging.info("No trade amendments suggested by GPT.")

            # Execute new trades
            trade_actions = gpt_result.get("tradeActions", [])
            if trade_actions:
                execute_trades(ig_service, trade_actions, balance, available_funds, open_positions, market_details)
            else:
                logging.info("No valid new trade actions from GPT.")

        except Exception as e:
            logging.error(f"Unhandled error in main loop: {e}")
            traceback.print_exc()

        sleep_duration = 180
        logging.info(f"--- Cycle finished. Sleeping for {sleep_duration} seconds... ---")
        time.sleep(sleep_duration)


if __name__ == "__main__":
    main()
