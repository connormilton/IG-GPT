# trade_executor.py

import os
import json
import logging
import decimal
from datetime import datetime, timezone
from csv import DictWriter
import time

logger = logging.getLogger("TradingBot")

class TradeLogger:
    """Logs trade details to CSV."""
    def __init__(self, filename):
        self.filename = filename
        self.fieldnames = [
            "timestamp", "epic", "direction", "size", "signal_price", "entry_price",
            "stop_level", "limit_level", "stop_distance", "limit_distance",
            "confidence", "estimated_risk_gbp", "status", "response_status",
            "reason", "response_reason", "deal_id", "pnl", "outcome", "raw_response"
        ]
        self._ensure_header()

    def _ensure_header(self):
        file_exists = os.path.isfile(self.filename)
        if not file_exists:
            try:
                os.makedirs(os.path.dirname(self.filename), exist_ok=True)
                with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
                    writer = DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                    logger.info(f"Created trade log file: {self.filename}")
            except Exception as e:
                logger.error(f"Error ensuring trade log header {self.filename}: {e}")

    def log_trade(self, trade_data):
        log_entry = {field: trade_data.get(field, "") for field in self.fieldnames}
        log_entry["timestamp"] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        try:
            with open(self.filename, mode="a", newline="", encoding="utf-8") as f:
                writer = DictWriter(f, fieldnames=self.fieldnames)
                for key, value in log_entry.items():
                    if isinstance(value, decimal.Decimal):
                        log_entry[key] = f"{value:.6f}"
                    elif isinstance(value, float):
                        log_entry[key] = f"{value:.6f}"
                writer.writerow(log_entry)
        except Exception as e:
            logger.error(f"Error logging trade: {e}", exc_info=True)

class ExecutionHandler:
    """Handles placing trades and logging outcomes."""
    def __init__(self, broker_interface, portfolio, config):
        self.broker = broker_interface
        self.portfolio = portfolio
        self.config = config
        self.trade_logger = TradeLogger(config['TRADE_HISTORY_FILE'])

    def execute_new_trade(self, trade_details):
        epic = trade_details['epic']
        log_attempt = trade_details.copy()
        log_attempt["status"] = "ATTEMPT_OPEN"
        log_attempt["outcome"] = "PENDING"
        self.trade_logger.log_trade(log_attempt)

        result = self.broker.execute_trade(trade_details)
        status = result.get('status', 'ERROR')
        reason = result.get('reason', 'Unknown')
        deal_id = result.get('dealId')
        success = (status == 'SUCCESS' and deal_id is not None)

        log_outcome = trade_details.copy()
        log_outcome.update({
            "status": "EXECUTED" if success else "FAILED",
            "response_status": status,
            "response_reason": reason,
            "deal_id": deal_id or "N/A",
            "outcome": "OPENED" if success else "FAILED_OPEN",
            "raw_response": json.dumps(result)
        })
        self.trade_logger.log_trade(log_outcome)
        self.portfolio.add_trade_to_history(log_outcome)
        return success, deal_id if success else reason

    def close_trade(self, deal_id, size, direction, epic="Unknown"):
        log_attempt = {
            "deal_id": deal_id,
            "epic": epic,
            "direction": "CLOSE",
            "size": size,
            "status": "ATTEMPT_CLOSE",
            "outcome": "PENDING"
        }
        self.trade_logger.log_trade(log_attempt)

        result = self.broker.close_trade(deal_id, size, direction)
        status = result.get('status', 'ERROR')
        reason = result.get('reason', 'Unknown')
        success = (status == 'SUCCESS')

        log_outcome = {
            "deal_id": deal_id,
            "epic": epic,
            "direction": "CLOSE",
            "size": size,
            "status": "EXECUTED" if success else "FAILED",
            "response_status": status,
            "response_reason": reason,
            "outcome": "CLOSED" if success else "FAILED_CLOSE",
            "raw_response": json.dumps(result)
        }
        # Optionally fetch PnL for this deal_id after closure
        self.trade_logger.log_trade(log_outcome)
        self.portfolio.add_trade_to_history(log_outcome)
        return success

    def amend_trade(self, deal_id, stop_level=None, limit_level=None, epic="Unknown"):
        log_attempt = {
            "deal_id": deal_id,
            "epic": epic,
            "direction": "AMEND",
            "status": "ATTEMPT_AMEND",
            "outcome": "PENDING",
            "stop_level": stop_level,
            "limit_level": limit_level
        }
        self.trade_logger.log_trade(log_attempt)

        result = self.broker.amend_trade(deal_id, stop_level, limit_level)
        status = result.get('status', 'ERROR')
        reason = result.get('reason', 'Unknown')
        success = (status == 'SUCCESS')

        log_outcome = {
            "deal_id": deal_id,
            "epic": epic,
            "direction": "AMEND",
            "status": "EXECUTED" if success else "FAILED",
            "response_status": status,
            "response_reason": reason,
            "outcome": "AMENDED" if success else "FAILED_AMEND",
            "stop_level": stop_level,
            "limit_level": limit_level,
            "raw_response": json.dumps(result)
        }
        self.trade_logger.log_trade(log_outcome)
        self.portfolio.add_trade_to_history(log_outcome)
        return success
