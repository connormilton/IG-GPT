# portfolio.py

import logging
import decimal
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger("TradingBot")

def safe_format_number(x):
    """
    Attempts to convert x to a Decimal and format it with 5 decimal places.
    If conversion fails, returns the original value as string.
    """
    try:
        val = decimal.Decimal(str(x))
        return f"{val:.5f}"
    except Exception:
        return str(x)

class Portfolio:
    """Tracks account state and open positions."""
    def __init__(self, broker_interface, config):
        self.broker = broker_interface
        self.config = config
        self.balance = decimal.Decimal('0.0')
        self.available = decimal.Decimal('0.0')
        self.open_positions = pd.DataFrame()
        self.trade_history = []
        self.last_update_time = None

    def update_state(self):
        logger.debug("Updating portfolio state...")
        account_details = self.broker.get_account_details()
        if account_details:
            self.balance = account_details.get('balance', self.balance)
            self.available = account_details.get('available', self.available)
        else:
            logger.error("Failed to fetch account details for portfolio update.")

        fetched_positions = self.broker.get_open_positions()
        if isinstance(fetched_positions, pd.DataFrame):
            self.open_positions = fetched_positions
        else:
            logger.error("Failed to fetch open positions for portfolio update.")

        self.last_update_time = datetime.now(timezone.utc)
        logger.info(
            f"Portfolio Updated: Balance={self.balance:.2f}, "
            f"Available={self.available:.2f}, "
            f"Open Positions={len(self.open_positions)}"
        )

    def get_balance(self):
        return self.balance

    def get_available_funds(self):
        return self.available

    def get_open_positions_df(self):
        return self.open_positions

    def get_open_positions_dict(self):
        """Converts open positions DataFrame to a list of dictionaries suitable for JSON.
        It formats numeric Decimal values with 5 decimals using safe_format_number()."""
        if self.open_positions.empty:
            return []
        try:
            df_copy = self.open_positions.copy()
            cols_to_keep = ['dealId', 'epic', 'direction', 'size', 'level', 'stopLevel', 'limitLevel', 'createdDateUTC']
            df_copy = df_copy[[c for c in cols_to_keep if c in df_copy.columns]]

            for col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(lambda x: safe_format_number(x) if x is not None else None)
            return df_copy.to_dict('records')
        except Exception as e:
            logger.error(f"Error converting open positions to dict: {e}", exc_info=True)
            return []

    def add_trade_to_history(self, log_data):
        try:
            summary_keys = [
                'timestamp', 'epic', 'direction', 'size', 'outcome',
                'pnl', 'reason', 'status', 'deal_id'
            ]
            summary = {k: log_data.get(k) for k in summary_keys}
            if 'size' in summary and isinstance(summary['size'], decimal.Decimal):
                summary['size'] = f"{summary['size']:.2f}"
            self.trade_history.append(summary)
            max_history = self.config.get('N_RECENT_TRADES_FEEDBACK', 5)
            self.trade_history = self.trade_history[-max_history:]
        except Exception as e:
            logger.error(f"Error adding trade to internal history: {e}")

    def get_recent_trade_summary(self):
        return self.trade_history

    def get_total_open_risk(self):
        logger.warning("Portfolio.get_total_open_risk() placeholder.")
        return decimal.Decimal('0.0')

    def get_currency_exposures(self):
        logger.warning("Portfolio.get_currency_exposures() placeholder.")
        return {}
