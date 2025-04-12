# data_provider.py

import logging
import decimal
from datetime import datetime, timedelta, timezone
import pandas as pd

try:
    from polygon import RESTClient
    from polygon.exceptions import NoResultsError, BadResponse
    POLYGON_CLIENT_AVAILABLE = True
except ImportError:
    RESTClient = None
    NoResultsError = Exception
    BadResponse = Exception
    POLYGON_CLIENT_AVAILABLE = False

logger = logging.getLogger("TradingBot")

class DataProvider:
    """Handles fetching market data from Polygon.io."""
    def __init__(self, config):
        self.config = config
        self.polygon_api_key = config.get('POLYGON_API_KEY')
        self.polygon_client = None
        if POLYGON_CLIENT_AVAILABLE and self.polygon_api_key:
            try:
                self.polygon_client = RESTClient(self.polygon_api_key, timeout=15)
                logger.info("Polygon.io client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Polygon.io client: {e}")
        elif not self.polygon_api_key:
            logger.error("Polygon API Key missing.")

    def get_polygon_ticker(self, epic):
        mapping = self.config.get("EPIC_TO_POLYGON_MAP", {})
        ticker = mapping.get(epic)
        if not ticker:
            logger.warning(f"No Polygon ticker mapping for IG Epic: {epic}")
        return ticker

    def get_historical_data_polygon(self, epic, end_dt=None, days_history=None):
        if not self.polygon_client:
            logger.error("Polygon client not available.")
            return pd.DataFrame()

        polygon_ticker = self.get_polygon_ticker(epic)
        if not polygon_ticker:
            logger.error(f"Cannot fetch history for {epic}: No mapping.")
            return pd.DataFrame()

        timeframe = self.config.get("HISTORICAL_DATA_TIMEFRAME", "hour")
        if days_history is None:
            days_history = self.config.get("HISTORICAL_DATA_PERIOD_DAYS", 30)
        if end_dt is None:
            end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days_history)
        end_date_str = end_dt.strftime('%Y-%m-%d')
        start_date_str = start_dt.strftime('%Y-%m-%d')

        logger.info(f"Fetching Polygon history: {polygon_ticker} ({epic}) "
                    f"[{start_date_str} to {end_date_str}], TF: {timeframe}")
        try:
            multiplier = 1
            timespan = timeframe.lower()
            if timespan not in ['minute', 'hour', 'day', 'week', 'month']:
                timespan = 'hour'

            aggs = self.polygon_client.get_aggs(
                ticker=polygon_ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date_str,
                to=end_date_str,
                adjusted=True,
                limit=50000
            )
            if not aggs:
                logger.warning(f"No results from Polygon for {polygon_ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(aggs)
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                't': 'Timestamp'
            })
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
            df = df.set_index('Timestamp')

            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: decimal.Decimal(str(x)) if pd.notna(x) else pd.NA)

            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].apply(lambda x: decimal.Decimal(str(x)) if pd.notna(x) else decimal.Decimal('0'))

            start_dt_aware = pd.Timestamp(start_dt.replace(tzinfo=timezone.utc))
            end_dt_aware = pd.Timestamp(end_dt.replace(tzinfo=timezone.utc))
            df = df[(df.index >= start_dt_aware) & (df.index <= end_dt_aware)]

            logger.info(f"Fetched {len(df)} bars for {polygon_ticker} ({epic}) from Polygon.")
            return df
        except NoResultsError:
            logger.warning(f"No results on Polygon for {polygon_ticker}.")
            return pd.DataFrame()
        except BadResponse as br:
            logger.error(f"Polygon API Error for {polygon_ticker}: Status {br.status}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Polygon fetch error for {polygon_ticker}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_news_sentiment(self, epic):
        logger.debug(f"News/sentiment placeholder for {epic}.")
        return {"headlines": [], "sentiment_score": 0.0}
