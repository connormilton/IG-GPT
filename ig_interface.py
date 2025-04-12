# ig_interface.py

import time
import logging
import decimal
from datetime import datetime, timezone

try:
    from trading_ig import IGService
    from trading_ig.rest import ApiExceededException, IGException
    TRADING_IG_AVAILABLE = True
except ImportError:
    TRADING_IG_AVAILABLE = False
    IGService = None
    ApiExceededException = Exception
    IGException = Exception

logger = logging.getLogger("TradingBot")

class IGInterface:
    """Handles IG Broker API interactions."""
    def __init__(self, config):
        self.config = config
        self.ig_service = None
        self.account_id = config.get("IG_ACCOUNT_ID")

        # Custom mapping for instrument currency info
        self.CUSTOM_EPIC_CURRENCY_MAP = {
            'CS.D.USCGC.TODAY.IP': ('XAU', 'USD'),
            'IX.D.FTSE.DAILY.IP': (None, 'GBP'),
            "CS.D.EURUSD.MINI.IP": ('EUR', 'USD'),
            "CS.D.USDJPY.MINI.IP": ('USD', 'JPY'),
            "CS.D.GBPUSD.MINI.IP": ('GBP', 'USD'),
            "CS.D.AUDUSD.MINI.IP": ('AUD', 'USD'),
        }

        if not TRADING_IG_AVAILABLE:
            logger.critical("IGInterface cannot be initialized: trading_ig library missing.")
            return

        self._connect()

    def _connect(self):
        try:
            logger.info(f"Connecting to IG as {self.config.get('IG_USERNAME')} ({self.config.get('IG_ACC_TYPE')})...")
            # Removed requests_timeout argument because the IGService no longer supports it.
            self.ig_service = IGService(
                self.config['IG_USERNAME'],
                self.config['IG_PASSWORD'],
                self.config['IG_API_KEY'],
                self.config['IG_ACC_TYPE']
            )
            self.ig_service.create_session()

            if self.account_id:
                logger.info(f"Setting default IG account ID: {self.account_id}")
                try:
                    self.ig_service.switch_account(self.account_id, False)
                except Exception as switch_err:
                    err_str = str(switch_err)
                    # If the account is already active, IG might return an error like "error.switch.accountId-must-be-different".
                    if "error.switch.accountId-must-be-different" in err_str:
                        logger.info("Account already active, continuing without switching.")
                    else:
                        raise switch_err
            else:
                logger.warning("IG_ACCOUNT_ID not set. Operations might use unexpected account.")

            logger.info("IG connection successful.")
        except (IGException, Exception) as e:
            logger.error(f"Failed to connect/setup IG session: {e}", exc_info=False)
            self.ig_service = None

    def _get_currency_from_epic(self, epic: str):
        if epic in self.CUSTOM_EPIC_CURRENCY_MAP:
            return self.CUSTOM_EPIC_CURRENCY_MAP[epic]
        if not isinstance(epic, str):
            return None, None
        parts = epic.split('.')
        if len(parts) > 2 and parts[0] == 'CS' and parts[1] == 'D':
            pair = parts[2]
            if len(pair) == 6 and pair.isupper():
                return pair[:3], pair[3:]
        elif epic:
            if "XAUUSD" in epic:
                return 'XAU', 'USD'
            if "FTSE" in epic:
                return None, 'GBP'
        logger.debug(f"Could not determine currency pair from epic pattern: {epic}.")
        return None, None

    def get_account_details(self):
        if not self.ig_service:
            logger.error("IG service not connected.")
            return None
        try:
            accounts = self.ig_service.fetch_accounts()
            if self.account_id:
                target_account = accounts[accounts['accountId'] == self.account_id]
            else:
                if not accounts.empty:
                    logger.warning("No IG_ACCOUNT_ID specified, using first account.")
                    target_account = accounts.iloc[[0]]
                else:
                    logger.error("No IG accounts found.")
                    return None

            if target_account.empty:
                logger.error(f"Account ID {self.account_id} not found.")
                return None

            details = target_account.iloc[0].to_dict()
            for key in ['balance', 'available', 'deposit', 'profitLoss']:
                details[key] = decimal.Decimal(str(details.get(key, 0)))
            return details
        except (ApiExceededException, IGException) as api_err:
            logger.error(f"IG API Error fetching accounts: {api_err}")
            time.sleep(60)
            return None
        except Exception as e:
            logger.error(f"Failed to fetch IG account details: {e}", exc_info=True)
            return None

    def get_open_positions(self):
        import pandas as pd
        if not self.ig_service:
            logger.error("IG service not connected.")
            return pd.DataFrame()
        try:
            df = self.ig_service.fetch_open_positions()
            if not df.empty:
                for col in ['size', 'level', 'stopLevel', 'limitLevel']:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: decimal.Decimal(str(x)) if pd.notna(x) else None
                        )
                if 'createdDateUTC' in df.columns:
                    df['createdDateUTC'] = pd.to_datetime(df['createdDateUTC'], utc=True)
            logger.info(f"Fetched {len(df)} open positions.")
            return df
        except (ApiExceededException, IGException) as api_err:
            logger.error(f"IG API Error fetching positions: {api_err}")
            time.sleep(60)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch IG open positions: {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_market_snapshot(self, epic: str):
        if not self.ig_service:
            logger.error("IG service not connected.")
            return None
        try:
            market_response = self.ig_service.fetch_market_by_epic(epic)
            time.sleep(0.6)
            if (market_response and 'snapshot' in market_response and
                    market_response['snapshot'].get('bid') is not None and
                    market_response['snapshot'].get('offer') is not None):
                snapshot = market_response['snapshot']
                converted_snapshot = {}
                for key in ['bid', 'offer', 'high', 'low', 'netChange', 'percentageChange']:
                    if key in snapshot and snapshot[key] is not None:
                        try:
                            converted_snapshot[key] = decimal.Decimal(str(snapshot[key]))
                        except decimal.InvalidOperation:
                            converted_snapshot[key] = None
                converted_snapshot['marketStatus'] = snapshot.get('marketStatus')
                converted_snapshot['updateTimeUTC'] = snapshot.get('updateTimeUTC')
                converted_snapshot['fetchTimeUTC'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
                return converted_snapshot
            else:
                logger.warning(f"No valid snapshot data found for epic: {epic}")
                return None
        except (ApiExceededException, IGException) as api_err:
            logger.error(f"IG API Error fetching snapshot {epic}: {api_err}")
            time.sleep(60)
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch snapshot for {epic}: {e}", exc_info=False)
            return None

    def get_instrument_details(self, epic: str):
        if not self.ig_service:
            logger.error("IG service not connected.")
            return None
        try:
            market_response = self.ig_service.fetch_market_by_epic(epic)
            time.sleep(0.6)
            if market_response and 'instrument' in market_response and 'dealingRules' in market_response:
                instrument = market_response['instrument']
                rules = market_response['dealingRules']

                vpp_val = instrument.get('valueOfOnePip') or instrument.get('valuePerPoint')
                vpp_str_hint = instrument.get('onePipMeans')
                value_per_point = None
                vpp_currency = None

                if vpp_val is not None:
                    try:
                        value_per_point = decimal.Decimal(str(vpp_val))
                    except decimal.InvalidOperation:
                        logger.warning(f"Invalid VPP value {vpp_val} for {epic}")

                if vpp_str_hint and isinstance(vpp_str_hint, str):
                    parts = vpp_str_hint.split()
                    if len(parts) > 1 and len(parts[0]) == 3 and parts[0].isupper():
                        vpp_currency = parts[0]

                margin_factor_val = rules.get('marginFactor', {}).get('value', 100.0)
                margin_factor_unit = rules.get('marginFactor', {}).get('unit', 'PERCENTAGE')
                margin_factor = decimal.Decimal(str(margin_factor_val))
                if margin_factor_unit == 'PERCENTAGE':
                    margin_factor /= decimal.Decimal(100.0)

                base_ccy, quote_ccy = self._get_currency_from_epic(epic)
                final_quote_ccy = quote_ccy or vpp_currency

                details = {
                    'name': instrument.get('name'),
                    'epic': epic,
                    'type': instrument.get('type'),
                    'lotSize': decimal.Decimal(str(instrument.get('lotSize', 1))),
                    'scalingFactor': decimal.Decimal(str(instrument.get('scalingFactor', 1))),
                    'minDealSize': decimal.Decimal(str(rules.get('minDealSize', {}).get('value', 0.1))),
                    'marginFactor': margin_factor,
                    'valuePerPoint': value_per_point,
                    'valuePerPointCurrency': vpp_currency,
                    'quoteCurrency': final_quote_ccy,
                }
                return details
            else:
                logger.warning(f"Could not fetch full instrument/dealing details for {epic}")
                return None
        except (ApiExceededException, IGException) as api_err:
            logger.error(f"IG API Error fetching instrument {epic}: {api_err}")
            time.sleep(60)
            return None
        except (decimal.InvalidOperation, TypeError) as dec_err:
            logger.error(f"Decimal conversion error fetching details for {epic}: {dec_err}")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch instrument details for {epic}: {e}", exc_info=False)
            return None

    def execute_trade(self, trade_details):
        if not self.ig_service:
            return {'status': 'FAILURE', 'reason': 'Not connected'}
        try:
            size_float = float(trade_details['size'])
            stop_float = float(trade_details['stop_level']) if trade_details.get('stop_level') else None
            limit_float = float(trade_details['limit_level']) if trade_details.get('limit_level') else None
            epic = trade_details['epic']
            direction = trade_details['direction']

            logger.info(f"Executing {direction} {size_float:.2f} {epic} | Stop: {stop_float} Limit: {limit_float}")
            resp = self.ig_service.create_open_position(
                epic=epic,
                direction=direction,
                size=size_float,
                order_type='MARKET',
                expiry='DFB',
                currency_code=self.config['ACCOUNT_CURRENCY'],
                force_open=True,
                guaranteed_stop=False,
                stop_level=stop_float,
                limit_level=limit_float
            )
            deal_ref = resp.get('dealReference', 'N/A')
            ig_status = resp.get('dealStatus', 'UNKNOWN')
            reason = resp.get('reason', 'N/A')
            deal_id = resp.get('dealId')

            if ig_status == 'ACCEPTED' and deal_id:
                return {'status': 'SUCCESS', 'dealId': deal_id, 'reason': reason}
            else:
                logger.error(f"Failed to open {epic}. Reason: {reason}, IG Status: {ig_status} (Ref: {deal_ref})")
                return {'status': 'FAILURE', 'reason': reason, 'ig_status': ig_status}
        except (ApiExceededException, IGException) as api_err:
            logger.error(f"IG API Error executing trade {trade_details.get('epic')}: {api_err}")
            time.sleep(60)
            return {'status': 'FAILURE', 'reason': f'API Error: {api_err}'}
        except Exception as e:
            logger.error(f"Error executing trade {trade_details.get('epic')}: {e}", exc_info=True)
            return {'status': 'ERROR', 'reason': str(e)}

    def close_trade(self, deal_id, size, direction):
        if not self.ig_service:
            return {'status': 'FAILURE', 'reason': 'Not connected'}
        try:
            close_direction = "SELL" if direction == "BUY" else "BUY"
            size_float = float(size)
            logger.info(f"Closing position dealId={deal_id} size={size_float:.2f} direction={close_direction}")
            resp = self.ig_service.close_open_position(
                deal_id=deal_id,
                direction=close_direction,
                size=size_float,
                order_type='MARKET'
            )
            ig_status = resp.get('dealStatus', 'UNKNOWN')
            reason = resp.get('reason', 'N/A')

            if ig_status == 'ACCEPTED':
                return {'status': 'SUCCESS', 'reason': reason, 'dealId': deal_id, 'ig_status': ig_status}
            else:
                logger.error(f"Failed to close {deal_id}. Reason: {reason}, IG Status: {ig_status}")
                return {'status': 'FAILURE', 'reason': reason, 'ig_status': ig_status}
        except (ApiExceededException, IGException) as api_err:
            logger.error(f"IG API Error closing trade {deal_id}: {api_err}")
            time.sleep(60)
            return {'status': 'FAILURE', 'reason': f'API Error: {api_err}'}
        except Exception as e:
            logger.error(f"Error closing trade {deal_id}: {e}", exc_info=True)
            return {'status': 'ERROR', 'reason': str(e)}

    def amend_trade(self, deal_id, stop_level=None, limit_level=None):
        if not self.ig_service:
            return {'status': 'FAILURE', 'reason': 'Not connected'}
        if stop_level is None and limit_level is None:
            logger.warning("Amend trade called with no stop or limit level.")
            return {'status': 'FAILURE', 'reason': 'No levels provided'}

        try:
            logger.info(f"Amending position {deal_id}: stop={stop_level}, limit={limit_level}")
            resp = self.ig_service.update_open_position(
                deal_id=deal_id,
                stop_level=stop_level,
                limit_level=limit_level
            )
            ig_status = resp.get('dealStatus', 'UNKNOWN')
            reason = resp.get('reason', 'N/A')

            if ig_status == 'ACCEPTED':
                return {'status': 'SUCCESS', 'reason': reason, 'dealId': deal_id, 'ig_status': ig_status}
            else:
                logger.error(f"Failed to amend {deal_id}. Reason: {reason}, IG Status: {ig_status}")
                return {'status': 'FAILURE', 'reason': reason, 'ig_status': ig_status}
        except (ApiExceededException, IGException) as api_err:
            logger.error(f"IG API Error amending trade {deal_id}: {api_err}")
            time.sleep(60)
            return {'status': 'FAILURE', 'reason': f'API Error: {api_err}'}
        except Exception as e:
            logger.error(f"Error amending trade {deal_id}: {e}", exc_info=True)
            return {'status': 'ERROR', 'reason': str(e)}
