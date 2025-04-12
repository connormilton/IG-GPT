# risk_manager.py

import logging
import decimal

logger = logging.getLogger("TradingBot")

class RiskManager:
    """Applies risk rules, calculates trade sizing, margin checks, etc."""
    def __init__(self, config):
        self.config = config
        self.balance = decimal.Decimal('0.0')
        self.available_funds = decimal.Decimal('0.0')
        self.account_currency = config['ACCOUNT_CURRENCY']

        self.APPROX_VPP_GBP = {
            "CS.D.EURUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.USDJPY.MINI.IP": decimal.Decimal("0.74"),
            "CS.D.GBPUSD.MINI.IP": decimal.Decimal("0.81"),
            "CS.D.AUDUSD.MINI.IP": decimal.Decimal("0.81"),
        }

    def update_account(self, portfolio_state):
        self.balance = portfolio_state.get_balance()
        self.available_funds = portfolio_state.get_available_funds()
        logger.debug(f"RiskManager updated: Bal={self.balance:.2f}, Avail={self.available_funds:.2f}")

    def _get_value_per_point(self, instrument_details):
        epic = instrument_details.get('epic')
        target_currency = self.account_currency
        vpp_direct = instrument_details.get('valuePerPoint')
        vpp_currency_hint = instrument_details.get('valuePerPointCurrency')

        logger.debug(f"Getting VPP for {epic}. Direct: {vpp_direct}, Currency: {vpp_currency_hint}, Target: {target_currency}")
        if vpp_direct and vpp_direct > 0:
            if vpp_currency_hint == target_currency:
                logger.info(f"Using direct VPP from broker for {epic}: {vpp_direct:.4f} {target_currency}")
                return vpp_direct
            else:
                logger.warning(f"Broker VPP found for {epic} ({vpp_direct:.4f} {vpp_currency_hint}) but needs conversion. NOT IMPLEMENTED. Falling back.")
        if epic in self.APPROX_VPP_GBP:
            approx_vpp = self.APPROX_VPP_GBP[epic]
            logger.warning(f"Using hardcoded approx VPP for {epic}: {approx_vpp:.4f} {target_currency}")
            return approx_vpp

        logger.error(f"Cannot determine VPP for {epic}. Using placeholder VPP=1.0. Sizing may be incorrect.")
        return decimal.Decimal("1.0")

    def calculate_trade_details(self, proposed_trade, instrument_details, broker_interface):
        epic = proposed_trade['symbol']
        logger.debug(f"Calculating trade details for: {epic}, Proposal: {proposed_trade}")

        stop_distance_pips = proposed_trade.get('stop_loss_pips')
        limit_pips = proposed_trade.get('limit_pips')
        signal_price = proposed_trade.get('signal_price')

        if not isinstance(stop_distance_pips, decimal.Decimal) or stop_distance_pips <= 0:
            return None, "Invalid stop loss distance"
        if not isinstance(signal_price, decimal.Decimal):
            return None, "Invalid signal price"

        if limit_pips is not None and (not isinstance(limit_pips, decimal.Decimal) or limit_pips <= 0):
            limit_pips = None

        vpp = self._get_value_per_point(instrument_details)
        if vpp is None or vpp <= 0:
            return None, f"Invalid Value Per Point for {epic}"

        risk_per_trade_pct = self.config['RISK_PER_TRADE_PERCENT'] / 100
        confidence = proposed_trade.get('confidence', 'medium').lower()
        conf_mults = self.config.get('CONFIDENCE_RISK_MULTIPLIERS', {})
        confidence_multiplier = conf_mults.get(confidence, decimal.Decimal('1.0'))

        if self.balance <= 0:
            return None, "Zero or negative account balance"

        max_risk_target_acc_ccy = self.balance * risk_per_trade_pct * confidence_multiplier
        try:
            calculated_size = max_risk_target_acc_ccy / (stop_distance_pips * vpp)
            calculated_size = calculated_size.quantize(decimal.Decimal("0.01"), rounding=decimal.ROUND_DOWN)
            if calculated_size <= 0:
                logger.warning(f"Calculated size for {epic} is zero after rounding.")
                calculated_size = decimal.Decimal('0')
        except (decimal.InvalidOperation, ZeroDivisionError) as e:
            logger.error(f"Size calc error {epic}: {e}")
            return None, "Size calculation error"

        min_deal_size = instrument_details.get('minDealSize', decimal.Decimal("0.1"))
        final_size = max(min_deal_size, calculated_size)
        if calculated_size < min_deal_size:
            logger.warning(f"Calculated size {calculated_size:.2f} < Min {min_deal_size} for {epic}. Using Min.")

        estimated_risk_acc_ccy = final_size * stop_distance_pips * vpp

        direction = proposed_trade['direction']
        stop_level_abs = None
        limit_level_abs = None

        if direction == 'BUY':
            stop_level_abs = float(signal_price - stop_distance_pips)
            if limit_pips:
                limit_level_abs = float(signal_price + limit_pips)
        else:  # SELL
            stop_level_abs = float(signal_price + stop_distance_pips)
            if limit_pips:
                limit_level_abs = float(signal_price - limit_pips)

        final_details = {
            'epic': epic,
            'direction': direction,
            'size': final_size,
            'stop_level': stop_level_abs,
            'limit_level': limit_level_abs,
            'order_type': 'MARKET',
            'estimated_risk_gbp': float(estimated_risk_acc_ccy.quantize(decimal.Decimal("0.01"))),
            'confidence': confidence,
            'signal_price': signal_price,
        }
        return final_details, None

    def check_portfolio_constraints(self, calculated_trade, instrument_details, portfolio_state):
        epic = calculated_trade['epic']
        trade_size = calculated_trade['size']
        signal_price = calculated_trade.get('signal_price')
        margin_factor = instrument_details.get('marginFactor')

        logger.debug(f"Checking constraints for {epic} (Size: {trade_size})...")
        margin_needed = decimal.Decimal('0.0')
        margin_check_passed = False

        if signal_price and margin_factor is not None and margin_factor >= 0 and trade_size > 0:
            try:
                margin_needed = trade_size * signal_price * margin_factor
                margin_buffer_factor = self.config['MARGIN_BUFFER_FACTOR']
                effective_available = self.available_funds * margin_buffer_factor
                logger.debug(f"Margin needed {epic}: {margin_needed:.2f}, Available (<{margin_buffer_factor*100}%): {effective_available:.2f}")

                if margin_needed <= effective_available:
                    margin_check_passed = True
                else:
                    reason = (f"Insufficient margin. Needed: {margin_needed:.2f}, "
                              f"Max Allowed: {effective_available:.2f}")
                    logger.warning(reason)
                    return False, reason
            except Exception as margin_err:
                logger.error(f"Error checking margin for {epic}: {margin_err}")
                return False, "Margin calculation error"
        else:
            logger.warning(f"Cannot calculate margin for {epic}: Missing data.")
            return False, "Missing data for margin calc"

        if margin_check_passed:
            logger.info(f"Trade for {epic} passed constraints.")
            return True, "Risk checks passed"
        else:
            return False, "Unknown constraint failure"
