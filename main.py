# main.py

import os
import time
import logging
from datetime import datetime, timezone

# Local imports
from config_loader import load_and_configure, get_config
from logging_setup import setup_logging
from ig_interface import IGInterface
from data_provider import DataProvider
from risk_manager import RiskManager
from llm_interface import LLMInterface
from portfolio import Portfolio
from trade_executor import ExecutionHandler
from trading_ig.rest import ApiExceededException

logger = logging.getLogger("TradingBot")

def process_llm_recommendations(recommendations, portfolio, risk_manager, data_provider, broker, executor):
    open_positions_df = portfolio.get_open_positions_df()

    # --- Amendments ---
    trade_amendments = recommendations.get("tradeAmendments", [])
    if trade_amendments:
        logger.info(f"Processing {len(trade_amendments)} LLM amendments...")
        for amend in trade_amendments:
            epic = amend.get("epic")
            action = amend.get("action")
            if not epic or not action:
                logger.warning("Skipping invalid amendment structure (no epic/action)")
                continue
                
            # Check if the epic might actually be a dealId
            position_rows = open_positions_df[open_positions_df['epic'] == epic]
            
            # If no position found by epic, try looking up by dealId instead
            if position_rows.empty and epic.startswith('DI'):
                position_rows = open_positions_df[open_positions_df['dealId'] == epic]
                if not position_rows.empty:
                    # If found by dealId, log this for clarity
                    logger.info(f"Position found using dealId instead of epic: {epic}")
            
            if position_rows.empty:
                logger.warning(f"Cannot {action} {epic}: No open position.")
                continue

            position = position_rows.iloc[0].to_dict()
            deal_id = position.get('dealId')
            pos_size = position.get('size')
            pos_dir = position.get('direction')
            entry_level = position.get('level')

            if not all([deal_id, pos_size is not None, pos_dir]):
                logger.warning(f"Missing critical data for open position {epic}")
                continue

            if action == "CLOSE":
                logger.info(f"LLM Recommends CLOSE for {epic} (DealID: {deal_id})")
                executor.close_trade(deal_id, pos_size, pos_dir, epic=epic)

            elif action == "AMEND":
                new_stop_dist_dec = amend.get("new_stop_distance_dec")
                new_limit_dist_dec = amend.get("new_limit_distance_dec")
                if new_stop_dist_dec is None and new_limit_dist_dec is None:
                    logger.warning(f"AMEND action for {epic} has no new distances.")
                    continue

                snapshot = broker.fetch_market_snapshot(epic)
                # If the epic is actually a dealId, we need to use the real epic from the position
                if epic.startswith('DI') and 'epic' in position:
                    real_epic = position['epic']
                    snapshot = broker.fetch_market_snapshot(real_epic)
                    
                if not snapshot or snapshot.get('bid') is None:
                    logger.warning(f"Cannot calculate AMEND levels for {epic}: No snapshot.")
                    continue

                current_price = snapshot['offer'] if pos_dir == 'BUY' else snapshot['bid']
                new_stop_level, new_limit_level = None, None
                try:
                    if new_stop_dist_dec is not None:
                        new_stop_level_dec = (current_price - new_stop_dist_dec) if pos_dir == 'BUY' else (current_price + new_stop_dist_dec)
                        current_stop_dec = position.get('stopLevel')
                        if current_stop_dec is not None:
                            is_loss_increasing = (
                                (pos_dir == 'BUY' and new_stop_level_dec < current_stop_dec) or
                                (pos_dir == 'SELL' and new_stop_level_dec > current_stop_dec)
                            )
                            if is_loss_increasing:
                                logger.warning(f"REJECTING AMEND Stop for {epic}: New stop {new_stop_level_dec} increases risk from {current_stop_dec}.")
                            else:
                                new_stop_level = float(new_stop_level_dec)
                        else:
                            new_stop_level = float(new_stop_level_dec)

                    if new_limit_dist_dec is not None:
                        new_limit_level_dec = (current_price + new_limit_dist_dec) if pos_dir == 'BUY' else (current_price - new_limit_dist_dec)
                        new_limit_level = float(new_limit_level_dec)

                    if new_stop_level or new_limit_level:
                        logger.info(f"LLM Recommends AMEND for {epic} (DealID: {deal_id}): Stop={new_stop_level}, Limit={new_limit_level}")
                        executor.amend_trade(deal_id, new_stop_level, new_limit_level, epic=epic)
                    else:
                        logger.info(f"No valid levels to AMEND for {epic} after safety check.")
                except Exception as calc_err:
                    logger.error(f"Error calculating AMEND levels for {epic}: {calc_err}", exc_info=True)

            elif action == "BREAKEVEN":
                if entry_level is not None:
                    logger.info(f"LLM Recommends BREAKEVEN for {epic} (DealID: {deal_id})")
                    executor.amend_trade(deal_id, stop_level=float(entry_level), limit_level=None, epic=epic)
                else:
                    logger.warning(f"Cannot set breakeven for {epic}: Entry level missing.")
            else:
                logger.warning(f"Unsupported amendment action '{action}' for {epic}.")
            time.sleep(0.5)
    else:
        logger.info("No LLM amendments to process.")

    # --- New Trades ---
    trade_actions = recommendations.get("tradeActions", [])
    if trade_actions:
        logger.info(f"Processing {len(trade_actions)} LLM new trade actions...")
        for action in trade_actions:
            epic = action.get("epic")
            direction = action.get("action")
            stop_dist_dec = action.get("stop_loss_pips")
            limit_dist_dec = action.get("limit_pips")
            confidence = action.get("confidence")

            if not epic or not direction or stop_dist_dec is None:
                logger.warning(f"Skipping invalid action structure: {action}")
                continue

            if not open_positions_df[open_positions_df['epic'] == epic].empty:
                logger.info(f"Skipping new {direction} for {epic}: Position exists.")
                continue

            instrument_details = broker.get_instrument_details(epic)
            snapshot = broker.fetch_market_snapshot(epic)
            if not instrument_details or not snapshot or snapshot.get('bid') is None:
                logger.warning(f"Skipping {epic}: Missing details or snapshot.")
                continue

            signal_price = snapshot['offer'] if direction == 'BUY' else snapshot['bid']
            proposed_trade = {
                'symbol': epic,
                'direction': direction,
                'signal_price': signal_price,
                'stop_loss_pips': stop_dist_dec,
                'limit_pips': limit_dist_dec,
                'confidence': confidence,
            }

            final_trade_details, calc_reason = risk_manager.calculate_trade_details(proposed_trade, instrument_details, broker)
            if not final_trade_details:
                logger.warning(f"Trade calc failed for {epic}: {calc_reason}")
                continue

            is_viable, constraint_reason = risk_manager.check_portfolio_constraints(final_trade_details, instrument_details, portfolio)
            if is_viable:
                logger.info(f"Executing viable trade for {epic}...")
                success, result = executor.execute_new_trade(final_trade_details)
                if success:
                    logger.info(f"Trade submitted successfully for {epic}. Deal ID: {result}")
                    portfolio.update_state()
                    risk_manager.update_account(portfolio)
                    time.sleep(1.5)
                else:
                    logger.error(f"Execution failed for {epic}: {result}")
            else:
                logger.warning(f"Trade for {epic} rejected by constraints: {constraint_reason}")
    else:
        logger.info("No new LLM trade actions to process.")

def run_trading_bot():
    logger.info(f"🚀 Initializing Turnkey LLM Trader (Refactored) PID: {os.getpid()}")
    config = get_config()

    try:
        broker = IGInterface(config)
        data_provider = DataProvider(config)
        llm_interface = LLMInterface(config)
        portfolio = Portfolio(broker, config)
        risk_manager = RiskManager(config)
        executor = ExecutionHandler(broker, portfolio, config)

        # Initial updates
        portfolio.update_state()
        risk_manager.update_account(portfolio)
    except Exception as init_err:
        logger.critical(f"Initialization failed: {init_err}", exc_info=True)
        return

    while True:
        cycle_start_time = time.time()
        logger.info(f"--- Cycle {datetime.now(timezone.utc).isoformat(timespec='seconds')} ---")
        try:
            portfolio.update_state()
            if portfolio.get_balance() <= 0:
                logger.error("Balance zero or negative. Stopping.")
                break

            risk_manager.update_account(portfolio)
            assets_to_analyze = config.get('INITIAL_ASSET_FOCUS_EPICS', [])
            if not assets_to_analyze:
                logger.error("No assets configured. Stopping.")
                break

            market_snapshots = {}
            for epic in assets_to_analyze:
                snapshot = broker.fetch_market_snapshot(epic)
                if snapshot:
                    market_snapshots[epic] = snapshot
                time.sleep(0.1)

            valid_snapshots = {
                e: s for e, s in market_snapshots.items() if s and s.get('bid') is not None
            }

            if not valid_snapshots:
                logger.warning("No valid snapshots. Skipping LLM call.")
                recommendations = {"tradeActions": [], "tradeAmendments": []}
            else:
                recommendations = llm_interface.get_trade_recommendations(
                    portfolio,
                    valid_snapshots,
                    portfolio.get_recent_trade_summary()
                )

            process_llm_recommendations(
                recommendations,
                portfolio,
                risk_manager,
                data_provider,
                broker,
                executor
            )

        except KeyboardInterrupt:
            logger.info("Stop requested by user.")
            break
        except ApiExceededException:
            logger.error("IG API Rate Limit hit. Pausing 60s.")
            time.sleep(60)
        except Exception as loop_err:
            logger.critical("Unhandled exception in main loop!", exc_info=True)
            logger.error("Pausing 30s before retry.")
            time.sleep(30)

        cycle_duration = time.time() - cycle_start_time
        sleep_time = max(10, config['TRADING_CYCLE_SECONDS'] - cycle_duration)
        logger.info(f"Cycle ended (Duration: {cycle_duration:.2f}s). Sleeping {sleep_time:.2f}s...")
        time.sleep(sleep_time)

def main():
    try:
        load_and_configure()  # This populates the global CONFIG in config_loader
        config = get_config()
        logger_ = setup_logging(config)
        run_trading_bot()
    except Exception as e:
        logger.critical(f"Fatal error starting up: {e}", exc_info=True)

if __name__ == "__main__":
    main()