import os
import json
import logging
import decimal
import re  # Added for regex support

logger = logging.getLogger("TradingBot")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class LLMInterface:
    """Handles LLM calls (OpenAI or Claude) for trade recommendations."""
    def __init__(self, config):
        self.config = config
        self.provider = config.get("LLM_PROVIDER", "OpenAI").lower()
        self.api_key = config.get(f"{self.provider.upper()}_API_KEY")
        self.prompt_dir = config.get("LLM_PROMPT_DIR")
        self.client = None

        if self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                logger.critical("OpenAI library missing!")
            elif not self.api_key:
                logger.critical("OPENAI_API_KEY missing!")
            else:
                openai.api_key = self.api_key
                logger.info("OpenAI setup completed.")
                self.client = openai
        elif self.provider == 'claude':
            if not ANTHROPIC_AVAILABLE:
                logger.critical("Anthropic library missing!")
            elif not self.api_key:
                logger.critical("CLAUDE_API_KEY missing!")
            else:
                logger.info("Anthropic client setup completed.")
                self.client = anthropic
        if not self.client:
            logger.error(f"LLMInterface for {self.provider} failed to initialize client.")

    def _load_prompt_template(self, template_filename):
        filepath = os.path.join(self.prompt_dir, template_filename)
        try:
            if not os.path.exists(filepath):
                logger.error(f"Prompt template '{filepath}' not found.")
                return f"--- PROMPT {template_filename} MISSING ---\nContext:\n{{CONTEXT_JSON}}"
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Could not load prompt template {filepath}: {e}")
            return f"--- ERROR LOADING {template_filename} ---"

    def _call_llm(self, model: str, messages, temperature=0.2, max_tokens=3000, expect_json=False):
        if not self.client:
            logger.error("LLM client not available.")
            return None

        if self.provider == 'openai':
            try:
                # Create a client instance for OpenAI v1.0+
                client = openai.OpenAI(api_key=self.api_key)
                
                # Use the new API format
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract content from the new response format
                content = response.choices[0].message.content
                return content
            except Exception as e:
                logger.error(f"Error calling OpenAI: {e}", exc_info=True)
                return None
        elif self.provider == 'claude':
            try:
                cli = self.client.Client(api_key=self.api_key)
                system_text = ""
                user_text = ""
                for msg in messages:
                    if msg['role'] == 'system':
                        system_text += msg['content'] + "\n"
                    else:
                        user_text += msg['content'] + "\n"
                resp = cli.completions.create(
                    model=model,
                    prompt=f"{anthropic.HUMAN_PROMPT} {system_text}\n{user_text}{anthropic.AI_PROMPT}",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature,
                )
                return resp.completion
            except Exception as e:
                logger.error(f"Error calling Claude: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            return None

    def get_trade_recommendations(self, portfolio_state, market_snapshots, history_summary):
        logger.info("Querying LLM for trade recommendations...")
        template_name = "combined_trade_analysis.txt"
        template = self._load_prompt_template(template_name)

        default_response = {
            "tradeActions": [],
            "tradeAmendments": [],
            "reasoning": {}
        }
        if not template:
            logger.error(f"Cannot get recommendations: Prompt template '{template_name}' missing.")
            return default_response

        try:
            # Safely convert MAX_TOTAL_RISK_PERCENT from the config.
            max_total_risk_raw = self.config.get('MAX_TOTAL_RISK_PERCENT')
            try:
                if isinstance(max_total_risk_raw, decimal.Decimal):
                    max_total_risk_str = f"{max_total_risk_raw:.2f}"
                elif max_total_risk_raw is not None:
                    # Convert to string first to handle various input types
                    max_total_risk_str = str(max_total_risk_raw).strip()
                    # Now attempt decimal conversion
                    max_total_risk = decimal.Decimal(max_total_risk_str)
                    max_total_risk_str = f"{max_total_risk:.2f}"
                else:
                    # Handle None case
                    logger.warning("MAX_TOTAL_RISK_PERCENT not found in config")
                    max_total_risk_str = "5.00"  # Default value
            except Exception as e:
                logger.warning(f"Invalid MAX_TOTAL_RISK_PERCENT value: {max_total_risk_raw} - {str(e)}")
                max_total_risk_str = "5.00"  # Fallback to a default value.

            # Get currency risk cap from config or use default
            per_currency_risk_cap = self.config.get('PER_CURRENCY_RISK_CAP')
            try:
                if isinstance(per_currency_risk_cap, decimal.Decimal):
                    per_currency_risk_cap_str = f"{per_currency_risk_cap:.2f}"
                elif per_currency_risk_cap is not None:
                    per_currency_risk_cap_str = str(per_currency_risk_cap).strip()
                    per_currency_risk_cap = decimal.Decimal(per_currency_risk_cap_str)
                    per_currency_risk_cap_str = f"{per_currency_risk_cap:.2f}"
                else:
                    # Default value if not in config
                    logger.warning("PER_CURRENCY_RISK_CAP not found in config, using default")
                    per_currency_risk_cap_str = "3.00"  # Default value
            except Exception as e:
                logger.warning(f"Invalid PER_CURRENCY_RISK_CAP value: {per_currency_risk_cap} - {str(e)}")
                per_currency_risk_cap_str = "3.00"  # Fallback to a default value

            # Safely format market snapshot data
            formatted_snapshots = {}
            for k, v in market_snapshots.items():
                if not v:
                    continue
                
                formatted_snapshots[k] = {}
                for kk, vv in v.items():
                    if vv is None:
                        formatted_snapshots[k][kk] = "null"
                        continue
                    
                    # Try to format as decimal, fall back to string representation
                    try:
                        if isinstance(vv, (int, float, str)) and str(vv).strip():
                            dec_val = decimal.Decimal(str(vv))
                            formatted_snapshots[k][kk] = f"{dec_val:.6f}"
                        else:
                            formatted_snapshots[k][kk] = str(vv)
                    except Exception as e:
                        logger.debug(f"Could not convert {vv} to Decimal: {e}")
                        formatted_snapshots[k][kk] = str(vv)
            
            # Special handling for strange keys that may be in the template
            # Add weird keys with spaces and quotes that might be in the template
            special_keys = {
                ' "epic"': '" epic"',
                ' "action"': '" action"',
                ' "confidence"': '" confidence"',
                ' "stop_distance"': '" stop_distance"',
                ' "limit_distance"': '" limit_distance"',
                ' "size"': '" size"'
            }
            
            context_data = {
                "ACCOUNT_BALANCE": f"{portfolio_state.get_balance():.2f}",
                "AVAILABLE_MARGIN": f"{portfolio_state.get_available_funds():.2f}",
                "ACCOUNT_CURRENCY": self.config['ACCOUNT_CURRENCY'],
                "RISK_PER_TRADE_PERCENT": f"{self.config['RISK_PER_TRADE_PERCENT']:.2f}",
                "MAX_TOTAL_RISK_PERCENT": max_total_risk_str,
                "PER_CURRENCY_RISK_CAP": per_currency_risk_cap_str,
                "OPEN_POSITIONS_JSON": json.dumps(portfolio_state.get_open_positions_dict(), indent=2),
                "MARKET_SNAPSHOT_JSON": json.dumps(formatted_snapshots, indent=2),
                "TRADE_HISTORY_JSON": json.dumps(history_summary, indent=2),
                "N_RECENT_TRADES": self.config['N_RECENT_TRADES_FEEDBACK']
            }
            
            # Add all special keys to the context data
            for special_key, value in special_keys.items():
                context_data[special_key] = value
            
            try:
                # First attempt direct formatting
                prompt = template.format(**context_data)
            except ValueError as val_err:
                # Handle special case with invalid format specifiers
                logger.warning(f"Format specifier error: {val_err}")
                
                # Try handling invalid format specifiers by replacing curly braces in strings
                # This is a common issue when templates contain JSON examples with format specifiers
                try:
                    # Load template again
                    raw_template = self._load_prompt_template(template_name)
                    
                    # Replace all keys with their values, handling each one individually
                    prompt = raw_template
                    for key, value in context_data.items():
                        placeholder = '{' + key + '}'
                        if placeholder in prompt:
                            prompt = prompt.replace(placeholder, str(value))
                    
                    # Handle remaining complex format specifiers by escaping them to prevent future errors
                    # Look for patterns that might be causing problems
                    # 1. Find remaining curly braces that might be part of JSON examples
                    if '{' in prompt and '}' in prompt:
                        # Only warn if we expect there might be actual unprocessed placeholders
                        # Don't warn for JSON examples or escaped braces
                        if re.search(r'\{[A-Z_]+\}', prompt):
                            logger.warning("Not all placeholders were replaced in the template")
                except Exception as replace_err:
                    logger.error(f"Error with manual template replacement: {replace_err}", exc_info=True)
                    return default_response
            except KeyError as e:
                # If there's still a missing key, log it and examine the template to find what's needed
                missing_key = str(e).strip("'")
                logger.error(f"Prompt '{template_name}' missing key: {missing_key}")
                
                # Try to load the template to see what's needed
                try:
                    with open(os.path.join(self.prompt_dir, template_name), 'r', encoding='utf-8') as f:
                        template_content = f.read()
                        # Log a small part of the template around where the missing key might be
                        search_key = '{' + missing_key + '}'
                        pos = template_content.find(search_key)
                        if pos >= 0:
                            context = template_content[max(0, pos-50):min(len(template_content), pos+50)]
                            logger.debug(f"Context around missing key: '...{context}...'")
                except Exception as read_err:
                    logger.error(f"Could not examine template: {read_err}")
                
                return default_response
        except KeyError as e:
            logger.error(f"Prompt '{template_name}' missing key: {e}")
            return default_response
        except Exception as fmt_err:
            logger.error(f"Error formatting prompt '{template_name}': {fmt_err}", exc_info=True)
            return default_response

        system_prompt_content = (
            "You are an AI trading analyst for Forex Majors, trading on a GBP account via IG.\n"
            "Provide actions/amendments in strict JSON. Ensure 'stop_distance' is always positive.\n"
            "Base decisions ONLY on provided context.\n"
        )
        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": prompt}
        ]
        response_content = self._call_llm(
            model=self.config['LLM_MODEL_ANALYSIS'],
            messages=messages,
            expect_json=True
        )
        if response_content:
            try:
                parsed = json.loads(response_content)
                validated = default_response.copy()
                validated["reasoning"] = parsed.get("reasoning", {}) if isinstance(parsed.get("reasoning"), dict) else {}

                raw_actions = parsed.get("tradeActions", [])
                if isinstance(raw_actions, list):
                    for action in raw_actions:
                        if (
                            isinstance(action, dict)
                            and action.get('epic')
                            and action.get('action') in ['BUY', 'SELL']
                            and action.get('stop_distance') is not None
                        ):
                            try:
                                action['stop_loss_pips'] = decimal.Decimal(str(action['stop_distance']))
                                if action['stop_loss_pips'] <= 0:
                                    raise ValueError("Stop distance must be positive")
                                if action.get('limit_distance') is not None:
                                    action['limit_pips'] = decimal.Decimal(str(action['limit_distance']))
                                else:
                                    action['limit_pips'] = None
                                action['confidence'] = action.get('confidence', 'medium').lower()
                                validated["tradeActions"].append(action)
                            except (ValueError, decimal.InvalidOperation) as conv_err:
                                logger.warning(f"Invalid numeric distance in action: {action} -> {conv_err}")
                        else:
                            logger.warning(f"Skipping invalid tradeAction structure: {action}")

                raw_amendments = parsed.get("tradeAmendments", [])
                if isinstance(raw_amendments, list):
                    for amend in raw_amendments:
                        # Check for either epic or dealId
                        has_identifier = amend.get('epic') or amend.get('dealId')
                        if (
                            isinstance(amend, dict)
                            and has_identifier
                            and amend.get('action') in ['CLOSE', 'AMEND', 'BREAKEVEN']
                        ):
                            try:
                                # If we have dealId but not epic, copy it to epic for consistency
                                if amend.get('dealId') and not amend.get('epic'):
                                    amend['epic'] = amend['dealId']
                                    
                                if 'new_stop_distance' in amend and amend['new_stop_distance'] is not None:
                                    amend['new_stop_distance_dec'] = decimal.Decimal(str(amend['new_stop_distance']))
                                if 'new_limit_distance' in amend and amend['new_limit_distance'] is not None:
                                    amend['new_limit_distance_dec'] = decimal.Decimal(str(amend['new_limit_distance']))
                                validated["tradeAmendments"].append(amend)
                            except (ValueError, decimal.InvalidOperation) as conv_err:
                                logger.warning(f"Invalid numeric distance in amendment: {amend} -> {conv_err}")
                        else:
                            logger.warning(f"Skipping invalid amendment structure: {amend}")

                logger.info(
                    f"LLM Recommends: {len(validated['tradeActions'])} New, "
                    f"{len(validated['tradeAmendments'])} Amend. "
                    f"Reasoning keys: {list(validated['reasoning'].keys())}"
                )
                return validated
            except (json.JSONDecodeError, TypeError) as json_err:
                logger.error(f"LLM JSON parsing error: {json_err}\nResponse:\n{response_content}")
        return default_response