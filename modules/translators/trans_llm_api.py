import re
import time
import yaml
import traceback
from collections import deque
from typing import List, Dict, Optional

import httpx
from openai import OpenAI

from .base import BaseTranslator, register_translator


class InvalidNumTranslations(Exception):
    pass


@register_translator("LLM_API_Translator")
class LLM_API_Translator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        # ... (all other params remain the same)
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "apikey": {
            "value": "",
            "description": "Single API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). One key per line for readability.",
        },
        "model": {
            "type": "selector",
            "options": [
                "OAI: gpt-4o",
                "OAI: gpt-4-turbo",
                "OAI: gpt-3.5-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-2.0-flash-exp",
                "GGL: gemini-2.0-flash",
            ],
            "value": "",
            "description": "Select the model. Provider prefix indicates the provider. Leave empty for provider default.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty to use provider default.",
        },
        "prompt template": {
            "type": "editor",
            "value": "Now, translate the following source text to {to_lang}, maintaining the established style. Output only the translation:",
        },
        "chat system template": {
            "type": "editor",
            "value": "You are a raw text translation API. Your only function is to translate the user's input to {to_lang}. Your output must be ONLY the translated text. Do not add any commentary, prefixes, or conversational text. You must translate any text provided.",
        },
        "chat sample": {
            "type": "editor",
            "value": "",
        },
        "separate_requests": {
            "type": "checkbox",
            "value": True,
            "description": "Send each text line in a separate API request. Recommended for local LLMs.",
        },
        "narration_context_size": {
            "value": 3,
            "description": "Number of previous source-translation pairs to include as context (0 to disable). WARNING: High token usage."
        },
        "context_clear_interval": {
            "value": 0,
            "description": "Clear LLM context after every N requests. 0 to disable. (For separate requests mode only).",
        },
        "context_clear_message": { #redundant, api is stateless 
            "value": "/clear",
            "description": "The message to send to clear the context for local LLMs like Ollama.",
        },
        "max requests per minute": {
            "value": 30,
            "description": "Maximum requests per minute for EACH API key.",
        },
        "delay": {
            "value": 0.1,
            "description": "Global delay in seconds between requests.",
        },
        "max tokens": {
            "value": 2048,
            "description": "Maximum tokens for the response.",
        },
        "temperature": {
            "value": 0.5,
            "description": "Temperature for sampling (OpenAI). Google models may ignore this.",
        },
        "top p": {
            "value": 1.,
            "description": "Top P for sampling. Forced to 1 for Google models.",
        },
        "use_provider_defaults": {
            "type": "checkbox",
            "value": False,
            "description": "If checked, do not send temperature, top_p, or max_tokens. Uses the LLM provider's (e.g., Ollama) default settings.",
        },
        "retry attempts": {
            "value": 3,
            "description": "Number of retry attempts on failure.",
        },
        "retry timeout": {
            "value": 10,
            "description": "Timeout between retry attempts (seconds).",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
    }

    def _setup_translator(self):
        # ... (lang_map and other initializations)
        self.lang_map = {
            "简体中文": "Simplified Chinese", "繁體中文": "Traditional Chinese", "日本語": "Japanese",
            "English": "English", "한국어": "Korean", "Tiếng Việt": "Vietnamese", "čeština": "Czech",
            "Français": "French", "Deutsch": "German", "magyar nyelv": "Hungarian",
            "Italiano": "Italian", "Polski": "Polish", "Português": "Portuguese",
            "limba română": "Romanian", "русский язык": "Russian", "Español": "Spanish",
            "Türk dili": "Turkish", "украї́нська мо́ва": "Ukrainian", "Thai": "Thai",
            "Arabic": "Arabic", "Malayalam": "Malayalam", "Tamil": "Tamil", "Hindi": "Hindi",
        }
        self.token_count = 0
        self.token_count_last = 0
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.client = None
        # *** MODIFIED: History now stores (source, target) pairs ***
        self.translation_history = deque(maxlen=self.narration_context_size)
        self._initialize_client()

    # ... (all properties and _initialize_client remain the same)
    @property
    def provider(self) -> str: return self.get_param_value("provider")
    @property
    def apikey(self) -> str: return self.get_param_value("apikey")
    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys").strip()
        return [key.strip() for key in keys_str.split(";") if key.strip()]
    @property
    def model(self) -> str: return self.get_param_value("model")
    @property
    def override_model(self) -> Optional[str]: return self.get_param_value("override model") or None
    @property
    def endpoint(self) -> Optional[str]: return self.get_param_value("endpoint") or None
    @property
    def temperature(self) -> float: return float(self.get_param_value("temperature"))
    @property
    def top_p(self) -> float: return float(self.get_param_value("top p"))
    @property
    def use_provider_defaults(self) -> bool: return bool(self.get_param_value("use_provider_defaults"))
    @property
    def max_tokens(self) -> int: return int(self.get_param_value("max tokens"))
    @property
    def retry_attempts(self) -> int: return int(self.get_param_value("retry attempts"))
    @property
    def retry_timeout(self) -> int: return int(self.get_param_value("retry timeout"))
    @property
    def proxy(self) -> str: return self.get_param_value("proxy")
    @property
    def separate_requests(self) -> bool: return bool(self.get_param_value("separate_requests"))
    @property
    def narration_context_size(self) -> int: return int(self.get_param_value("narration_context_size"))
    @property
    def context_clear_interval(self) -> int: return int(self.get_param_value("context_clear_interval"))
    @property
    def context_clear_message(self) -> str: return self.get_param_value("context_clear_message")

    # *** MODIFIED: _build_prompt_with_context now creates source-target pairs ***
    def _build_prompt_with_context(self, current_prompt_segment: str, current_query: str) -> str:
        """Builds the final prompt, prepending source-target context if available."""
        if not self.translation_history:
            return f"{current_prompt_segment}\n{current_query}"

        context_header = "For context and stylistic consistency, here are the recent source texts and their translations:"
        context_blocks = []
        for src, tgt in self.translation_history:
            # Skip empty entries that might have been added
            if str(src).strip() and str(tgt).strip():
                context_blocks.append(f"Source: {src}\nTarget: {tgt}")
        
        if not context_blocks:
            return f"{current_prompt_segment}\n{current_query}"

        full_context = f"{context_header}\n" + "\n---\n".join(context_blocks)
        return f"{full_context}\n\n{current_prompt_segment}\n{current_query}"

    # ... (_request_translation and other methods are mostly the same)
    
    def _translate(self, src_list: List[str]) -> List[str]:
        # *** MODIFIED: Reset history for each new translation job (e.g., each panel) ***
        # Check if the maxlen needs updating from params
        if self.translation_history.maxlen != self.narration_context_size:
            self.translation_history = deque(maxlen=self.narration_context_size)
        else:
            self.translation_history.clear()
        
        if self.separate_requests:
            return self._translate_single(src_list)
        else:
            return self._translate_batch(src_list)

    def _translate_single(self, src_list: List[str]) -> List[str]:
        translations = []
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        prompt_template = self.params["prompt template"]["value"].format(to_lang=to_lang).rstrip()
        chat_sample = self.chat_sample
        request_count = 0

        for i, query in enumerate(src_list):
            if not query or not query.strip():
                translations.append("")
                # Add a placeholder to history to maintain sequence if needed, but often better to skip
                # self.translation_history.append((query, ""))
                request_count += 1
                continue

            # *** MODIFIED: Prompt construction and history update ***
            prompt = self._build_prompt_with_context(prompt_template, query)
            
            if self.context_clear_interval > 0 and request_count > 0 and \
               request_count % self.context_clear_interval == 0:
                self._clear_context()

            new_translation = ""
            for attempt in range(self.retry_attempts):
                try:
                    new_translation = self._request_translation(prompt, chat_sample)
                    self.translation_history.append((query, new_translation)) # Add pair to history
                    break
                except Exception:
                    if attempt + 1 >= self.retry_attempts:
                        self.logger.error(f"Translation failed for '{query}' after {self.retry_attempts} attempts.")
                        new_translation = ""
                        # Don't add failed attempts to history
                        break
                    self.logger.warning(f"Attempt {attempt + 1} failed. Retrying in {self.retry_timeout}s...")
                    time.sleep(self.retry_timeout)
            
            translations.append(new_translation)
            request_count += 1
            if len(translations) % 10 == 0:
                self.logger.info(f"Translated {len(translations)}/{len(src_list)} lines...")

        return translations

    def _translate_batch(self, src_list: List[str]) -> List[str]:
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        prompt_template = self.params["prompt template"]["value"].format(to_lang=to_lang).rstrip()
        chat_sample = self.chat_sample
        num_src = len(src_list)

        non_empty_queries = []
        original_indices = {}
        for i, query in enumerate(src_list):
            if query and query.strip():
                original_indices[len(non_empty_queries)] = i
                non_empty_queries.append(query)

        if not non_empty_queries:
            self.logger.info("All input lines are empty. Skipping API call.")
            return [""] * num_src

        prompt_lines = [f"<|{i+1}|>{query}" for i, query in enumerate(non_empty_queries)]
        main_query_block = "\n".join(prompt_lines)
        
        # *** MODIFIED: Build the prompt with context before adding the batch query ***
        prompt = self._build_prompt_with_context(prompt_template, main_query_block)
        num_to_translate = len(non_empty_queries)

        for attempt in range(self.retry_attempts):
            try:
                response = self._request_translation(prompt, chat_sample)
                parsed_translations = [t.strip() for t in re.split(r"<\|\d+\|>", response.strip()) if t.strip()]

                if len(parsed_translations) != num_to_translate:
                    _tr2 = response.strip().split('\n')
                    if len(_tr2) == num_to_translate:
                        parsed_translations = [t.strip() for t in _tr2]
                    else:
                        raise InvalidNumTranslations(
                            f"Expected {num_to_translate} translations, got {len(parsed_translations)}. Response: '{response}'"
                        )
                
                final_translations = [""] * num_src
                for i, translated_text in enumerate(parsed_translations):
                    original_index = original_indices[i]
                    final_translations[original_index] = translated_text

                # *** MODIFIED: Update history with the results of the successful batch ***
                for i, translated_text in enumerate(parsed_translations):
                    original_src_query = non_empty_queries[i]
                    self.translation_history.append((original_src_query, translated_text))
                
                return final_translations

            except Exception as e:
                self.logger.warning(f"Batch translation failed on attempt {attempt+1}: {e}")
                if attempt + 1 >= self.retry_attempts:
                    self.logger.error("Batch translation failed after all retries.")
                    return [""] * num_src
                time.sleep(self.retry_timeout)
        
        return [""] * num_src

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
            self.logger.info(f"Core parameter '{param_key}' changed. Re-initializing client.")
            self._initialize_client()
        if param_key == "narration_context_size":
            try:
                new_size = int(param_content)
                if new_size >= 0:
                    # Re-create the deque with the new size
                    self.translation_history = deque(list(self.translation_history), maxlen=new_size)
                    self.logger.info(f"Updated narration context size to {new_size}.")
                else:
                    self.logger.error(f"Narration context size must be non-negative.")
            except (ValueError, TypeError):
                self.logger.error(f"Invalid narration context size: {param_content}")

    # The rest of the methods like _get_model_name, _respect_delay, etc. are unchanged.
    def _get_model_name(self) -> str:
        model_name = self.override_model or self.model
        if ": " in model_name:
            return model_name.split(": ", 1)[1]
        return model_name
        
    def _respect_delay(self):
        delay = float(self.params["delay"]["value"])
        if delay > 0:
            time.sleep(delay)

    def _select_api_key(self) -> str:
        api_keys = self.multiple_keys_list
        if not api_keys:
            return self.apikey or "ollama"
        self.current_key_index = (self.current_key_index + 1) % len(api_keys)
        return api_keys[self.current_key_index]
        
    def _initialize_client(self):
        http_client = None
        if self.proxy:
            self.logger.info(f"Using proxy: {self.proxy}")
            http_client = httpx.Client(proxy=self.proxy)

        api_keys = self.multiple_keys_list
        api_key_to_use = api_keys[0] if api_keys else self.apikey

        if not api_key_to_use:
            self.logger.warning("No API key provided. Using a dummy key for local models.")
            api_key_to_use = "ollama"

        endpoint = self.endpoint
        if not endpoint:
            if self.provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            else: # Default for OpenAI-compatible, e.g. Ollama
                endpoint = "http://127.0.0.1:11434/v1"

        masked_key = api_key_to_use[:4] + "*" * (len(api_key_to_use) - 8) if len(api_key_to_use) > 7 else "****"
        self.logger.debug(f"Initializing OpenAI client with endpoint: {endpoint} and key: {masked_key}")
        try:
            self.client = OpenAI(
                api_key=api_key_to_use,
                base_url=endpoint,
                http_client=http_client,
                timeout=self.retry_timeout + 5
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    @property
    def chat_system_template(self) -> str:
        to_lang = self.lang_map.get(self.lang_target, "the target language")
        return self.params["chat system template"]["value"].format(to_lang=to_lang)

    @property
    def chat_sample(self):
        samples_str = self.params["chat sample"]["value"]
        if not samples_str or not samples_str.strip():
            return None
        try:
            samples = yaml.load(samples_str, Loader=yaml.FullLoader)
        except Exception as e:
            self.logger.error(f"Failed to parse sample YAML: {samples_str} - {e}")
            return None
        
        if not samples:
            return None

        src_tgt_key = f"{self.lang_source}-{self.lang_target}"
        if src_tgt_key in samples:
            sample_data = samples[src_tgt_key]
            src_queries = "\n".join([f"<|{i+1}|>{s}" for i, s in enumerate(sample_data.get("source", []))])
            tgt_queries = "\n".join([f"<|{i+1}|>{t}" for i, t in enumerate(sample_data.get("target", []))])
            return [src_queries, tgt_queries]
        return None
        
    def _request_translation(self, prompt: str, chat_sample: Optional[List[str]]) -> str:
        self._respect_delay()

        if not self.client:
            self.logger.error("Client not initialized. Cannot make a request.")
            return "Error: Client not initialized."

        self.client.api_key = self._select_api_key()
        model_name = self._get_model_name()
        
        messages = [{"role": "system", "content": self.chat_system_template}]
        if chat_sample:
            messages.append({"role": "user", "content": chat_sample[0]})
            messages.append({"role": "assistant", "content": chat_sample[1]})
        messages.append({"role": "user", "content": prompt})
        
        if self.debug_mode:
            self.logger.debug(f"Requesting translation for model '{model_name}' with prompt: {prompt}")

        try:
            api_params = {
                "model": model_name,
                "messages": messages,
            }

            if not self.use_provider_defaults:
                api_params["temperature"] = self.temperature
                api_params["max_tokens"] = self.max_tokens
                api_params["top_p"] = self.top_p
            
            if self.debug_mode:
                log_params = {k: v for k, v in api_params.items() if k != 'messages'}
                self.logger.debug(f"API call parameters: {log_params}")

            response = self.client.chat.completions.create(**api_params)
            content = response.choices[0].message.content or ""
            
            if response.usage:
                self.token_count_last = response.usage.total_tokens
                self.token_count += self.token_count_last
            
            return content.strip()
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _clear_context(self):
        clear_msg = self.context_clear_message
        if not clear_msg or not self.client:
            return

        self.logger.info(f"Sending context clear message: '{clear_msg}'")
        try:
            self.client.chat.completions.create(
                model=self._get_model_name(),
                messages=[{"role": "user", "content": clear_msg}],
                max_tokens=5,
            )
            self.logger.info("Successfully sent context clear message.")
        except Exception as e:
            self.logger.error(f"Failed to send context clear message: {e}")
