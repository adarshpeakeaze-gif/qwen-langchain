"""
LangChain classifier agent for document type classification and structured extraction.
Includes key rotation and retry logic for high-throughput processing.
All models use OpenRouter API.
"""

import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config import config
from .tools import EXTRACTION_TOOLS
from prompts import CLASSIFIER_SYSTEM_PROMPT, ENTERPRISE_EXTRACTION_PROMPT

# Setup logging
logger = logging.getLogger(__name__)

# OpenRouter base URL for LangChain (without /chat/completions)
OPENROUTER_BASE_URL_V1 = "https://openrouter.ai/api/v1"


# =============================================================================
# Key Rotation Manager
# =============================================================================

class KeyRotator:
    """
    Manages multiple API keys with rotation and health tracking.
    Handles rate limiting and temporary failures gracefully.
    """

    def __init__(self, keys: List[str], name: str = "api"):
        self.name = name
        self.keys = keys
        self.current_index = 0
        self.failed_keys: Dict[str, datetime] = {}  # key -> cooldown_until
        self.request_counts: Dict[str, int] = {k: 0 for k in keys}
        self.last_reset: datetime = datetime.now()
        self.cooldown_seconds = 60

    def _reset_counts_if_needed(self):
        """Reset request counts every minute."""
        now = datetime.now()
        if (now - self.last_reset).seconds >= 60:
            self.request_counts = {k: 0 for k in self.keys}
            self.last_reset = now
            # Clear expired cooldowns
            self.failed_keys = {
                k: v for k, v in self.failed_keys.items()
                if v > now
            }

    def get_key(self) -> Optional[str]:
        """Get next available API key using round-robin with health checks."""
        if not self.keys:
            return None

        self._reset_counts_if_needed()
        now = datetime.now()

        # Try each key
        for _ in range(len(self.keys)):
            key = self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)

            # Skip if in cooldown
            if key in self.failed_keys and self.failed_keys[key] > now:
                continue

            # Skip if over rate limit
            if self.request_counts.get(key, 0) >= config.REQUESTS_PER_MINUTE_PER_KEY:
                continue

            return key

        # All keys exhausted - return least-used one
        available = [k for k in self.keys if k not in self.failed_keys or self.failed_keys[k] <= now]
        if available:
            return min(available, key=lambda k: self.request_counts.get(k, 0))

        logger.warning(f"{self.name}: All keys in cooldown, waiting...")
        return None

    def record_success(self, key: str):
        """Record successful request."""
        self.request_counts[key] = self.request_counts.get(key, 0) + 1

    def record_rate_limit(self, key: str, retry_after: int = 60):
        """Mark key as rate-limited."""
        self.failed_keys[key] = datetime.now() + timedelta(seconds=retry_after)
        logger.warning(f"{self.name}: Key {key[:8]}... rate limited for {retry_after}s")

    def record_failure(self, key: str, cooldown: int = 30):
        """Mark key as temporarily failed."""
        self.failed_keys[key] = datetime.now() + timedelta(seconds=cooldown)
        logger.warning(f"{self.name}: Key {key[:8]}... failed, cooldown {cooldown}s")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all keys."""
        now = datetime.now()
        return {
            "total_keys": len(self.keys),
            "available_keys": sum(
                1 for k in self.keys
                if k not in self.failed_keys or self.failed_keys[k] <= now
            ),
            "total_requests_this_minute": sum(self.request_counts.values()),
            "keys_in_cooldown": len([k for k, v in self.failed_keys.items() if v > now]),
        }


# =============================================================================
# Global Key Rotator (OpenRouter only)
# =============================================================================

openrouter_rotator = KeyRotator(config.OPENROUTER_API_KEYS, "OpenRouter")


def get_key_status() -> Dict[str, Any]:
    """Get status of API key rotator."""
    return {
        "openrouter": openrouter_rotator.get_status(),
    }


# =============================================================================
# Retry Logic
# =============================================================================

def retry_with_backoff(
    func,
    max_retries: int = None,
    base_delay: float = None,
    max_delay: float = None,
):
    """
    Decorator/wrapper for retry with exponential backoff.
    """
    max_retries = max_retries or config.MAX_RETRIES
    base_delay = base_delay or config.RETRY_BASE_DELAY
    max_delay = max_delay or config.RETRY_MAX_DELAY

    def wrapper(*args, **kwargs):
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if retryable
                retryable = any(x in error_str for x in [
                    "429", "rate limit", "500", "502", "503", "504",
                    "timeout", "connection", "temporarily"
                ])

                if not retryable or attempt >= max_retries:
                    raise

                # Calculate delay with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, 0.1 * delay)
                total_delay = delay + jitter

                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {total_delay:.1f}s: {e}"
                )
                time.sleep(total_delay)

        raise last_error

    return wrapper


# =============================================================================
# OpenRouter API Calls (with key rotation)
# =============================================================================

def call_openrouter_vision(
    image_url: str,
    prompt: str,
    max_tokens: int = None,
    temperature: float = None,
    timeout: int = None,
    request_id: str = None,
) -> str:
    """
    Call OpenRouter vision model with key rotation and retry logic.

    Args:
        image_url: Base64 data URL of the image
        prompt: Text prompt for the model
        max_tokens: Maximum tokens in response
        temperature: Model temperature
        timeout: Request timeout in seconds
        request_id: Optional request ID for logging

    Returns:
        Model response text
    """
    max_tokens = max_tokens or config.VISION_MAX_TOKENS
    temperature = temperature if temperature is not None else config.VISION_TEMPERATURE
    timeout = timeout or config.VISION_TIMEOUT
    log_prefix = f"[{request_id}]" if request_id else "[VISION]"

    logger.info(f"{log_prefix} Starting vision API call | model={config.VISION_MODEL} | max_tokens={max_tokens} | temp={temperature}")

    @retry_with_backoff
    def _make_request():
        api_key = openrouter_rotator.get_key()
        if not api_key:
            logger.error(f"{log_prefix} No OpenRouter API keys available")
            raise Exception("No OpenRouter API keys available")

        key_short = api_key[:12] + "..."
        logger.debug(f"{log_prefix} Using API key: {key_short}")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:5051',
            'X-Title': 'Document Assistant'
        }

        payload = {
            'model': config.VISION_MODEL,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_url}}
                    ]
                }
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
        }

        # Add thinking budget if configured
        if config.VISION_THINKING_BUDGET:
            payload['thinking'] = {'budget_tokens': config.VISION_THINKING_BUDGET}
            logger.debug(f"{log_prefix} Thinking budget: {config.VISION_THINKING_BUDGET}")

        try:
            logger.info(f"{log_prefix} Sending request to OpenRouter | timeout={timeout}s")
            start_time = time.time()

            response = requests.post(
                config.OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )

            elapsed = time.time() - start_time
            logger.info(f"{log_prefix} Response received | status={response.status_code} | time={elapsed:.2f}s")

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                openrouter_rotator.record_rate_limit(api_key, retry_after)
                logger.warning(f"{log_prefix} Rate limited (429) | retry_after={retry_after}s")
                raise Exception(f"Rate limit 429, retry after {retry_after}s")

            if response.status_code != 200:
                openrouter_rotator.record_failure(api_key)
                logger.error(f"{log_prefix} API error | status={response.status_code} | response={response.text[:500]}")
                raise Exception(f"API error: {response.status_code} - {response.text}")

            openrouter_rotator.record_success(api_key)
            result = response.json()

            # Log usage stats if available
            if 'usage' in result:
                usage = result['usage']
                logger.info(f"{log_prefix} Token usage | prompt={usage.get('prompt_tokens', 'N/A')} | completion={usage.get('completion_tokens', 'N/A')} | total={usage.get('total_tokens', 'N/A')}")

            content = result['choices'][0]['message']['content']
            logger.info(f"{log_prefix} Vision completed | response_length={len(content)} chars")
            logger.debug(f"{log_prefix} Vision response preview: {content[:200]}...")

            return content

        except requests.exceptions.Timeout:
            openrouter_rotator.record_failure(api_key, cooldown=10)
            logger.error(f"{log_prefix} Request timeout after {timeout}s")
            raise Exception("Request timeout")
        except requests.exceptions.ConnectionError as e:
            openrouter_rotator.record_failure(api_key, cooldown=10)
            logger.error(f"{log_prefix} Connection error: {e}")
            raise Exception(f"Connection error: {e}")

    return _make_request()


# =============================================================================
# Classifier Agent (uses OpenRouter with key rotation)
# =============================================================================

def create_classifier_agent(api_key: str = None):
    """Create a LangChain agent for document classification via OpenRouter."""
    key = api_key or openrouter_rotator.get_key()
    if not key:
        return None

    llm = ChatOpenAI(
        model=config.CLASSIFIER_MODEL,
        api_key=key,
        base_url=OPENROUTER_BASE_URL_V1,
        temperature=config.CLASSIFIER_TEMPERATURE,
        max_tokens=config.CLASSIFIER_MAX_TOKENS,
        default_headers={
            "HTTP-Referer": "http://localhost:5051",
            "X-Title": "Document Assistant"
        }
    )

    # Pass system prompt directly (can be str or SystemMessage)
    agent = create_react_agent(
        llm,
        EXTRACTION_TOOLS,
        prompt=SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT)
    )

    return agent


def classify_document(vision_description: str, request_id: str = None) -> dict:
    """
    Takes the vision model's description and classifies the document
    using OpenRouter API via LangGraph agent.

    Args:
        vision_description: Text description from the vision model
        request_id: Optional request ID for logging

    Returns:
        dict with document_type, extraction_schema, and classifier_summary
    """
    log_prefix = f"[{request_id}]" if request_id else "[CLASSIFIER]"
    logger.info(f"{log_prefix} Starting document classification | model={config.CLASSIFIER_MODEL}")
    logger.debug(f"{log_prefix} Input description length: {len(vision_description)} chars")

    @retry_with_backoff
    def _classify():
        api_key = openrouter_rotator.get_key()
        if not api_key:
            logger.error(f"{log_prefix} No OpenRouter API keys available")
            raise Exception("No OpenRouter API keys available")

        key_short = api_key[:12] + "..."
        logger.debug(f"{log_prefix} Using API key: {key_short}")

        logger.info(f"{log_prefix} Creating LangGraph ReAct agent")
        agent = create_classifier_agent(api_key)
        if agent is None:
            logger.error(f"{log_prefix} Failed to create classifier agent")
            raise Exception("Failed to create classifier agent")

        try:
            logger.info(f"{log_prefix} Invoking LangGraph agent")
            start_time = time.time()

            result = agent.invoke({
                "messages": [
                    HumanMessage(content=f"Analyze this document description and determine what type of document it is, then call the appropriate tool:\n\n{vision_description}")
                ]
            })

            elapsed = time.time() - start_time
            logger.info(f"{log_prefix} Agent completed | time={elapsed:.2f}s")

            openrouter_rotator.record_success(api_key)

            # Extract the tool results and final response
            tool_results = []
            final_response = ""
            tool_calls_made = []

            logger.debug(f"{log_prefix} Processing {len(result['messages'])} messages from agent")

            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_made.append(tc.get('name', 'unknown'))
                    logger.debug(f"{log_prefix} Tool calls: {tool_calls_made}")
                elif msg.type == "tool":
                    tool_results.append(json.loads(msg.content))
                    logger.debug(f"{log_prefix} Tool result received")
                elif msg.type == "ai" and msg.content:
                    final_response = msg.content

            extraction_schema = tool_results[0] if tool_results else None
            doc_type = extraction_schema.get("document_type", "unknown") if extraction_schema else "unknown"

            logger.info(f"{log_prefix} Classification result | document_type={doc_type} | tools_called={tool_calls_made}")

            return {
                "success": True,
                "document_type": doc_type,
                "extraction_schema": extraction_schema,
                "classifier_summary": final_response,
                "tools_called": tool_calls_made
            }

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                openrouter_rotator.record_rate_limit(api_key)
                logger.warning(f"{log_prefix} Rate limited during classification")
            else:
                openrouter_rotator.record_failure(api_key)
                logger.error(f"{log_prefix} Classification error: {e}")
            raise

    try:
        return _classify()
    except Exception as e:
        logger.error(f"{log_prefix} Classification failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_type": "unknown",
            "extraction_schema": None
        }


# =============================================================================
# Structured Extraction
# =============================================================================

def perform_structured_extraction(image_url: str, document_type: str = None, request_id: str = None) -> dict:
    """
    Performs enterprise-grade structured extraction on a document image.
    Uses the vision model with the enterprise extraction prompt.

    Args:
        image_url: Base64 data URL of the image
        document_type: Optional document type from classifier to provide context
        request_id: Optional request ID for logging

    Returns:
        dict with structured extraction results or error
    """
    log_prefix = f"[{request_id}]" if request_id else "[EXTRACTION]"
    logger.info(f"{log_prefix} Starting structured extraction | model={config.EXTRACTION_MODEL} | doc_type={document_type}")

    if not config.OPENROUTER_API_KEYS:
        logger.error(f"{log_prefix} No OpenRouter API keys configured")
        return {
            "success": False,
            "error": "No OpenRouter API keys configured",
            "structured_data": None
        }

    try:
        # Add document type context if available
        prompt = ENTERPRISE_EXTRACTION_PROMPT
        if document_type and document_type != "unknown":
            prompt += f"\n\nCONTEXT: This document has been pre-classified as a '{document_type}'. Use this as a hint but verify based on what you see."
            logger.debug(f"{log_prefix} Added document type context to prompt")

        logger.info(f"{log_prefix} Calling vision model for extraction")
        start_time = time.time()

        raw_response = call_openrouter_vision(
            image_url=image_url,
            prompt=prompt,
            max_tokens=config.EXTRACTION_MAX_TOKENS,
            temperature=config.EXTRACTION_TEMPERATURE,
            timeout=config.EXTRACTION_TIMEOUT,
            request_id=request_id,
        )

        elapsed = time.time() - start_time
        logger.info(f"{log_prefix} Extraction response received | time={elapsed:.2f}s | length={len(raw_response)} chars")

        # Try to parse JSON from response
        try:
            json_str = raw_response
            if '```json' in raw_response:
                json_str = raw_response.split('```json')[1].split('```')[0]
                logger.debug(f"{log_prefix} Extracted JSON from markdown code block")
            elif '```' in raw_response:
                json_str = raw_response.split('```')[1].split('```')[0]
                logger.debug(f"{log_prefix} Extracted content from code block")

            structured_data = json.loads(json_str.strip())
            logger.info(f"{log_prefix} Successfully parsed structured data | fields={len(structured_data.keys())}")

            # Log extracted key fields
            if isinstance(structured_data, dict):
                doc_type_extracted = structured_data.get('document_type', {})
                if isinstance(doc_type_extracted, dict):
                    logger.info(f"{log_prefix} Extracted document_type: {doc_type_extracted.get('value')}")
                issuer = structured_data.get('issuer_name', {})
                if isinstance(issuer, dict):
                    logger.info(f"{log_prefix} Extracted issuer: {issuer.get('value')}")
                gross = structured_data.get('gross_amount', {})
                if isinstance(gross, dict):
                    logger.info(f"{log_prefix} Extracted gross_amount: {gross.get('value')}")

            return {
                "success": True,
                "structured_data": structured_data,
                "raw_response": raw_response
            }
        except json.JSONDecodeError as e:
            logger.warning(f"{log_prefix} Failed to parse JSON: {e}")
            logger.debug(f"{log_prefix} Raw response: {raw_response[:500]}...")
            return {
                "success": True,
                "structured_data": None,
                "raw_response": raw_response,
                "parse_error": f"Could not parse JSON from response: {e}"
            }

    except Exception as e:
        logger.error(f"{log_prefix} Extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "structured_data": None
        }
