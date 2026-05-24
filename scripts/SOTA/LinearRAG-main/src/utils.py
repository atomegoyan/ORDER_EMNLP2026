from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os
import time

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


class LLM_Model:
    """LLM wrapper backed by the Cohere Chat API.

    Args:
        llm_model: Cohere model name, e.g. ``"command-r-plus"``.
        api_key:   Cohere API key. Falls back to the ``COHERE_API_KEY``
                   environment variable when not provided.
    """

    def __init__(self, llm_model: str = "command-r-plus", api_key: str | None = None):
        import cohere  # lazy import

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError(
                "Cohere API key must be provided via the api_key argument "
                "or the COHERE_API_KEY environment variable."
            )
        self.client = cohere.ClientV2(api_key=key)
        self.llm_model = llm_model

    def infer(self, messages: list[dict]) -> str:
        """Call the Cohere Chat API with retry logic.

        Args:
            messages: List of ``{role, content}`` dicts (OpenAI-compatible).

        Returns:
            The model's text response.
        """
        for attempt in range(5):
            try:
                response = self.client.chat(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0,
                )
                return response.message.content[0].text
            except Exception as exc:
                wait = 2 ** attempt
                logging.getLogger(__name__).warning(
                    "Cohere chat error (attempt %d/5): %s — retrying in %ds",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError("Cohere chat failed after 5 retries.")



def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s) 
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val
