import json
import os
import threading

from app.config import REWRITE_CACHE_ENABLED


REWRITE_CACHE_PATH = "data/cache/rewrite_cache.json"
_cache_lock = threading.Lock()


def _normalize_question(question):
    return " ".join(
        question.strip().lower().split()
    )


def _load_cache():
    if not os.path.exists(REWRITE_CACHE_PATH):
        return {}

    try:
        with open(REWRITE_CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache):
    os.makedirs(
        os.path.dirname(REWRITE_CACHE_PATH),
        exist_ok=True
    )

    with open(REWRITE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def get_cached_rewrite(question):
    if not REWRITE_CACHE_ENABLED:
        return None

    key = _normalize_question(question)

    with _cache_lock:
        cache = _load_cache()

    return cache.get(key)


def save_cached_rewrite(question, rewritten_question):
    if not REWRITE_CACHE_ENABLED:
        return

    key = _normalize_question(question)

    with _cache_lock:
        cache = _load_cache()
        cache[key] = rewritten_question
        _save_cache(cache)
