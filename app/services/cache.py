import hashlib
from cachetools import TTLCache

# Cache Q&A: TTL 1 giờ, tối đa 512 entries
# Key bao gồm uid để tránh cross-user cache leak
_qa_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)


def _make_key(uid: str, question: str, source_key: str) -> str:
    raw = f"{uid}::{source_key}::{question.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(uid: str, question: str, source_key: str) -> dict | None:
    return _qa_cache.get(_make_key(uid, question, source_key))


def set_cached(uid: str, question: str, source_key: str, result: dict) -> None:
    _qa_cache[_make_key(uid, question, source_key)] = result
