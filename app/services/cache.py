import hashlib
from cachetools import TTLCache

# Cache Q&A: key = hash(question + doc_id), TTL 1 giờ, tối đa 512 entries
_qa_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)


def _make_key(question: str, doc_id: str) -> str:
    raw = f"{doc_id}::{question.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(question: str, doc_id: str) -> dict | None:
    return _qa_cache.get(_make_key(question, doc_id))


def set_cached(question: str, doc_id: str, result: dict) -> None:
    _qa_cache[_make_key(question, doc_id)] = result
