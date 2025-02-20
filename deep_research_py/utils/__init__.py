from rich.console import Console
from typing import Optional

console = Console()
_service: Optional[str] = None
_model: Optional[str] = None

def get_service() -> str:
    return _service or "openai"

def set_service(service: str) -> None:
    global _service
    _service = service

def get_model() -> str:
    return _model or "gpt-3.5-turbo"

def set_model(model: str) -> None:
    global _model
    _model = model

from .retry import retry_with_exponential_backoff

__all__ = [
    'retry_with_exponential_backoff',
    'console',
    'get_service',
    'set_service',
    'get_model',
    'set_model'
]