from typing import TypeVar, Callable, Any
import asyncio
import functools
import logging

T = TypeVar("T")

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (429,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries the wrapped function with exponential backoff.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        raise e
                    
                    # Check if we should retry based on error
                    should_retry = False
                    error_str = str(e)
                    for status_code in retry_on:
                        if str(status_code) in error_str:
                            should_retry = True
                            break
                    
                    if not should_retry:
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** (retries - 1)), max_delay)
                    logging.warning(f"Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator