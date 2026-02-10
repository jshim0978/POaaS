"""
Async utilities for baseline implementations.

Handles running async code from synchronous contexts, including
when already inside an event loop.
"""

import asyncio


def run_async(coro):
    """
    Run async coroutine from sync context.
    
    Handles the case when we're already in a running event loop
    (e.g., when called from an async evaluation framework).
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)
    
    # We're in an async context
    # Try to use nest_asyncio if available
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(coro)
    except ImportError:
        pass
    
    # Fallback: run in a new thread with its own event loop
    import concurrent.futures
    
    def run_in_thread():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        return future.result(timeout=300)  # 5 minute timeout

