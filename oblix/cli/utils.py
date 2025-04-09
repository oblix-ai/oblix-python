# oblix/cli/utils.py
import asyncio
import sys
import logging
import colorama
from colorama import Fore, Style
from typing import Callable, Awaitable, Any

from oblix.client import OblixClient

# Initialize colorama
colorama.init()

def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def handle_async_command(async_func: Callable[[], Awaitable[Any]]) -> Any:
    """
    Wrapper to handle async CLI commands
    
    Args:
        async_func: Async function to execute
        
    Returns:
        Any: The return value from the async function
    """
    loop = None
    try:
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the async function
        return loop.run_until_complete(async_func())
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        # The inner function should handle specific exceptions
        # This is just a generic handler for any unhandled exceptions
        print_error(f"Error: {e}")
        
        # Only show traceback in debug mode
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            traceback.print_exc()
            
        return None
    finally:
        # Clean up pending tasks and close the event loop properly
        if loop is not None:
            try:
                # Cancel all pending tasks
                tasks = [task for task in asyncio.all_tasks(loop) 
                        if not task.done() and task is not asyncio.current_task()]
                if tasks:
                    for task in tasks:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                
                # Close the loop
                loop.close()
            except Exception:
                # Ignore cleanup errors
                pass

async def setup_client() -> OblixClient:
    """
    Set up an Oblix client
    
    Returns:
        Initialized OblixClient instance
    """
    try:
        # Create client instance
        client = OblixClient()
        
        return client
    
    except Exception as e:
        print_error(f"Client setup error: {e}")
        sys.exit(1)