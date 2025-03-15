# oblix/cli/utils.py
import asyncio
import sys
import colorama
from colorama import Fore, Style
from typing import Callable, Awaitable

from oblix.client import OblixClient

# Initialize colorama
colorama.init()

def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def handle_async_command(async_func: Callable[[], Awaitable[None]]):
    """
    Wrapper to handle async CLI commands
    
    Args:
        async_func: Async function to execute
    """
    try:
        asyncio.run(async_func())
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

async def setup_client(api_key: str = None) -> OblixClient:
    """
    Set up an Oblix client with optional API key
    
    Args:
        api_key: Optional API key for client initialization
    
    Returns:
        Initialized OblixClient instance
    """
    try:
        # Create client instance
        client = OblixClient(oblix_api_key=api_key)
        
        # Ensure authentication
        await client._ensure_authenticated()
        
        return client
    
    except Exception as e:
        print_error(f"Client setup error: {e}")
        sys.exit(1)