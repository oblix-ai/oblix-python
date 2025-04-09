# oblix/cli/commands/init.py
import click
import os
import sys
import colorama
from colorama import Fore, Style
import asyncio

from oblix.cli.utils import handle_async_command

# Initialize colorama
colorama.init()

def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")

def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_info(text):
    """Print information with cyan color"""
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")

@click.command('setup')
def init_command():
    """Set up Oblix configuration directories"""
    async def run_init():
        try:
            # Import here to avoid circular imports
            from oblix.client import OblixClient
            from oblix.config import ConfigManager
            
            # Create the configuration manager and ensure config directory exists
            config_manager = ConfigManager()
            
            print_success("\nOblix configuration directories have been set up.")
            
            # Create a client
            client = OblixClient()
            
            print_info("\nNext steps:")
            print_info("1. Hook a local model: oblix models hook --type ollama --name llama2")
            print_info("2. Hook a cloud model (option 1): oblix models hook --type openai --name gpt-3.5-turbo --api-key YOUR_OPENAI_API_KEY")
            print_info("   OR (option 2): oblix models hook --type claude --name claude-3-7-sonnet-20250219 --api-key YOUR_ANTHROPIC_API_KEY")
            print_info("3. Start the server: oblix server --local-model ollama:llama2 --cloud-model claude:claude-3-7-sonnet-20250219")
            
        except Exception as e:
            print_error(f"Initialization error: {e}")
            sys.exit(1)
    
    handle_async_command(run_init)