# oblix/cli/commands/models.py
import click
import colorama
from colorama import Fore, Style
import httpx
import asyncio
import sys
import json

# Initialize colorama
colorama.init()

def print_header(text):
    """Print a header with formatting"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{text}{Style.RESET_ALL}")

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

def print_model_item(name, description=""):
    """Print a model item with bullet point"""
    if description:
        print(f"  • {Style.BRIGHT}{name}{Style.RESET_ALL} - {description}")
    else:
        print(f"  • {Style.BRIGHT}{name}{Style.RESET_ALL}")

async def get_ollama_models():
    """Get list of locally available Ollama models"""
    try:
        # Try connecting to Ollama's API (default is http://localhost:11434)
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model.get("name") for model in models]
            else:
                return None
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
        # Connection failed
        return None
    except Exception as e:
        print_error(f"Error connecting to Ollama: {e}")
        return None

async def get_openai_models():
    """Return a list of recommended OpenAI models"""
    # Import from the centralized model definitions
    from oblix.models.supported_models import SUPPORTED_OPENAI_MODELS
    return SUPPORTED_OPENAI_MODELS

async def get_claude_models():
    """Return a list of recommended Claude models"""
    # Import from the centralized model definitions
    from oblix.models.supported_models import SUPPORTED_CLAUDE_MODELS
    return SUPPORTED_CLAUDE_MODELS

@click.command(name='models')
@click.option('--refresh', is_flag=True, help='Force refresh of model list')
def models_group(refresh):
    """Show available AI models for use with Oblix"""
    # We need to use asyncio.run since this is a synchronous Click command
    # but we need to make async API calls
    asyncio.run(async_models_group(refresh))

async def async_models_group(refresh):
    print_header("Supported Model Providers")
    print_info("Oblix supports the following AI model providers and models:")
    
    # Show Ollama models
    print_header("Ollama")
    print_info("Local models served through Ollama")
    
    # Try to get list of installed Ollama models
    ollama_models = await get_ollama_models()
    
    if ollama_models:
        if ollama_models:
            print_success("\nInstalled Ollama models:")
            for model in ollama_models:
                print_model_item(model)
        else:
            print_warning("\nNo Ollama models found.")
            print_info("Run 'ollama pull <model_name>' to download models.")
    else:
        print_warning("\nCould not connect to Ollama server.")
        print_info("To use local models, please:")
        print_info("1. Install Ollama from https://ollama.com")
        print_info("2. Start the Ollama service")
        print_info("3. Run 'ollama pull <model_name>' to download models")
    
    # Show OpenAI models
    print_header("OpenAI")
    print_info("GPT models via OpenAI API")
    openai_models = await get_openai_models()
    print_success("\nRecommended OpenAI models:")
    for model in openai_models:
        description = f"{model['description']} - {model.get('features', '')}"
        context = model.get('context_window', '')
        if context:
            description += f" (Context: {context})"
        print_model_item(model["name"], description)
    
    # Show Claude models
    print_header("Claude")
    print_info("Claude models via Anthropic API")
    claude_models = await get_claude_models()
    print_success("\nRecommended Claude models:")
    for model in claude_models:
        description = f"{model['description']} - {model.get('features', '')}"
        context = model.get('context_window', '')
        if context:
            description += f" (Context: {context})"
        print_model_item(model["name"], description)
    
    print() # Add a blank line at the end