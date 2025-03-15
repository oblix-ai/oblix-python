#!/usr/bin/env python
# oblix/cli/main.py
import sys
import asyncio
import click
import logging
import colorama
from colorama import Fore, Style

# Change relative imports to absolute imports
from oblix.cli.commands.models import models_group
from oblix.cli.commands.agents import agents_group
from oblix.cli.commands.sessions import sessions_group
from oblix.cli.utils import handle_async_command, setup_client
from oblix.client.client import OblixClient
from oblix.models.base import ModelType

# Initialize colorama
colorama.init()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    datefmt="[%X]"
)
logger = logging.getLogger("oblix.cli")

def print_success(text):
    """
    Print success message with green text.
    
    Args:
        text: Text to print
    """
    print(f"{Fore.GREEN}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_warning(text):
    """
    Print warning message with yellow text.
    
    Args:
        text: Text to print
    """
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

def print_error(text):
    """
    Print error message with red text.
    
    Args:
        text: Text to print
    """
    print(f"{Fore.RED}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_info(text):
    """
    Print information with cyan color.
    
    Args:
        text: Text to print
    """
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")

@click.group()
@click.version_option(package_name="oblix")
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--check-updates', is_flag=True, help='Check for new versions of Oblix')
def cli(debug, check_updates):
    """
    Oblix AI SDK Command Line Interface

    Manage AI models, agents, and interactive sessions with ease.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Always check for updates, but only force display if --check-updates is used
    from oblix import check_for_updates
    check_for_updates(print_notification=check_updates)

# Add subcommands
cli.add_command(models_group)
cli.add_command(agents_group)
cli.add_command(sessions_group)

@cli.command()
def check_updates():
    """
    Check for new versions of Oblix.
    
    Queries PyPI to check if a newer version of Oblix is available
    and displays upgrade instructions if needed.
    """
    from oblix import check_for_updates, version_info
    
    current_info = version_info()
    update_info = check_for_updates(print_notification=True)
    
    if not update_info.get('update_available'):
        print_success(f"You are using the latest version of Oblix ({current_info['version']}).")

@cli.command(help="""
Start an interactive chat session with a local and cloud model.

IMPORTANT: 
- You must specify a local Ollama model (e.g., tinyllama, llama2)
- You must specify a cloud model (e.g., gpt-3.5-turbo from OpenAI)
- A monitoring agent will be automatically attached for system observability

EXAMPLES:
  oblix chat --local-model tinyllama --cloud-model gpt-3.5-turbo
  oblix chat --local-model llama2 --cloud-model claude-3-haiku
""")
@click.option('--api-key', help='Oblix API key (optional)', envvar='OBLIX_API_KEY')
@click.option('--local-model', required=True, help='Local Ollama model (e.g., tinyllama, llama2)')
@click.option('--cloud-model', required=True, help='Cloud model (e.g., gpt-3.5-turbo from OpenAI or claude-3-haiku from Anthropic)')
@click.option('--cloud-api-key', help='API key for cloud model', envvar='OPENAI_API_KEY')
def chat(api_key, local_model, cloud_model, cloud_api_key):
    """
    Start an interactive chat session with a local and cloud model.
    
    Sets up a hybrid configuration with both local (Ollama) and cloud (OpenAI/Claude)
    models, automatically attaching monitoring agents. The system will intelligently
    route prompts between models based on resource availability and network connectivity.
    
    Args:
        api_key: Oblix API key
        local_model: Local Ollama model name
        cloud_model: Cloud model name
        cloud_api_key: API key for cloud model
    """
    async def run_chat(passed_cloud_api_key):
        try:
            # Create client
            client = OblixClient(oblix_api_key=api_key)
            
            # Ensure authentication
            await client._ensure_authenticated()
            
            # Hook local Ollama model
            local_hook_success = await client.hook_model(
                model_type=ModelType.OLLAMA, 
                model_name=local_model
            )
            if not local_hook_success:
                print_error(f"Failed to hook local Ollama model: {local_model}")
                return
            
            # Determine cloud model type and hook
            cloud_model_parts = cloud_model.lower().split('-', 1)
            cloud_provider = cloud_model_parts[0]
            
            # Map cloud providers to ModelType
            cloud_model_map = {
                'gpt': ModelType.OPENAI,
                'claude': ModelType.CLAUDE
            }
            
            if cloud_provider not in cloud_model_map:
                print_error(f"Unsupported cloud model provider: {cloud_provider}")
                print("Supported providers: gpt (OpenAI), claude (Anthropic)")
                return
            
            # Use passed API key or prompt
            local_cloud_api_key = passed_cloud_api_key
            if not local_cloud_api_key:
                local_cloud_api_key = click.prompt(
                    f"Enter API key for {cloud_provider.upper()} model", 
                    hide_input=True
                )
            
            # Hook cloud model
            cloud_hook_success = await client.hook_model(
                model_type=cloud_model_map[cloud_provider], 
                model_name=cloud_model,
                api_key=local_cloud_api_key
            )
            
            if not cloud_hook_success:
                print_error(f"Failed to hook cloud model: {cloud_model}")
                return
            
            # Automatically hook monitoring agents
            from oblix.agents.resource_monitor import ResourceMonitor
            from oblix.agents.connectivity import ConnectivityAgent
            
            # Hook Resource Monitor
            resource_monitor = ResourceMonitor()
            client.hook_agent(resource_monitor)
            
            # Hook Connectivity Agent
            connectivity_agent = ConnectivityAgent()
            client.hook_agent(connectivity_agent)
            
            # Start chat session
            await client.start_chat()
        
        except Exception as e:
            print_error(f"Chat setup error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure all async resources are properly closed
            await asyncio.sleep(0.1)
    
    # Resolve cloud API key before passing to async function
    import os  # Add this import
    resolved_cloud_api_key = cloud_api_key or os.getenv('OPENAI_API_KEY')
    
    # Use handle_async_command to run the async function
    handle_async_command(lambda: run_chat(resolved_cloud_api_key))


@cli.command()
@click.argument('prompt')
@click.option('--api-key', help='Oblix API key (optional)', envvar='OBLIX_API_KEY')
@click.option('--model', default=None, help='Specify a model to use')
def execute(prompt, api_key, model):
    """
    Execute a single prompt.
    
    Processes a prompt with the specified model (or auto-selects an appropriate model)
    and displays the result.
    
    Args:
        prompt: Text prompt to process
        api_key: Oblix API key
        model: Optional specific model to use
    """
    async def run_execute():
        try:
            client = await setup_client(api_key)
            
            # Prepare execution parameters
            kwargs = {}
            if model:
                # TODO: Add logic to parse model type and name if needed
                kwargs['model_id'] = model
            
            result = await client.execute(prompt, **kwargs)
            
            if result:
                if 'error' in result:
                    print_error(f"Error: {result['error']}")
                else:
                    print_success("\nResponse:")
                    print(result['response'])
                    
                    # Optionally print metrics if available
                    if result.get('metrics'):
                        print_info("\nMetrics:")
                        for k, v in result['metrics'].items():
                            print(f"{k}: {v}")
            else:
                print_warning("No response generated.")
        except Exception as e:
            print_error(f"Execution error: {e}")
    
    handle_async_command(run_execute)

if __name__ == '__main__':
    cli()