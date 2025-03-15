# oblix/cli/commands/models.py
import click
import sys
import os
import colorama
from colorama import Fore, Style

# Change relative import to absolute import
from oblix.cli.utils import setup_client, handle_async_command

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

def print_table(title, headers, rows):
    """Print a simple table"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print headers
    header_row = " | ".join(f"{h:{w}s}" for h, w in zip(headers, col_widths))
    print(f"{Fore.CYAN}{header_row}{Style.RESET_ALL}")
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(f"{str(cell):{w}s}" for cell, w in zip(row, col_widths))
        print(row_str)

def print_panel(title, content):
    """Print a simple panel with a title and content"""
    width = max(len(title) + 4, max(len(line) for line in content.split('\n')) + 4)
    
    print(f"\n{Fore.BLUE}{Style.BRIGHT}┌─{title}{'─' * (width - len(title) - 3)}┐{Style.RESET_ALL}")
    for line in content.split('\n'):
        print(f"{Fore.BLUE}{Style.BRIGHT}│{Style.RESET_ALL} {line}{' ' * (width - len(line) - 2)} {Fore.BLUE}{Style.BRIGHT}│{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}└{'─' * (width - 2)}┘{Style.RESET_ALL}")

@click.group(name='models')
def models_group():
    """Manage and interact with AI models"""
    pass

@models_group.command('list')
@click.option('--api-key', help='Oblix API key (optional)', envvar='OBLIX_API_KEY')
def list_models(api_key):
    """List all available models"""
    async def run_list_models():
        try:
            client = await setup_client(api_key)
            models = client.list_models()
            
            if not models:
                print_warning("No models configured.")
                return
            
            rows = []
            for model_type, model_names in models.items():
                rows.append([model_type, ", ".join(model_names)])
            
            print_table("Configured Models", ["Model Type", "Model Names"], rows)
        except Exception as e:
            print_error(f"Error listing models: {e}")
    
    handle_async_command(run_list_models)

@models_group.command('hook')
@click.option('--api-key', help='Oblix API key (optional)', envvar='OBLIX_API_KEY')
@click.option('--type', required=True, help='Model type (ollama, openai, claude)')
@click.option('--name', required=True, help='Model name')
@click.option('--endpoint', help='Model endpoint (for Ollama)')
@click.option('--api-key-model', help='Model-specific API key')
def hook_model(api_key, type, name, endpoint, api_key_model):
    """Hook a new AI model to the Oblix client"""
    async def run_hook_model():
        try:
            from oblix.models.base import ModelType
            
            # Map string to ModelType enum
            model_type_map = {
                'ollama': ModelType.OLLAMA,
                'openai': ModelType.OPENAI,
                'claude': ModelType.CLAUDE
            }
            
            # Validate model type
            if type.lower() not in model_type_map:
                print_error(f"Invalid model type: {type}")
                print("Supported types: ollama, openai, claude")
                return
            
            # Prepare hook parameters
            hook_params = {
                'model_type': model_type_map[type.lower()],
                'model_name': name
            }
            
            # Add optional parameters
            if endpoint:
                hook_params['endpoint'] = endpoint
            if api_key_model:
                hook_params['api_key'] = api_key_model
            
            # Setup client and hook model
            client = await setup_client(api_key)
            success = await client.hook_model(**hook_params)
            
            if success:
                print_success(f"Model {name} successfully hooked!")
            else:
                print_error(f"Failed to hook model {name}")
        except Exception as e:
            print_error(f"Model hook error: {e}")
    
    handle_async_command(run_hook_model)

@models_group.command('info')
@click.option('--api-key', help='Oblix API key (optional)', envvar='OBLIX_API_KEY')
@click.option('--type', required=True, help='Model type')
@click.option('--name', required=True, help='Model name')
def model_info(api_key, type, name):
    """Get information about a specific model"""
    async def run_model_info():
        try:
            client = await setup_client(api_key)
            model_config = client.get_model(type, name)
            
            if model_config:
                # Create content for panel
                api_key_status = f"{Fore.GREEN}Configured{Style.RESET_ALL}" if model_config.get('api_key') else f"{Fore.YELLOW}Not Set{Style.RESET_ALL}"
                content = (
                    f"Type: {model_config['type']}\n"
                    f"Name: {model_config['name']}\n"
                    f"Endpoint: {model_config.get('endpoint', 'N/A')}\n"
                    f"API Key: {api_key_status}"
                )
                print_panel("Model Configuration", content)
            else:
                print_warning(f"No configuration found for model {name} of type {type}")
        except Exception as e:
            print_error(f"Error retrieving model info: {e}")
    
    handle_async_command(run_model_info)