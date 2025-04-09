#!/usr/bin/env python
# oblix/cli/main.py
import sys
import asyncio
import click
import logging
import colorama
from colorama import Fore, Style
import time
import os

# Change relative imports to absolute imports
from oblix.cli.commands.models import models_group as models_command
from oblix.cli.commands.agents import agents_group as agents_command
from oblix.cli.commands.sessions import sessions_group
from oblix.cli.utils import handle_async_command, setup_client
from oblix.client.client import OblixClient
from oblix.models.base import ModelType

# Initialize colorama
colorama.init()

# Configure basic logging with WARNING level
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    datefmt="[%X]"
)
# Create a deduplication filter class to avoid repeating model config messages
class DuplicateFilter(logging.Filter):
    def __init__(self, name=""):
        super(DuplicateFilter, self).__init__(name)
        self.last_log = {}
        
    def filter(self, record):
        # Skip filtering for levels higher than INFO
        if record.levelno > logging.INFO:
            return True
            
        # Get log message
        current_log = record.getMessage()
        
        # Check if this message contains model configuration text
        if "Added model configuration:" in current_log or "model configuration:" in current_log:
            model_name = current_log.split(":")[-1].strip()
            current_time = time.time()
            
            # If we've seen this model recently, skip it
            if model_name in self.last_log and current_time - self.last_log[model_name] < 5:
                return False
                
            self.last_log[model_name] = current_time
        
        return True

# Set specific loggers to higher level to avoid duplicate messages
logging.getLogger("oblix.client").setLevel(logging.WARNING)
logging.getLogger("oblix.config").setLevel(logging.WARNING)
logging.getLogger("oblix.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress HTTP request logs
logging.getLogger("oblix.agents").setLevel(logging.WARNING)  # Suppress agent initialization logs

# Create a special filter for execution logs
class ExecutionFilter(logging.Filter):
    def filter(self, record):
        # Filter out specific routing decision logs from execution module
        if record.name == 'oblix.core.execution':
            msg = record.getMessage()
            if "Selecting cloud model" in msg or \
               "Selecting local model" in msg or \
               "Selecting default model type" in msg:
                return False  # Filter out these specific messages
        return True

# Add filters
root_logger = logging.getLogger()
duplicate_filter = DuplicateFilter()
execution_filter = ExecutionFilter()
root_logger.addFilter(duplicate_filter)
root_logger.addFilter(execution_filter)

# Set execution logger to WARNING level for extra suppression
logging.getLogger("oblix.core.execution").setLevel(logging.WARNING)

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

class CustomHelpGroup(click.Group):
    def format_commands(self, ctx, formatter):
        """Custom command formatter that doesn't show function names."""
        rows = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None:
                continue
            help = cmd.short_help or cmd.help or ''
            
            # Remove function name and signature from help text
            # This handles both func() and func(arg1, arg2) patterns
            if '(' in help and ')' in help:
                # Split on the closing parenthesis
                parts = help.split(')', 1)
                if len(parts) > 1:
                    # Take only the part after the closing parenthesis
                    help = parts[1].strip()
                else:
                    # Remove any parentheses content as a fallback
                    help = help.replace('()', '')
            
            # Fix the specific "smart decisions" line-wrap issue
            if subcommand == "agents" and "make smart" in help and "decisions" in help:
                help = "Show the monitoring agents that help Oblix make smart decisions"
            
            rows.append((subcommand, help))
        
        if rows:
            # Save the original formatter width
            original_width = formatter.width
            
            with formatter.section('Commands'):
                try:
                    # Try to set a wider formatter width to avoid line wrapping
                    formatter.width = 100
                    formatter.write_dl(rows)
                finally:
                    # Restore the original width
                    formatter.width = original_width

class CustomCliGroup(CustomHelpGroup):
    def get_help(self, ctx):
        """Customize the help text to make it more compact"""
        formatter = ctx.make_formatter()
        
        # Custom usage lines, one below the other
        formatter.write("Usage: oblix [OPTIONS]\n")
        formatter.write("       oblix [COMMANDS]\n")
        
        # Add options section to show --help and --version
        with formatter.section('Options'):
            formatter.write_dl([
                ("--help", "Show this message and exit."),
                ("--version", "Show the version and exit.")
            ])
        
        # Skip the description (which we don't want to show)
        # and just format the commands section
        self.format_commands(ctx, formatter)
        
        # Add a minimal footer
        formatter.write("\nRun 'oblix COMMAND --help' for more information on a command.\n")
        
        return formatter.getvalue().rstrip('\n')
        
    def format_command(self, ctx, command, *args, **kwargs):
        """Override to clean up the command help text for specific commands"""
        # First get the original help text
        result = super().format_command(ctx, command, *args, **kwargs)
        
        # Check if the help text contains a function signature
        if '(' in result and ')' in result:
            # Extract the command name
            cmd_name = command.name if hasattr(command, 'name') else ''
            
            # Find the first occurrence of the command name
            idx = result.find(cmd_name)
            
            if idx >= 0:
                # Find the opening parenthesis after the command name
                paren_idx = result.find('(', idx)
                
                if paren_idx >= 0:
                    # Find the closing parenthesis
                    close_paren_idx = result.find(')', paren_idx)
                    
                    if close_paren_idx >= 0:
                        # Replace the function signature with just the command name
                        result = result[:idx + len(cmd_name)] + result[close_paren_idx + 1:]
        
        return result

@click.group(cls=CustomCliGroup)
@click.version_option(package_name="oblix")
def cli():
    """Oblix AI SDK Command Line Interface"""
    pass

# Import commands
from oblix.cli.commands.server import server_group

# Add subcommands
cli.add_command(models_command)  # Our simplified models command
cli.add_command(agents_command)
cli.add_command(sessions_group)
# cli.add_command(server_group)  # We're using our direct server command instead

@cli.command(name='server')
@click.option('--port', default=62549, help='Port to run the server on')
@click.option('--host', default='0.0.0.0', help='Host to bind the server to')
@click.option('--local-model', required=True, help='Local model (e.g., ollama:llama2)')
@click.option('--cloud-model', required=True, help='Cloud model (e.g., openai:gpt-3.5-turbo or claude:claude-3-7-sonnet-20250219)')
@click.option('--cloud-api-key', help='API key for cloud model (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
def start_server(port, host, local_model, cloud_model, cloud_api_key):
    """Start the Oblix API server to use with any client application"""
    # Parse the local and cloud model parameters
    local_model_parts = local_model.split(':')
    if len(local_model_parts) != 2:
        print_error(f"Invalid local model format: {local_model}. Should be 'provider:model_name'")
        return
    local_provider, local_model_name = local_model_parts
    
    cloud_model_parts = cloud_model.split(':')
    if len(cloud_model_parts) != 2:
        print_error(f"Invalid cloud model format: {cloud_model}. Should be 'provider:model_name'")
        return
    cloud_provider, cloud_model_name = cloud_model_parts
    
    # Get the appropriate API key based on the cloud provider
    if cloud_provider.lower() == 'claude':
        # For Claude models, check ANTHROPIC_API_KEY environment variable
        cloud_api_key = cloud_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not cloud_api_key:
            print_error("No API key provided for Claude model. Please provide --cloud-api-key or set ANTHROPIC_API_KEY environment variable.")
            return
    elif cloud_provider.lower() == 'openai':
        # For OpenAI models, check OPENAI_API_KEY environment variable
        cloud_api_key = cloud_api_key or os.getenv('OPENAI_API_KEY')
        if not cloud_api_key:
            print_error("No API key provided for OpenAI model. Please provide --cloud-api-key or set OPENAI_API_KEY environment variable.")
            return
            
    # Pass appropriate parameters to the server start command
    from oblix.cli.commands.server import start_server as server_start_fn
    
    # This function is now replaced by setup_and_run_server
    # Which properly handles the event loop issues
    
    # Complete function that handles both setup and server running in one async context
    async def setup_and_run_server():
        try:
            # Create client without API key
            from oblix.client import OblixClient
            client = OblixClient()
            
            # Hook local model
            from oblix.models.base import ModelType
            local_type = ModelType.OLLAMA if local_provider.lower() == 'ollama' else ModelType.CUSTOM
            local_hook_success = await client.hook_model(
                model_type=local_type, 
                model_name=local_model_name
            )
            if not local_hook_success:
                print_error(f"Failed to hook local model: {local_model}")
                return
                
            # Hook cloud model
            cloud_type = None
            if cloud_provider.lower() == 'openai':
                cloud_type = ModelType.OPENAI
            elif cloud_provider.lower() == 'claude':
                cloud_type = ModelType.CLAUDE
            else:
                print_error(f"Unsupported cloud provider: {cloud_provider}")
                print_info("Supported cloud providers: openai, claude")
                return
            
            cloud_hook_success = await client.hook_model(
                model_type=cloud_type, 
                model_name=cloud_model_name,
                api_key=cloud_api_key
            )
            if not cloud_hook_success:
                print_error(f"Failed to hook cloud model: {cloud_model}")
                return
            
            # Add monitoring agents
            from oblix.agents.resource_monitor import ResourceMonitor
            from oblix.agents.connectivity import ConnectivityAgent
            
            print_info("Initializing monitoring agents...")
            resource_monitor = ResourceMonitor(name="resource_monitor_agent")
            connectivity_agent = ConnectivityAgent()
            
            client.hook_agent(resource_monitor)
            client.hook_agent(connectivity_agent)
            
            # Store client in global API Manager
            from oblix.api.routes import OblixAPIManager
            OblixAPIManager._instance = client
            
            # Import the app
            from oblix.main import app
            import uvicorn
            
            # ASCII art logo with border
            print(f"{Fore.MAGENTA}{Style.BRIGHT}")
            print("""
╔═════════════════════════════════════════════╗
║                                             ║
║   ██████  ██████  ██      ██ ██   ██       ║
║  ██    ██ ██   ██ ██      ██  ██ ██        ║
║  ██    ██ ██████  ██      ██   ███         ║
║  ██    ██ ██   ██ ██      ██  ██ ██        ║
║   ██████  ██████  ███████ ██ ██   ██       ║
║                                             ║
║        AI Orchestration Framework           ║
║                                             ║
╚═════════════════════════════════════════════╝
            """)
            print(f"{Style.RESET_ALL}")
            
            # Print a custom message that shows localhost in the URL
            print_success(f"Server will be available at http://localhost:{port}")
            print_success(f"OpenAI-compatible endpoint: http://localhost:{port}/v1/chat/completions")
            
            # Print model usage information
            print_info("\nModel Usage:")
            print_info("Oblix uses intelligent orchestration to automatically select the best model")
            print_info("based on connectivity, system resources, and the specific request.")
            print_info("Always use model=\"auto\" to enable full orchestration capabilities.")
            print_info("\nExample API usage:")
            print_info("""
# Using any standard HTTP client
POST http://localhost:""" + str(port) + """/v1/chat/completions
Content-Type: application/json
Authorization: Bearer any-value

{
  "model": "auto",  # Always use "auto" for Oblix's intelligent orchestration
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are the three laws of robotics?"}
  ]
}

# Using Python requests
import requests

response = requests.post(
    "http://localhost:""" + str(port) + """/v1/chat/completions",
    headers={"Content-Type": "application/json", "Authorization": "Bearer any-value"},
    json={
        "model": "auto",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the three laws of robotics?"}
        ]
    }
)
print(response.json()["choices"][0]["message"]["content"])
""")
            
            # Configure logging - quiet but show access logs
            import logging
            from uvicorn.config import LOGGING_CONFIG
            
            # Create a custom logging config that shows access but hides other info
            custom_logging_config = LOGGING_CONFIG.copy()
            # Silence uvicorn startup/shutdown logs but keep access logs
            custom_logging_config["loggers"]["uvicorn"] = {"handlers": [], "level": "CRITICAL", "propagate": False}
            custom_logging_config["loggers"]["uvicorn.error"] = {"handlers": [], "level": "CRITICAL", "propagate": False}
            # Keep access logs at INFO level
            custom_logging_config["loggers"]["uvicorn.access"] = {"handlers": ["access"], "level": "INFO", "propagate": False}
            
            # Use uvicorn.Config and Server with selective logging
            config = uvicorn.Config(app, host=host, port=port, log_config=custom_logging_config, log_level="info")
            server = uvicorn.Server(config)
            
            # No custom signal handling - use uvicorn's default signal handling
            await server.serve()
            
            return client
            
        except Exception as e:
            print_error(f"Server startup error: {e}")
            if logging.getLogger().level == logging.DEBUG:
                import traceback
                traceback.print_exc()
    
    # Run the setup and server in ONE async context with proper cleanup
    async def main():
        client = None
        try:
            client = await setup_and_run_server()
            return client
        except KeyboardInterrupt:
            # Silently exit on keyboard interrupt
            pass
        except Exception as e:
            print_error(f"Server error: {e}")
        finally:
            # Ensure all aiohttp sessions are properly closed - silently
            if client and hasattr(client, 'cleanup'):
                try:
                    await client.cleanup()
                except Exception:
                    pass
            
            # Close any remaining event loop resources - silently
            for task in asyncio.all_tasks(asyncio.get_event_loop()):
                if not task.done():
                    task.cancel()
            
            # Find and close any lingering aiohttp sessions - silently
            try:
                import inspect
                for obj in list(gc.get_objects()):
                    # Only check objects that are actually aiohttp.ClientSession instances
                    if not inspect.isclass(obj) and obj.__class__.__name__ == 'ClientSession':
                        if hasattr(obj, 'closed') and not obj.closed:
                            try:
                                await obj.close()
                            except Exception:
                                pass
            except Exception:
                pass
    
    # Import garbage collector
    import gc
    
    # Suppress all tracebacks and error messages when exiting
    import sys

    # Redirect stderr to null device to suppress all error output on exit
    class SuppressOutput:
        def __enter__(self):
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.close()
            sys.stderr = self._original_stderr
            # Return True to suppress exception propagation
            return True
    
    # Use a combination of signal handling and error suppression
    try:
        # Only show output during normal operation, hide errors on exit
        asyncio.run(main())
    except KeyboardInterrupt:
        # Exit silently with no error trace
        with SuppressOutput():
            # Force immediate exit without any further error handling
            os._exit(0)

@cli.command('check-updates')
def check_updates():
    """Check if a newer version of Oblix is available"""
    from oblix import check_for_updates, version_info
    
    current_info = version_info()
    update_info = check_for_updates(print_notification=True)
    
    if not update_info.get('update_available'):
        print_success(f"You are using the latest version of Oblix ({current_info['version']}).")

@cli.command('chat')
@click.option('--local-model', required=True, help='Local model (e.g., ollama:llama2)')
@click.option('--cloud-model', required=True, help='Cloud model (e.g., openai:gpt-3.5-turbo or claude:claude-3-haiku)')
@click.option('--cloud-api-key', help='API key for cloud model (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
def chat(local_model, cloud_model, cloud_api_key):
    """Start an interactive AI chat in your terminal"""
    async def run_chat(passed_cloud_api_key):
        client = None
        try:
            # Create client without API key
            client = await setup_client()
            
            # Parse the local model parameter (provider:model_name)
            local_model_parts = local_model.lower().split(':', 1)
            if len(local_model_parts) != 2:
                print_error(f"Invalid local model format: {local_model}. Should be 'provider:model_name'")
                print("Example: 'ollama:llama2'")
                return
                
            local_provider, local_model_name = local_model_parts
            
            # Verify the provider is ollama
            if local_provider != 'ollama':
                print_error(f"Unsupported local model provider: {local_provider}")
                print("Only 'ollama' is supported as a local provider")
                return
                
            # Hook local Ollama model
            local_hook_success = await client.hook_model(
                model_type=ModelType.OLLAMA, 
                model_name=local_model_name
            )
            if not local_hook_success:
                print_error(f"Failed to hook local Ollama model: {local_model}")
                return
            
            # Determine cloud model type and hook
            cloud_model_parts = cloud_model.lower().split(':', 1)
            if len(cloud_model_parts) != 2:
                print_error(f"Invalid cloud model format: {cloud_model}. Should be 'provider:model_name'")
                print("Example: 'openai:gpt-3.5-turbo' or 'claude:claude-3-haiku'")
                return
                
            cloud_provider, cloud_model_name = cloud_model_parts
            
            # Map cloud providers to ModelType
            cloud_model_map = {
                'openai': ModelType.OPENAI,
                'claude': ModelType.CLAUDE
            }
            
            if cloud_provider not in cloud_model_map:
                print_error(f"Unsupported cloud model provider: {cloud_provider}")
                print("Supported providers: openai, claude")
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
                model_name=cloud_model_name,
                api_key=local_cloud_api_key
            )
            
            if not cloud_hook_success:
                print_error(f"Failed to hook cloud model: {cloud_model}")
                return
            
            # Automatically hook monitoring agents
            from oblix.agents.resource_monitor import ResourceMonitor
            from oblix.agents.connectivity import ConnectivityAgent
            
            print_info("Initializing monitoring agents...")
            
            # Initialize and hook all monitoring agents at once
            resource_monitor = ResourceMonitor(name="resource_monitor_agent")
            connectivity_agent = ConnectivityAgent()
            
            # Hook all agents
            client.hook_agent(resource_monitor)
            client.hook_agent(connectivity_agent)
            
            # Start chat session
            await client.start_chat()
        
        except Exception as e:
            # For non-auth errors, still show the error but no traceback in production
            print_error(f"Chat setup error: {e}")
            
            # Only show traceback in debug mode
            if logging.getLogger().level == logging.DEBUG:
                import traceback
                traceback.print_exc()
        finally:
            # Explicitly clean up the client
            if client:
                try:
                    # Call client cleanup explicitly
                    await client.cleanup()
                except Exception as e:
                    # Silent error, just for debugging
                    logger.debug(f"Error during client cleanup: {e}")
                    
            # Extra sleep to allow event loop to process cleanup
            await asyncio.sleep(0.2)
    
    # Resolve cloud API key before passing to async function
    # Check for cloud provider-specific environment variables
    cloud_provider = cloud_model.lower().split(':', 1)[0] if ':' in cloud_model else None
    
    if cloud_provider == 'claude':
        # For Claude models, check ANTHROPIC_API_KEY environment variable
        resolved_cloud_api_key = cloud_api_key or os.getenv('ANTHROPIC_API_KEY')
        if not resolved_cloud_api_key:
            print_error("No API key provided for Claude model. Please provide --cloud-api-key or set ANTHROPIC_API_KEY environment variable.")
            return
    elif cloud_provider == 'openai':
        # For OpenAI models, check OPENAI_API_KEY environment variable
        resolved_cloud_api_key = cloud_api_key or os.getenv('OPENAI_API_KEY')
        if not resolved_cloud_api_key:
            print_error("No API key provided for OpenAI model. Please provide --cloud-api-key or set OPENAI_API_KEY environment variable.")
            return
    else:
        print_error(f"Unsupported cloud provider: {cloud_provider}")
        print_info("Supported providers: openai, claude")
        return
    
    # Use handle_async_command to run the async function
    async def wrapped_run_chat():
        try:
            await run_chat(resolved_cloud_api_key)
        finally:
            # Force garbage collection to help find and close any remaining client sessions
            try:
                import gc
                import inspect
                
                # Run garbage collection to find unreferenced sessions
                gc.collect()
                
                # Find and explicitly close any lingering aiohttp sessions
                for obj in gc.get_objects():
                    try:
                        # Check if this is a client session by class name
                        if (not inspect.isclass(obj) and 
                            hasattr(obj, '__class__') and 
                            obj.__class__.__name__ == 'ClientSession'):
                            
                            # Check if it's closed
                            if hasattr(obj, 'closed') and not obj.closed:
                                try:
                                    # Try to close it
                                    await obj.close()
                                except Exception:
                                    pass
                                    
                        # Also check for connectors
                        elif (not inspect.isclass(obj) and 
                              hasattr(obj, '__class__') and 
                              obj.__class__.__name__ == 'TCPConnector'):
                              
                            # Check if it's closed
                            if hasattr(obj, 'closed') and not obj.closed:
                                try:
                                    # Try to close it
                                    await obj.close()
                                except Exception:
                                    pass
                    except Exception:
                        # Skip any objects that cause errors when accessing attributes
                        pass
                        
                # Wait a moment for event loop to process cleanup
                await asyncio.sleep(0.5)
            except Exception:
                # Ignore any errors from cleanup
                pass
    
    handle_async_command(wrapped_run_chat)


if __name__ == '__main__':
    cli()
