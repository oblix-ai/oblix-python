# oblix/cli/commands/server.py
import click
import sys
import os
import asyncio
import colorama
from colorama import Fore, Style
import uvicorn
import logging
import threading
import time

from oblix.cli.utils import setup_client, handle_async_command

# Initialize colorama
colorama.init()

logger = logging.getLogger("oblix.cli.server")

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

@click.group(name='server')
def server_group():
    """Start and manage the Oblix server"""
    pass

@server_group.command('start')
@click.option('--port', default=62549, help='Port to run the server on')
@click.option('--host', default='0.0.0.0', help='Host to bind the server to')
def start_server(port, host):
    """
    Start the Oblix server with OpenAI-compatible API
    
    Important: Before starting the server, you should hook at least one model
    
    Example workflow:
      oblix models hook         # Hook a model to use
      oblix server start        # Start the server
    """
    # ANSI escape codes for colored text
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def run_server_process():
        """Run the FastAPI server in a separate thread"""
        from oblix.main import app
        
        # Check if port is already in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host if host != '0.0.0.0' else '127.0.0.1', port))
        sock.close()
        
        if result == 0:
            # Port is already in use
            print_error(f"\nPort {port} is already in use!")
            print_warning("This could mean:")
            print_warning("1. Another Oblix server is already running")
            print_warning("2. Another application is using this port")
            print_warning("\nChoose a different port with: --port <number>")
            print_warning("Or check if an Oblix server is already running:")
            print_warning(f"  oblix server status --port {port}")
            sys.exit(1)
        
        # Start the server
        try:
            uvicorn.run(app, host=host, port=port)
        except Exception as e:
            print_error(f"\nFailed to start server: {e}")
            print_warning("Try troubleshooting with:")
            print_warning("1. Use a different port: --port <number>")
            print_warning("2. Make sure you have permissions to bind to the specified port")
            print_warning("3. Check if another service is running on this port")
            sys.exit(1)
    
    async def prepare_server():
        """Check if the configuration is valid before starting the server"""
        try:
            # Get the Oblix client
            from oblix.config import ConfigManager
            from oblix.client import OblixClient
            from oblix.api.routes import OblixAPIManager
            
            # First, check if Oblix is properly initialized
            config_manager = ConfigManager()
            
            # Create the client without authentication
            client = OblixClient()
            
            # Store the client in the API Manager
            OblixAPIManager._instance = client
            
            # Check if both local and cloud models are configured
            models = client.list_models()
            
            # Check for local models (Ollama)
            has_local_model = False
            local_model_names = models.get('ollama', [])
            if local_model_names:
                has_local_model = True
                
            # Check for cloud models (OpenAI or Claude)
            has_cloud_model = False
            openai_models = models.get('openai', [])
            claude_models = models.get('claude', [])
            
            if openai_models or claude_models:
                has_cloud_model = True
            
            # Show which models are configured
            print_info("\nConfigured models:")
            for model_type, model_names in models.items():
                if model_names:
                    print_info(f"  {model_type}: {', '.join(model_names)}")
                    
            # Require both local and cloud models for orchestration
            if not has_local_model or not has_cloud_model:
                print_error("\n⚠️ Oblix orchestration requires both a local and a cloud model!")
                
                if not has_local_model:
                    print_error("Missing: Local model (Ollama)")
                    print_warning("Hook a local model with:")
                    print_warning("  oblix models hook --type ollama --name llama2 --endpoint http://localhost:11434")
                
                if not has_cloud_model:
                    print_error("Missing: Cloud model (OpenAI or Claude)")
                    print_warning("Hook a cloud model with:")
                    print_warning("  oblix models hook --type openai --name gpt-3.5-turbo --api-key YOUR_OPENAI_KEY")
                
                sys.exit(1)
            
            # Add default agents if not already hooked
            if 'resource_monitor' not in client.agents:
                from oblix.agents.resource_monitor import ResourceMonitor
                print_info("Adding resource monitoring agent...")
                client.hook_agent(ResourceMonitor(name="resource_monitor"))
            
            if 'connectivity' not in client.agents:
                from oblix.agents.connectivity import ConnectivityAgent
                print_info("Adding connectivity monitoring agent...")
                client.hook_agent(ConnectivityAgent(name="connectivity"))
            
            # Create an ASCII art header for the server
            header = f"""
{MAGENTA}{BOLD}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓{RESET}
{MAGENTA}{BOLD}┃                  OBLIX SERVER READY                    ┃{RESET}
{MAGENTA}{BOLD}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛{RESET}

{CYAN}OpenAI-compatible endpoint:{RESET} {GREEN}http://localhost:{port}/v1/chat/completions{RESET}
{CYAN}Health check endpoint:{RESET}      {GREEN}http://localhost:{port}/health{RESET}

{YELLOW}Use with OpenAI client:{RESET}
  from openai import OpenAI
  client = OpenAI(base_url="{GREEN}http://localhost:{port}/v1{RESET}", api_key="oblix-dev")
  response = client.chat.completions.create(
    model="model-name",  # Use "ollama:llama2" or "openai:gpt-3.5-turbo"
    messages=[{{"role": "user", "content": "Hello, world!"}}]
  )

{YELLOW}Press Ctrl+C to stop the server.{RESET}
"""
            print(header)
            
            # Start the server in a thread
            server_thread = threading.Thread(target=run_server_process)
            server_thread.daemon = True
            server_thread.start()
            
            # Keep the main thread running to maintain client instance
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print_info("\nShutting down server...")
            print_success("Server stopped.")
            sys.exit(0)
        except Exception as e:
            print_error(f"Error preparing server: {e}")
            if logging.getLogger().level == logging.DEBUG:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    # Run the async prepare function
    handle_async_command(prepare_server)

@server_group.command('status')
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=62549, help='Server port')
def server_status(host, port):
    """Check the status of the Oblix server"""
    import requests
    
    try:
        url = f"http://{host}:{port}/health"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Server is running. Status: {data.get('status', 'unknown')}")
            print_info(f"Version: {data.get('version', 'unknown')}")
        else:
            print_warning(f"Server responded with status code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print_error(f"Could not connect to server at {host}:{port}")
        print_info("Make sure the server is running with: oblix server start")
    except Exception as e:
        print_error(f"Error checking server status: {e}")