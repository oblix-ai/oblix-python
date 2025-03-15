# oblix/cli/commands/agents.py
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

@click.group(name='agents')
def agents_group():
    """Manage and interact with Oblix agents"""
    pass

@agents_group.command('connectivity')
@click.option('--api-key', help='Oblix API key (optional)', envvar='OBLIX_API_KEY')
def connectivity_metrics(api_key):
    """Retrieve current connectivity metrics"""
    async def run_connectivity_metrics():
        try:
            client = await setup_client(api_key)
            metrics = await client.get_connectivity_metrics()
            
            if metrics:
                # Create rows for table
                rows = []
                for key, value in metrics.items():
                    rows.append([key, value])
                
                print_table("Connectivity Metrics", ["Metric", "Value"], rows)
            else:
                print_warning("No connectivity metrics available.")
        except Exception as e:
            print_error(f"Connectivity metrics error: {e}")
    
    handle_async_command(run_connectivity_metrics)

@agents_group.command('resource')
@click.option('--api-key', help='Oblix API key (optional)', envvar='OBLIX_API_KEY')
def resource_check(api_key):
    """Perform a system resource check"""
    async def run_resource_check():
        try:
            # Note: We'll simulate resource check via connectivity metrics as a placeholder
            client = await setup_client(api_key)
            
            # You might want to create a specific resource monitoring method in the client
            print_header("Performing system resource check...")
            
            metrics = await client.get_connectivity_metrics()
            
            if metrics:
                content = (
                    f"Connection Type: {metrics.get('connection_type', 'Unknown')}\n"
                    f"Bandwidth: {metrics.get('bandwidth', 'N/A')} Mbps\n"
                    f"Latency: {metrics.get('latency', 'N/A')} ms\n"
                    f"Packet Loss: {metrics.get('packet_loss', 'N/A')}%"
                )
                print_panel("System Resource Overview", content)
            else:
                print_warning("Unable to retrieve resource metrics.")
        except Exception as e:
            print_error(f"Resource check error: {e}")
    
    handle_async_command(run_resource_check)