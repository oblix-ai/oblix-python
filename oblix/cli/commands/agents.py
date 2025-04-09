# oblix/cli/commands/agents.py
import click
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

@click.command(name='agents')
def agents_group():
    """Show the monitoring agents that help Oblix make smart decisions"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}Supported Agents{Style.RESET_ALL}\n")
    print(f"{Fore.CYAN}Oblix supports the following agents for system monitoring and task management:{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Resource Monitor{Style.RESET_ALL}")
    print("  • Monitors system resources like CPU, memory, and GPU")
    print("  • Helps make intelligent routing decisions based on resource availability")
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Connectivity Agent{Style.RESET_ALL}")
    print("  • Monitors network connectivity and latency")
    print("  • Routes requests to local models when connectivity is limited")
    
    print()