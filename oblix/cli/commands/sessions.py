# oblix/cli/commands/sessions.py
import click
import sys
import os
import colorama
from colorama import Fore, Style
from datetime import datetime

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

@click.group(name='sessions')
def sessions_group():
    """View and manage your chat history"""
    pass

@sessions_group.command('list')
@click.option('--limit', default=50, help='Limit number of sessions to display')
def list_sessions(limit):
    """List recent chat sessions"""
    async def run_list_sessions():
        try:
            client = await setup_client()
            sessions = await client.list_sessions(limit)
            
            if not sessions:
                print_warning("No sessions found.")
                return
            
            rows = []
            for session in sessions:
                created_at = datetime.fromisoformat(session['created_at']).strftime("%Y-%m-%d %H:%M")
                rows.append([
                    session['id'], 
                    session['title'], 
                    created_at, 
                    str(session['message_count'])
                ])
            
            print_table("Recent Chat Sessions", ["Session ID", "Title", "Created", "Messages"], rows)
        except Exception as e:
            print_error(f"Error listing sessions: {e}")
    
    handle_async_command(run_list_sessions)

@sessions_group.command('view')
@click.argument('session_id')
def view_session(session_id):
    """View details of a specific session"""
    async def run_view_session():
        try:
            client = await setup_client()
            session_data = await client.load_session(session_id)
            
            if not session_data:
                print_warning(f"No session found with ID: {session_id}")
                return
            
            # Create session overview content
            overview_content = (
                f"Session ID: {session_data['id']}\n"
                f"Title: {session_data['title']}\n"
                f"Created: {session_data['created_at']}\n"
                f"Last Updated: {session_data['updated_at']}"
            )
            print_panel("Session Overview", overview_content)
            
            # Display messages
            print_header("Conversation:")
            for msg in session_data.get('messages', []):
                role = msg['role'].capitalize()
                timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                
                if role == 'User':
                    print(f"{Fore.GREEN}{role} ({timestamp}):{Style.RESET_ALL} {msg['content']}")
                else:
                    print(f"{Fore.BLUE}{role} ({timestamp}):{Style.RESET_ALL} {msg['content']}")
        
        except Exception as e:
            print_error(f"Error viewing session: {e}")
    
    handle_async_command(run_view_session)

@sessions_group.command('delete')
@click.argument('session_id')
@click.confirmation_option(prompt='Are you sure you want to delete this session?')
def delete_session(session_id):
    """Delete a specific chat session"""
    async def run_delete_session():
        try:
            client = await setup_client()
            success = await client.delete_session(session_id)
            
            if success:
                print_success(f"Session {session_id} deleted successfully.")
            else:
                print_warning(f"Could not delete session {session_id}.")
        
        except Exception as e:
            print_error(f"Error deleting session: {e}")
    
    handle_async_command(run_delete_session)

@sessions_group.command('create')
@click.option('--title', help='Title for the new session')
def create_session(title):
    """Create a new chat session"""
    async def run_create_session():
        try:
            client = await setup_client()
            session_id = await client.create_session(title=title)
            
            print_success(f"New session created: {session_id}")
            print_info(f"Title: {title or 'Untitled Session'}")
        
        except Exception as e:
            print_error(f"Error creating session: {e}")
    
    handle_async_command(run_create_session)