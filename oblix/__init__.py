"""
Oblix: AI Model Orchestration SDK

Oblix is a comprehensive SDK for orchestrating AI models with automatic
switching between local and cloud-based models based on connectivity and
system resource availability. The SDK provides:

- Unified interface for multiple AI model providers (OpenAI, Claude, Ollama)
- Intelligent routing between local and cloud models
- System resource monitoring and connectivity awareness
- Persistent chat session management
- Agent system for extensible monitoring and decision making
- CLI tools for common operations

Examples:
    # Basic usage
    from oblix import OblixClient, ModelType
    
    # Initialize client
    client = OblixClient(oblix_api_key="your_api_key")
    
    # Hook models
    await client.hook_model(ModelType.OLLAMA, "llama2")
    await client.hook_model(ModelType.OPENAI, "gpt-3.5-turbo", api_key="sk-...")
    
    # Add monitoring agents
    from oblix.agents.resource_monitor import ResourceMonitor
    from oblix.agents.connectivity import ConnectivityAgent
    
    client.hook_agent(ResourceMonitor())
    client.hook_agent(ConnectivityAgent())
    
    # Execute prompt with automatic model routing
    response = await client.execute("Explain quantum computing")
    print(response["response"])
    
    # Start interactive chat session
    await client.start_chat()
"""

# Main client classes
from .client.client import OblixClient
from .client.base_client import OblixBaseClient

# Model type definitions
from .models.base import ModelType

# Core agent classes
from .agents.base import BaseAgent

# Authentication classes
from .auth import OblixAuth, APIKeyValidationError, RateLimitExceededError

# Export main client class with alternative name for convenience
Oblix = OblixClient

__all__ = [
    # Main client classes
    'Oblix',
    'OblixClient',
    'OblixBaseClient',
    
    # Model types
    'ModelType',
    
    # Agent classes
    'BaseAgent',
    
    # Authentication
    'OblixAuth',
    'APIKeyValidationError',
    'RateLimitExceededError',
    
    # Version utilities
    'version_info',
    'check_for_updates'
]

__version__ = '0.1.0'

def version_info():
    """
    Return version information for the Oblix SDK.
    
    Returns:
        dict: Dictionary containing version details
    """
    return {
        'version': __version__,
        'name': 'Oblix AI Orchestration SDK',
        'models_supported': ['OpenAI', 'Claude', 'Ollama'],
        'python_requires': '>=3.8'
    }
    
def check_for_updates(print_notification=True):
    """
    Check if a newer version of Oblix is available on PyPI.
    
    Args:
        print_notification (bool): Whether to print update notification to console
        
    Returns:
        dict: Update information including current version, latest version, and update available status
    """
    import json
    import urllib.request
    from packaging import version
    
    current_version = __version__
    
    try:
        # Query PyPI API for the latest version
        with urllib.request.urlopen("https://pypi.org/pypi/oblix/json", timeout=2) as response:
            data = json.loads(response.read())
            latest_version = data["info"]["version"]
            
        # Compare versions
        update_available = version.parse(latest_version) > version.parse(current_version)
        
        # Print notification if requested and update is available
        if print_notification and update_available:
            print(f"\n⚠️ Update available: Oblix {latest_version} is now available. You are using {current_version}.")
            print("To update, run: pip install --upgrade oblix\n")
            
        return {
            "current_version": current_version,
            "latest_version": latest_version,
            "update_available": update_available
        }
    except Exception:
        # Fail silently if unable to check for updates
        return {
            "current_version": current_version,
            "latest_version": None,
            "update_available": False
        }
