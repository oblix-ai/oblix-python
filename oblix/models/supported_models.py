# oblix/models/supported_models.py
from typing import List, Dict

SUPPORTED_OPENAI_MODELS: List[Dict[str, str]] = [
    {
        "name": "gpt-4-turbo-preview",
        "description": "Most capable GPT-4 model",
        "context_window": "128K tokens",
        "features": "Latest model, high accuracy, JSON mode"
    },
    {
        "name": "gpt-4",
        "description": "More capable than GPT-3.5",
        "context_window": "8K tokens",
        "features": "High reasoning, creative tasks"
    },
    {
        "name": "gpt-3.5-turbo",
        "description": "Most capable GPT-3.5 model",
        "context_window": "16K tokens",
        "features": "Fast, cost-effective, good accuracy"
    }
]

SUPPORTED_CLAUDE_MODELS: List[Dict[str, str]] = [
    {
        "name": "claude-3-7-sonnet-20250219",
        "description": "Latest and most capable Claude model",
        "context_window": "200K tokens",
        "features": "Superior reasoning, code, and creative tasks"
    },
    {
        "name": "claude-3-5-sonnet-20241022",
        "description": "Improved Claude 3.5 sonnet model",
        "context_window": "200K tokens",
        "features": "Strong general purpose assistant"
    },
    {
        "name": "claude-3-5-haiku-20241022",
        "description": "Improved Claude 3.5 haiku model",
        "context_window": "200K tokens",
        "features": "Fast responses for simpler tasks"
    },
    {
        "name": "claude-3-5-sonnet-20240620",
        "description": "Claude 3.5 sonnet model (June 2024)",
        "context_window": "200K tokens",
        "features": "Balanced performance model"
    },
    {
        "name": "claude-3-haiku-20240307",
        "description": "Claude 3 haiku model",
        "context_window": "200K tokens",
        "features": "Quick responses, everyday tasks"
    },
    {
        "name": "claude-3-opus-20240229",
        "description": "Original Claude 3 opus model",
        "context_window": "200K tokens",
        "features": "Complex tasks, code analysis, long-form content"
    }
]

def get_supported_models(provider: str) -> List[Dict[str, str]]:
    """Get supported models for a specific provider"""
    provider_map = {
        "openai": SUPPORTED_OPENAI_MODELS,
        "claude": SUPPORTED_CLAUDE_MODELS,
    }
    return provider_map.get(provider.lower(), [])

def is_model_supported(provider: str, model_name: str) -> bool:
    """Check if a specific model is supported for API-based providers"""
    if provider.lower() == "ollama":
        # For Ollama, we don't maintain a static list as it's dynamic
        return True  # Allow any model name as it's verified during runtime
    
    models = get_supported_models(provider)
    is_supported = any(model["name"] == model_name for model in models)
    
    # For Claude, be more permissive - new models are released frequently
    # and we want to support them without code changes
    if provider.lower() == "claude":
        # Accept all Claude models that follow the standard naming pattern
        if model_name.startswith("claude-"):
            return True
            
    return is_supported
