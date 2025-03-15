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
        "name": "claude-3-opus-20240229",
        "description": "Most powerful Claude model",
        "context_window": "200K tokens",
        "features": "Complex tasks, code analysis, long-form content"
    },
    {
        "name": "claude-3-sonnet-20240229",
        "description": "Balanced performance model",
        "context_window": "200K tokens",
        "features": "Fast, high quality, good reasoning"
    },
    {
        "name": "claude-3-haiku-20240307",
        "description": "Fastest Claude model",
        "context_window": "200K tokens",
        "features": "Quick responses, everyday tasks"
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
    return any(model["name"] == model_name for model in models)
