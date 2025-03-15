# app/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    """
    Supported model types in the Oblix system.
    
    This enumeration defines the types of AI models that can be integrated
    with the Oblix SDK, each corresponding to a specific model provider.
    
    Attributes:
        OLLAMA (str): Local LLM models served via Ollama
        OPENAI (str): Cloud models provided by OpenAI (GPT series)
        CLAUDE (str): Cloud models provided by Anthropic (Claude series)
        CUSTOM (str): Custom model implementations
    """
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLAUDE = "claude"
    CUSTOM = "custom"

class BaseModel(ABC):
    """
    Base class for all AI models in the Oblix system.
    
    This abstract class defines the interface that all model implementations
    must follow to be compatible with the Oblix SDK. It provides a standard
    contract for initialization, text generation, and resource management.
    
    Concrete model implementations (like OllamaModel, OpenAIModel, ClaudeModel)
    should inherit from this class and implement its abstract methods.
    
    Attributes:
        model_type (ModelType): Type of the model (OLLAMA, OPENAI, etc.)
        model_name (str): Name of the specific model implementation
        is_ready (bool): Flag indicating whether the model is initialized and ready
    """
    
    def __init__(self, model_type: ModelType, model_name: str):
        """
        Initialize the base model.
        
        Args:
            model_type (ModelType): Type of the model
            model_name (str): Name of the specific model implementation
        """
        self.model_type = model_type
        self.model_name = model_name
        self.is_ready = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the model with required setup.
        
        This method should perform all necessary initialization steps including:
        - Setting up API clients or connections
        - Validating credentials and access
        - Checking model availability
        - Loading required resources
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        This method handles the main text generation functionality, taking
        a user prompt and producing a response according to the model's
        capabilities.
        
        Args:
            prompt (str): User input prompt to process
            **kwargs: Additional model-specific parameters such as:
                - temperature: Controls randomness (0.0 to 1.0)
                - max_tokens: Maximum tokens to generate
                - context: Previous conversation context
                - request_id: Unique identifier for the request
        
        Returns:
            Dict[str, Any]: Dictionary containing at minimum:
                - response (str): Generated text
                - metrics (Dict): Performance metrics (optional)
        
        Raises:
            RuntimeError: If the model is not initialized
            Exception: Model-specific exceptions during generation
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Clean up resources when shutting down.
        
        This method should handle proper cleanup of resources including:
        - Closing network connections
        - Releasing memory
        - Stopping background processes
        - Saving state if needed
        """
        pass
    
    def __str__(self) -> str:
        """
        Get string representation of the model.
        
        Returns:
            str: String in format "{model_type}:{model_name}"
        """
        return f"{self.model_type.value}:{self.model_name}"

class BaseAgent(ABC):
    """
    Base class for agents like resource monitors and connectivity checkers.
    
    Agents are components that can observe system state, perform checks,
    and influence decision-making in the Oblix system. They run alongside
    models to provide additional capabilities like resource monitoring,
    connectivity checking, and policy enforcement.
    
    Attributes:
        name (str): Unique name of the agent
        is_active (bool): Flag indicating whether the agent is active
    """
    
    def __init__(self, name: str):
        """
        Initialize the agent.
        
        Args:
            name (str): Unique name for the agent
        """
        self.name = name
        self.is_active = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent.
        
        This method should perform all necessary setup steps including:
        - Setting up monitoring
        - Initializing internal state
        - Validating requirements
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def check(self, **kwargs) -> Dict[str, Any]:
        """
        Run agent checks.
        
        This method executes the agent's primary function, which may include:
        - Monitoring system resources
        - Checking network connectivity
        - Validating input parameters
        - Applying business rules or policies
        
        Args:
            **kwargs: Context-specific parameters for the check
        
        Returns:
            Dict[str, Any]: Check results, typically including:
                - proceed (bool): Whether execution should proceed
                - state (str): Current state assessment
                - target (str): Recommended execution target
                - metrics (Dict): Relevant metrics
                - reason (str): Reasoning for the decision
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Clean up resources.
        
        This method should handle proper cleanup of resources including:
        - Stopping monitoring
        - Closing connections
        - Releasing system resources
        """
        pass