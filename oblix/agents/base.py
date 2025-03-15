# app/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """
    Base class for all agents in the Oblix system
    
    Agents can perform checks, intercept, or modify execution based on custom logic.
    They provide a pluggable architecture for enhancing the Oblix system with
    monitoring, policy enforcement, and decision-making capabilities.
    
    Examples of agents include:
    - ResourceMonitor: Monitors system resources and recommends local/cloud execution
    - ConnectivityAgent: Monitors network connectivity and manages fallbacks
    - SecurityAgent: Validates requests against security policies
    - ContentFilterAgent: Screens content for compliance with guidelines
    
    Attributes:
        name (str): Unique identifier for the agent
        is_active (bool): Whether the agent is currently active
    """
    
    def __init__(self, name: str):
        """
        Initialize the agent
        
        Args:
            name: Unique name for the agent
        """
        self.name = name
        self.is_active = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent
        
        This method should perform all necessary setup steps for the agent,
        including:
        - Setting up monitoring or connections
        - Loading configuration
        - Initializing internal state
        - Validating requirements
        
        Returns:
            bool: True if initialization is successful, False otherwise
            
        Raises:
            Exception: If initialization encounters an error
        """
        pass
    
    @abstractmethod
    async def check(self, **kwargs) -> Dict[str, Any]:
        """
        Perform agent-specific checks or modifications
        
        This method is called during execution to allow the agent to:
        - Monitor system state
        - Apply business rules or policies
        - Make routing decisions
        - Modify execution parameters
        
        The return dictionary should include a 'proceed' key with a boolean
        value indicating whether execution should continue.
        
        Args:
            **kwargs: Flexible keyword arguments for context, which may include:
                - prompt: The user prompt being processed
                - model_type: The model type being used
                - model_name: The specific model being used
                - parameters: Generation parameters
                
        Returns:
            Dict[str, Any]: Dictionary with check results and potential modifications.
            Should contain at minimum:
            {
                'proceed': bool,  # Whether execution should proceed
                # Additional agent-specific keys
            }
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shut down the agent and release any resources
        
        This method should handle proper cleanup of resources including:
        - Closing connections
        - Stopping background tasks
        - Releasing system resources
        - Saving state if needed
        
        Raises:
            Exception: If shutdown encounters an error
        """
        pass