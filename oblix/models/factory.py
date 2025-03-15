# oblix/models/factory.py
from typing import Optional, Dict, Any, Tuple
import logging
from .base import BaseModel, ModelType
from .ollama import OllamaModel
from .openai import OpenAIModel
from .claude import ClaudeModel
from .validation import ModelValidator, ValidationResult

logger = logging.getLogger(__name__)

class ModelCreationError(Exception):
    """
    Custom exception for model creation errors
    
    This exception provides structured information about failures during
    model creation, including validation results when available.
    
    Attributes:
        message (str): Error message
        validation_result (Optional[ValidationResult]): Validation result if available
    """
    def __init__(self, message: str, validation_result: Optional[ValidationResult] = None):
        super().__init__(message)
        self.validation_result = validation_result

class ModelFactory:
    """
    Factory class for creating and validating model instances
    
    The ModelFactory simplifies the creation and validation of model instances
    from configuration dictionaries. It ensures consistent model initialization,
    validation, and error handling across different model types.
    
    This factory handles:
    - Configuration validation
    - Model type detection and instantiation
    - Model initialization
    - Error handling and reporting
    
    Attributes:
        validator (ModelValidator): Validator for model configurations
    """
    
    def __init__(self):
        """
        Initialize the model factory with a validator
        """
        self.validator = ModelValidator()

    async def create_model(self, config: Dict[str, Any]) -> Optional[BaseModel]:
        """
        Create and initialize a model instance with validation
        
        This method handles the complete lifecycle of model creation:
        1. Validates the configuration
        2. Creates the appropriate model instance based on type
        3. Initializes the model
        4. Handles errors at each stage
        
        Args:
            config: Model configuration dictionary containing at minimum:
                - type: Model type (ModelType or string)
                - name: Model name
                Additional provider-specific keys may be required:
                - api_key: For OpenAI and Claude models
                - endpoint: For Ollama models (optional)
        
        Returns:
            Initialized model instance or None
        
        Raises:
            ModelCreationError: If validation or initialization fails
            
        Example:
            config = {
                'type': ModelType.OPENAI,
                'name': 'gpt-3.5-turbo',
                'api_key': 'sk-...'
            }
            model = await factory.create_model(config)
        """
        try:
            # Validate configuration
            validation_result = await self.validator.validate_model_config(config)
            
            if not validation_result.is_valid:
                error_messages = [
                    f"{error.error_type.value}: {error.message}"
                    for error in validation_result.errors
                ]
                raise ModelCreationError(
                    f"Model validation failed:\n" + "\n".join(error_messages),
                    validation_result
                )

            # Log any warnings
            for warning in validation_result.warnings:
                logger.warning(f"Model {config.get('name')} warning: {warning}")

            # Create model instance based on type
            model_type = validation_result.model_type
            model_name = validation_result.model_name

            if model_type == ModelType.OLLAMA:
                model = OllamaModel(
                    model_name=model_name,
                    endpoint=config.get('endpoint') or "http://localhost:11434"
                )
            elif model_type == ModelType.OPENAI:
                model = OpenAIModel(
                    model_name=model_name,
                    api_key=config['api_key']
                )
            elif model_type == ModelType.CLAUDE:
                model = ClaudeModel(
                    model_name=model_name,
                    api_key=config['api_key'],
                    max_tokens=config.get('max_tokens')
                )
            else:
                raise ModelCreationError(
                    f"Unsupported model type: {model_type}",
                    validation_result
                )

            # Initialize the model
            if not await model.initialize():
                raise ModelCreationError(
                    f"Failed to initialize {model_type.value} model: {model_name}",
                    validation_result
                )

            logger.info(
                f"Successfully created and initialized {model_type.value} "
                f"model: {model_name}"
            )

            return model

        except ModelCreationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating model: {e}")
            raise ModelCreationError(f"Unexpected error creating model: {str(e)}")

    async def cleanup(self):
        """
        Clean up resources
        
        This method should be called when the factory is no longer needed
        to ensure all resources are properly released.
        """
        await self.validator.cleanup()