# oblix/client/client.py
from typing import Dict, Any, Optional, List
import logging
import os
import asyncio
import uuid
from datetime import datetime

from .base_client import OblixBaseClient
from ..models.base import ModelType
from ..agents.base import BaseAgent
from ..agents.resource_monitor import ResourceMonitor
from ..agents.connectivity import ConnectivityAgent

logger = logging.getLogger(__name__)

class OblixClient(OblixBaseClient):
    """
    Main client class for Oblix AI Orchestration SDK.
    
    OblixClient provides a high-level, developer-friendly interface for working with
    multiple AI models, managing sessions, and utilizing intelligent routing
    based on system resources and connectivity.
    
    This client extends OblixBaseClient with convenient methods for execution,
    chat management, and common operations while abstracting away the complexity
    of model routing, resource monitoring, and connectivity management.
    
    Attributes:
        models (Dict): Dictionary of registered models
        agents (Dict): Dictionary of registered agents
        current_session_id (Optional[str]): ID of the active chat session
    
    Examples:
        # Initialize client
        client = OblixClient(oblix_api_key="your_api_key")
        
        # Hook models
        await client.hook_model(ModelType.OLLAMA, "llama2")
        await client.hook_model(ModelType.OPENAI, "gpt-3.5-turbo", api_key="sk-...")
        
        # Add monitoring
        client.hook_agent(ResourceMonitor())
        
        # Execute prompt
        response = await client.execute("Explain quantum computing")
        print(response["response"])
        
        # Manage sessions
        session_id = await client.create_session("My Chat")
        sessions = client.list_sessions()
    """

    async def execute(self, 
                     prompt: str, 
                     model_id: Optional[str] = None,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     request_id: Optional[str] = None,
                     **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute a prompt using available models and agents with intelligent routing.
        
        This method handles the execution flow including:
        1. Using the ExecutionManager to determine the best model based on resource/connectivity
        2. Retrieving conversation context (if in an active session)
        3. Generating the response
        4. Saving the interaction to the session (if active)
        
        Args:
            prompt (str): User prompt to process
            model_id (Optional[str]): Specific model to use (optional)
            temperature (Optional[float]): Sampling temperature for text generation
            max_tokens (Optional[int]): Maximum tokens to generate
            request_id (Optional[str]): Custom request identifier for tracking
            **kwargs: Additional model-specific generation parameters
        
        Returns:
            Optional[Dict[str, Any]]: Response containing:
                - response: Generated text
                - metrics: Performance metrics
                - agent_checks: Results from agent checks
                - error: Error message (if execution failed)
        
        Raises:
            RuntimeError: If requirements validation fails
        """
        # Validate all requirements are met
        await self._validate_requirements()
        
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            # Get conversation context if session exists
            context = []
            if self.current_session_id:
                try:
                    session_data = self.session_manager.load_session(self.current_session_id)
                    if session_data and 'messages' in session_data:
                        # Get last 5 messages for context
                        context = session_data['messages'][-5:]
                except Exception as e:
                    logger.warning(f"Error retrieving session context: {e}")
            
            # Prepare parameters for execution
            parameters = {
                'request_id': request_id,
                'context': context
            }
            
            # Add optional parameters if provided
            if temperature is not None:
                parameters['temperature'] = temperature
            if max_tokens is not None:
                parameters['max_tokens'] = max_tokens
                
            # Add any additional kwargs
            parameters.update(kwargs)
            
            # Determine model type and model name from model_id if provided
            model_type = None
            model_name = None
            if model_id:
                if ':' in model_id:
                    model_type_str, model_name = model_id.split(':', 1)
                    try:
                        model_type = ModelType(model_type_str)
                    except ValueError:
                        logger.error(f"Invalid model type: {model_type_str}")
                        return {
                            "request_id": request_id,
                            "error": f"Invalid model type: {model_type_str}"
                        }
                else:
                    # Try to find model by ID
                    model = self.models.get(model_id)
                    if model:
                        model_type = model.model_type
                        model_name = model.model_name
                    else:
                        logger.error(f"Model {model_id} not found")
                        return {
                            "request_id": request_id,
                            "error": f"Model {model_id} not found"
                        }
            
            # Use ExecutionManager for orchestration and model selection
            execution_result = await self.execution_manager.execute(
                prompt,
                model_type=model_type,
                model_name=model_name,
                parameters=parameters
            )
            
            # Handle execution errors
            if 'error' in execution_result:
                logger.error(f"Execution error: {execution_result['error']}")
                return execution_result
            
            # Get response from execution result
            response = execution_result['response']

            # Save to session if active
            if self.current_session_id:
                try:
                    # Save user message
                    self.session_manager.save_message(
                        self.current_session_id,
                        prompt,
                        role='user'
                    )
                    
                    # Save assistant message
                    self.session_manager.save_message(
                        self.current_session_id,
                        response,
                        role='assistant'
                    )
                except Exception as e:
                    logger.warning(f"Error saving session messages: {e}")
            
            # Return response with execution metadata
            return {
                "request_id": request_id,
                "model_id": execution_result.get('model_id', ''),
                "response": response,
                "metrics": execution_result.get('metrics', {}),
                "agent_checks": execution_result.get('agent_checks', []),
                "routing_decision": execution_result.get('routing_decision', {})
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "request_id": request_id,
                "error": str(e)
            }
            
    async def execute_streaming(self, 
                          prompt: str, 
                          model_id: Optional[str] = None,
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          request_id: Optional[str] = None,
                          **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute a prompt with streaming output directly to the terminal.
        
        This method is similar to execute() but streams the response token-by-token
        as it's generated, rather than waiting for the complete response. 
        
        Args:
            prompt (str): User prompt to process
            model_id (Optional[str]): Specific model to use (optional)
            temperature (Optional[float]): Sampling temperature for text generation
            max_tokens (Optional[int]): Maximum tokens to generate
            request_id (Optional[str]): Custom request identifier for tracking
            **kwargs: Additional model-specific generation parameters
        
        Returns:
            Optional[Dict[str, Any]]: Final response containing:
                - response: Complete generated text
                - metrics: Performance metrics
                - agent_checks: Results from agent checks
                - error: Error message (if execution failed)
        
        Raises:
            RuntimeError: If requirements validation fails
        """
        # Validate all requirements are met
        await self._validate_requirements()
        
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            # Get conversation context if session exists
            context = []
            if self.current_session_id:
                try:
                    session_data = self.session_manager.load_session(self.current_session_id)
                    if session_data and 'messages' in session_data:
                        # Get last 5 messages for context
                        context = session_data['messages'][-5:]
                except Exception as e:
                    logger.warning(f"Error retrieving session context: {e}")
            
            # Prepare parameters for execution
            parameters = {
                'request_id': request_id,
                'context': context,
                'stream': True  # Enable streaming
            }
            
            # Add optional parameters if provided
            if temperature is not None:
                parameters['temperature'] = temperature
            if max_tokens is not None:
                parameters['max_tokens'] = max_tokens
                
            # Add any additional kwargs
            parameters.update(kwargs)
            
            # Determine model type and model name from model_id if provided
            model_type = None
            model_name = None
            if model_id:
                if ':' in model_id:
                    model_type_str, model_name = model_id.split(':', 1)
                    try:
                        model_type = ModelType(model_type_str)
                    except ValueError:
                        logger.error(f"Invalid model type: {model_type_str}")
                        return {
                            "request_id": request_id,
                            "error": f"Invalid model type: {model_type_str}"
                        }
                else:
                    # Try to find model by ID
                    model = self.models.get(model_id)
                    if model:
                        model_type = model.model_type
                        model_name = model.model_name
                    else:
                        logger.error(f"Model {model_id} not found")
                        return {
                            "request_id": request_id,
                            "error": f"Model {model_id} not found"
                        }
                        
            # Use ExecutionManager for orchestration
            execution_result = await self.execution_manager.execute(
                prompt,
                model_type=model_type,
                model_name=model_name,
                parameters=parameters
            )
            
            # Handle execution errors
            if 'error' in execution_result:
                logger.error(f"Execution error: {execution_result['error']}")
                return execution_result
                
            # Get the model that was selected
            model_identifier = execution_result.get('model_id', 'Unknown model')
            
            # Print which model was selected
            print(f"\nUsing model: {model_identifier}")
            
            # Get the response text from execution result (may already be complete for non-streaming models)
            if 'response' in execution_result:
                print(execution_result['response'])
                full_response = execution_result['response']
            else:
                # Otherwise we expect streaming data from the execution
                print("\nAssistant: ", end="", flush=True)
                full_response = ""
                
                # Extract streaming content if available
                if 'stream' in execution_result:
                    try:
                        async for token in execution_result['stream']:
                            print(token, end="", flush=True)
                            full_response += token
                        print()  # Add newline at the end
                    except Exception as e:
                        logger.error(f"Error streaming response: {e}")
                        print(f"\nError: {str(e)}")
                else:
                    print("\nError: No streaming response available")
                    full_response = "Streaming response not available"
                
            # Save to session if active
            if self.current_session_id:
                try:
                    # Save user message
                    self.session_manager.save_message(
                        self.current_session_id,
                        prompt,
                        role='user'
                    )
                    
                    # Save assistant message
                    self.session_manager.save_message(
                        self.current_session_id,
                        full_response,
                        role='assistant'
                    )
                except Exception as e:
                    logger.warning(f"Error saving session messages: {e}")
            
            # Return response with execution metadata
            return {
                "request_id": request_id,
                "model_id": model_identifier,
                "response": full_response,
                "metrics": execution_result.get('metrics', {}),
                "agent_checks": execution_result.get('agent_checks', []),
                "routing_decision": execution_result.get('routing_decision', {})
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "request_id": request_id,
                "error": str(e)
            }
            
    async def chat_streaming(self, session_id: Optional[str] = None) -> str:
        """
        Start an interactive chat session with streaming responses.
        
        Begins an interactive chat session where user inputs get streaming responses 
        directly to the terminal, with conversation context maintained.
        
        Args:
            session_id (Optional[str]): Optional existing session ID to resume
        
        Returns:
            str: Session ID for the current chat
                
        Notes:
            - Use 'exit' to quit the session
            - Use '/list' to see recent sessions
            - Use '/load <session_id>' to load a previous session
        """
        # Validate all requirements are met
        await self._validate_requirements()
        
        # Create or load session
        if session_id:
            # Try to load existing session
            try:
                session_data = self.session_manager.load_session(session_id)
                if not session_data:
                    logger.info(f"Session {session_id} not found. Starting a new session.")
                    session_id = None
            except FileNotFoundError as e:
                logger.warning(f"Session file for {session_id} not found: {e}")
                session_id = None
            except Exception as e:
                logger.error(f"Error loading session {session_id}: {e}")
                session_id = None
        
        # Create new session if needed
        if not session_id:
            session_id = self.session_manager.create_session(
                title=f"Interactive Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        
        # Set current session
        self.current_session_id = session_id
        
        # Don't preselect a model, let the orchestration decide based on agent feedback
        
        print(f"Oblix Streaming Chat Session Started (Session ID: {session_id})")
        print("Type 'exit' to quit")
        print("Type '/list' to see recent sessions")
        print("Type '/load <session_id>' to load a previous session")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Handle special commands
                if user_input.lower() == 'exit':
                    print("Chat session ended.")
                    break
                elif user_input.lower() == '/list':
                    # List recent sessions
                    sessions = self.session_manager.list_sessions()
                    print("\nRecent Sessions:")
                    for session in sessions:
                        print(f"ID: {session['id']} | Title: {session['title']} | Updated: {session['updated_at']}")
                    continue
                elif user_input.startswith('/load '):
                    # Load specific session
                    load_id = user_input.split(' ', 1)[1].strip()
                    session_data = self.session_manager.load_session(load_id)
                    if session_data:
                        self.current_session_id = load_id
                        print(f"\nLoaded session: {load_id}")
                        
                        # Display last few messages for context
                        if 'messages' in session_data and session_data['messages']:
                            print("\nRecent messages:")
                            for msg in session_data['messages'][-3:]:
                                role = msg.get('role', '').capitalize()
                                content = msg.get('content', '')
                                print(f"{role}: {content}\n")
                    else:
                        print(f"Could not load session: {load_id}")
                    continue
                
                # Execute prompt with streaming, using resource-based orchestration
                await self.execute_streaming(user_input)
                
            except KeyboardInterrupt:
                print("\nChat session ended.")
                break
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                logger.error(f"Chat error: {e}")
        
        return self.current_session_id
    
    def _select_model_from_agent_checks(self, agent_checks: Dict[str, Any]) -> Optional[Any]:
        """
        Select the best model based on agent check results.
        
        Analyzes agent check results to determine the optimal model for execution
        based on resource availability and connectivity status.
        
        Args:
            agent_checks (Dict[str, Any]): Results from agent checks
        
        Returns:
            Optional[Any]: Selected model instance or None
        """
        # Default to None
        selected_model = None
        
        try:
            # Extract resource check results if available
            resource_target = None
            for agent_name, check_result in agent_checks.items():
                if 'resource_monitor' in agent_name.lower():
                    resource_target = check_result.get('target')
                    break
            
            # Extract connectivity check results if available
            connectivity_target = None
            for agent_name, check_result in agent_checks.items():
                if 'connectivity' in agent_name.lower():
                    connectivity_target = check_result.get('target')
                    break
            
            # Log the agent recommendations
            logger.debug(f"Agent recommendations - Connectivity: {connectivity_target}, Resources: {resource_target}")
            
            # Handle conflicting recommendations
            target_type = None
            
            # If connectivity check suggests cloud but resources are constrained
            if connectivity_target == 'cloud' and resource_target == 'local':
                # Check if we need to override based on severity
                severity = None
                for agent_name, check_result in agent_checks.items():
                    if 'resource_monitor' in agent_name.lower():
                        severity = check_result.get('severity', 'medium')
                        break
                
                # If resource constraint is severe, prioritize it over connectivity
                if severity == 'high':
                    logger.info("Resource constraints (high severity) override connectivity recommendation")
                    target_type = ModelType.OLLAMA
                else:
                    # Otherwise follow connectivity recommendation
                    target_type = ModelType.OPENAI
            # Normal prioritization when no conflict
            elif connectivity_target == 'local':
                target_type = ModelType.OLLAMA
            elif connectivity_target == 'cloud':
                target_type = ModelType.OPENAI
            elif resource_target == 'local':
                target_type = ModelType.OLLAMA
            elif resource_target == 'cloud':
                target_type = ModelType.OPENAI
            
            # Select first model of target type if available
            if target_type:
                for model in self.models.values():
                    if model.model_type == target_type:
                        selected_model = model
                        break
                
                logger.info(f"Selected model type: {target_type.value if target_type else 'none'}")
            else:
                logger.info("No specific model type recommended by agents")
        
        except Exception as e:
            logger.warning(f"Error selecting model from agent checks: {e}")
        
        return selected_model

    async def create_session(self, 
                           title: Optional[str] = None, 
                           initial_context: Optional[Dict] = None) -> str:
        """
        Create a new chat session with optional title and initial context.
        
        Args:
            title (Optional[str]): Optional session title
            initial_context (Optional[Dict]): Optional initial context dictionary
        
        Returns:
            str: New session ID
        """
        session_id = self.session_manager.create_session(
            title=title,
            initial_context=initial_context
        )
        return session_id

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent chat sessions with metadata.
        
        Args:
            limit (int): Maximum number of sessions to return
        
        Returns:
            List[Dict[str, Any]]: List of session metadata dictionaries containing:
                - id: Unique session identifier
                - title: Session title
                - created_at: Creation timestamp
                - updated_at: Last update timestamp
                - message_count: Number of messages in the session
        """
        return self.session_manager.list_sessions(limit)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific chat session by ID.
        
        Args:
            session_id (str): Session identifier
        
        Returns:
            Optional[Dict[str, Any]]: Session data if found, None otherwise
        """
        return self.session_manager.load_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session permanently.
        
        Args:
            session_id (str): Session identifier
        
        Returns:
            bool: True if session was deleted successfully
        """
        return self.session_manager.delete_session(session_id)

    async def start_chat(self, session_id: Optional[str] = None) -> str:
        """
        Start an interactive chat session in the terminal.
        
        Begins an interactive chat session where the user can input prompts
        and receive responses, with conversation context maintained. The
        session will use intelligent model routing based on agent recommendations.
        
        Args:
            session_id (Optional[str]): Optional existing session ID to resume
        
        Returns:
            str: Session ID for the current chat
            
        Notes:
            - Use 'exit' to quit the session
            - Use '/list' to see recent sessions
            - Use '/load <session_id>' to load a previous session
        """
        # Validate all requirements are met
        await self._validate_requirements()
        
        # Create or load session
        if session_id:
            # Try to load existing session
            try:
                session_data = self.session_manager.load_session(session_id)
                if not session_data:
                    logger.info(f"Session {session_id} not found. Starting a new session.")
                    session_id = None
            except FileNotFoundError as e:
                logger.warning(f"Session file for {session_id} not found: {e}")
                session_id = None
            except Exception as e:
                logger.error(f"Error loading session {session_id}: {e}")
                session_id = None
        
        # Create new session if needed
        if not session_id:
            session_id = self.session_manager.create_session(
                title=f"Interactive Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        
        # Set current session
        self.current_session_id = session_id
        
        print(f"Oblix Chat Session Started (Session ID: {session_id})")
        print("Type 'exit' to quit")
        print("Type '/list' to see recent sessions")
        print("Type '/load <session_id>' to load a previous session")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Handle special commands
                if user_input.lower() == 'exit':
                    print("Chat session ended.")
                    break
                elif user_input.lower() == '/list':
                    # List recent sessions
                    sessions = self.session_manager.list_sessions()
                    print("\nRecent Sessions:")
                    for session in sessions:
                        print(f"ID: {session['id']} | Title: {session['title']} | Updated: {session['updated_at']}")
                    continue
                elif user_input.startswith('/load '):
                    # Load specific session
                    load_id = user_input.split(' ', 1)[1].strip()
                    session_data = self.session_manager.load_session(load_id)
                    if session_data:
                        self.current_session_id = load_id
                        print(f"\nLoaded session: {load_id}")
                        
                        # Display last few messages for context
                        if 'messages' in session_data and session_data['messages']:
                            print("\nRecent messages:")
                            for msg in session_data['messages'][-3:]:
                                role = msg.get('role', '').capitalize()
                                content = msg.get('content', '')
                                print(f"{role}: {content}\n")
                    else:
                        print(f"Could not load session: {load_id}")
                    continue
                
                # Execute prompt
                print("Thinking...")
                result = await self.execute(user_input)
                
                if result:
                    if "error" in result:
                        print(f"\nError: {result['error']}")
                    else:
                        print(f"\nAssistant: {result['response']}")
                else:
                    print("\nSorry, I couldn't generate a response.")
                
            except KeyboardInterrupt:
                print("\nChat session ended.")
                break
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                logger.error(f"Chat error: {e}")
        
        return self.current_session_id

    def list_models(self) -> Dict[str, List[str]]:
        """
        List all available models grouped by type.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping model types to lists of model names
        """
        return self.config_manager.list_models()
    
    async def get_resource_metrics(self) -> Dict[str, Any]:
        """
        Get current resource metrics from the resource monitor agent.
        
        Returns:
            Dict[str, Any]: Dictionary of resource metrics or None if unavailable
        """
        try:
            # Find the resource monitor agent
            resource_agent = next(
                (agent for agent in self.agents.values() 
                if isinstance(agent, ResourceMonitor)), 
                None
            )
            
            if resource_agent:
                metrics = await resource_agent.check()
                return metrics
            
            logger.warning("No resource monitoring agent found")
            return None
        
        except Exception as e:
            logger.error(f"Error getting resource metrics: {e}")
            return None

    def get_model(self, model_type: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.
        
        Args:
            model_type (str): Type of model (e.g., 'ollama', 'openai', 'claude')
            model_name (str): Name of model
        
        Returns:
            Optional[Dict[str, Any]]: Model configuration if found
        """
        return self.config_manager.get_model(model_type, model_name)
    
    
    
    async def chat_once(self, prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a single message in a chat session with automatic session handling.
        
        This is a convenience method that:
        1. Creates a new session or uses the provided one
        2. Sends the prompt and gets a response
        3. Maintains the session context for future interactions
        
        This provides a simpler interface than execute() when building chat applications.
        
        Args:
            prompt (str): User message
            session_id (Optional[str]): Existing session ID or None to create new session
        
        Returns:
            Dict[str, Any]: Response containing:
                - response: Generated text
                - session_id: ID of the session used
                - metrics: Performance metrics
        """
        # Create or use provided session
        if not session_id:
            session_id = await self.create_session()
        
        # Set as current session
        self.current_session_id = session_id
        
        # Execute the prompt
        result = await self.execute(prompt)
        
        # Add session ID to result
        if result:
            result["session_id"] = session_id
        
        return result