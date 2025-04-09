# oblix/client/client.py
from typing import Dict, Any, Optional, List
import logging
import os
import asyncio
import uuid
import json
from datetime import datetime
import sys
import gc

from .base_client import OblixBaseClient
from ..models.base import ModelType
from ..agents.base import BaseAgent
from ..agents.resource_monitor import ResourceMonitor
from ..agents.connectivity import ConnectivityAgent
from ..agents.connectivity.policies import ConnectivityState, ConnectionTarget

logger = logging.getLogger(__name__)

# Helper function for user-friendly error printing
def print_error(message):
    """Print error message in red if available, otherwise standard print"""
    try:
        # Try to use colorama if available
        from colorama import Fore, Style
        print(f"{Fore.RED}{Style.BRIGHT}{message}{Style.RESET_ALL}")
    except ImportError:
        # Fallback to standard print
        print(f"ERROR: {message}")

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
        client = OblixClient()
        
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
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
            
        try:
            # Validate all requirements are met
            await self._validate_requirements()
        except Exception as e:
            # Handle validation errors
            logger.error(f"Validation error in execute: {e}")
            return {
                "request_id": request_id,
                "error": str(e)
            }
        
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
            
            # No model selection - rely entirely on policy-based orchestration in ExecutionManager
            model_type = None
            model_name = None
            
            logger.info("Executing with pure policy-based orchestration")
            
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
                    
                    # Save assistant message - ensure it's properly formatted
                    if isinstance(response, str):
                        message_to_save = response
                    elif isinstance(response, dict):
                        # Don't wrap an existing dict in another dict
                        message_to_save = response
                    else:
                        # For other types, convert to string
                        message_to_save = str(response)
                        
                    self.session_manager.save_message(
                        self.current_session_id,
                        message_to_save,
                        role='assistant'
                    )
                except Exception as e:
                    logger.warning(f"Error saving session messages: {e}")
            
            # Calculate metrics 
            metrics = {}
            
            # For non-streaming responses, metrics are in the execution_result
            metrics = execution_result.get('metrics', {})
            
            # Ensure response is a string
            if isinstance(response, dict):
                response_str = str(response)
            else:
                response_str = str(response)
                
            # Return response with execution metadata
            return {
                "request_id": request_id,
                "model_id": execution_result.get('model_id', ''),
                "response": response_str,  # Ensure response is a string
                "metrics": metrics,
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
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          request_id: Optional[str] = None,
                          display_metrics: bool = True,
                          **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute a prompt with streaming output directly to the terminal.
        
        This method is similar to execute() but streams the response token-by-token
        as it's generated, rather than waiting for the complete response. 
        
        Args:
            prompt (str): User prompt to process
            temperature (Optional[float]): Sampling temperature for text generation
            max_tokens (Optional[int]): Maximum tokens to generate
            request_id (Optional[str]): Custom request identifier for tracking
            display_metrics (bool): Whether to display performance metrics
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
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
            
        try:
            # Validate all requirements are met
            await self._validate_requirements()
        except Exception as e:
            # Handle validation errors
            logger.error(f"Validation error in execute_streaming: {e}")
            return {
                "request_id": request_id,
                "error": str(e)
            }
        
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
            
            # No model selection or redundant connectivity checks here - relying entirely on the ExecutionManager's orchestration
            # which will check connectivity and resource policies automatically
            model_type = None
            model_name = None
            
            logger.info("Executing streaming with pure policy-based orchestration")
                        
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
            
            # Get the response text from execution result (may already be complete for non-streaming models)
            if 'response' in execution_result:
                print(f"\nA: {execution_result['response']}")
                full_response = execution_result['response']
            else:
                # Otherwise we expect streaming data from the execution
                print("\nA: ", end="", flush=True)
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
                
                # Calculate metrics regardless of whether we display them
                import json
                
                # Retrieve metrics from model instance after streaming is complete
                metrics = {}
                
                # For streaming responses, metrics are stored in the model's last_metrics property
                if 'model_instance' in execution_result:
                    model_instance = execution_result.get('model_instance')
                    if hasattr(model_instance, 'last_metrics') and model_instance.last_metrics:
                        metrics = model_instance.last_metrics
                else:
                    # Fallback to any metrics directly in the execution result
                    metrics = execution_result.get("metrics", {})
                
                # Format metrics using helper function
                enhanced_metrics = self.format_metrics(metrics)
                
                # Extract model type and name from model_id
                model_type = model_identifier.split(":")[0] if ":" in model_identifier else ""
                model_name = model_identifier.split(":")[1] if ":" in model_identifier else model_identifier
                
                # Create a response object with the required fields
                response_json = {
                    "response": full_response,
                    "model_name": model_name,
                    "model_type": model_type,
                    "time_to_first_token": enhanced_metrics.get("time_to_first_token"),
                    "tokens_per_second": enhanced_metrics.get("tokens_per_second"),
                    "latency": enhanced_metrics.get("total_latency"),
                    "input_tokens": enhanced_metrics.get("input_tokens"),
                    "output_tokens": enhanced_metrics.get("output_tokens")
                }
                
                # Remove null values
                response_json = {k: v for k, v in response_json.items() if v is not None}
                
                # Only display metrics if requested (controlled by display_metrics parameter)
                if display_metrics:
                    print(f"\nJSON response:\n{json.dumps(response_json, indent=2)}")
                    
                    # Display routing decisions after the JSON
                    if execution_result.get('routing_decision'):
                        resource_routing = execution_result['routing_decision'].get('resource_routing')
                        connectivity_routing = execution_result['routing_decision'].get('connectivity_routing')
                        
                        print("\nRouting decisions:")
                        if resource_routing:
                            print(f"Resource: {{\n  'target': '{resource_routing.get('target')}',\n  'state': '{resource_routing.get('state')}',\n  'reason': '{resource_routing.get('reason')}'\n}}")
                        if connectivity_routing:
                            print(f"Connectivity: {{\n  'state': '{connectivity_routing.get('state')}',\n  'target': '{connectivity_routing.get('target')}',\n  'reason': '{connectivity_routing.get('reason')}'\n}}")
            
            # Save to session if active
            if self.current_session_id:
                try:
                    # Save user message
                    self.session_manager.save_message(
                        self.current_session_id,
                        prompt,
                        role='user'
                    )
                    
                    # Save assistant message - ensure it's properly formatted
                    if isinstance(full_response, str):
                        message_to_save = full_response
                    elif isinstance(full_response, dict):
                        # Don't wrap an existing dict in another dict
                        message_to_save = full_response
                    else:
                        # For other types, convert to string
                        message_to_save = str(full_response)
                        
                    self.session_manager.save_message(
                        self.current_session_id,
                        message_to_save,
                        role='assistant'
                    )
                except Exception as e:
                    logger.warning(f"Error saving session messages: {e}")
            
            # Ensure metrics are also included in the return value
            # Use the same metrics retrieval logic as above
            metrics = {}
            if 'model_instance' in execution_result:
                model_instance = execution_result.get('model_instance')
                if hasattr(model_instance, 'last_metrics') and model_instance.last_metrics:
                    metrics = model_instance.last_metrics
            else:
                metrics = execution_result.get('metrics', {})
                
            # Return response with execution metadata
            return {
                "request_id": request_id,
                "model_id": model_identifier,
                "response": full_response,
                "metrics": metrics,
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
        try:
            # Validate all requirements are met
            await self._validate_requirements()
        except Exception as e:
            # Handle validation errors
            print_error(f"Error starting chat: {e}")
            return None
        
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
    
    # The _select_model_from_agent_checks method has been removed
    # All model selection logic is now handled exclusively in the ExecutionManager

    async def create_session(self, 
                           title: Optional[str] = None, 
                           initial_context: Optional[Dict] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new chat session with optional title and initial context.
        
        Args:
            title (Optional[str]): Optional session title
            initial_context (Optional[Dict]): Optional initial context dictionary
            metadata (Optional[Dict[str, Any]]): Optional additional metadata
        
        Returns:
            str: New session ID
        """
        session_id = self.session_manager.create_session(
            title=title,
            initial_context=initial_context,
            metadata=metadata
        )
        return session_id
        
    async def create_and_use_session(self, 
                           title: Optional[str] = None, 
                           initial_context: Optional[Dict] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new chat session and automatically set it as the current session.
        
        This convenience method creates a new session and sets it as the
        current session, making it immediately available for conversation.
        
        Args:
            title (Optional[str]): Optional session title
            initial_context (Optional[Dict]): Optional initial context dictionary
            metadata (Optional[Dict[str, Any]]): Optional additional metadata
        
        Returns:
            str: New session ID (already set as current_session_id)
        """
        session_id = await self.create_session(
            title=title,
            initial_context=initial_context,
            metadata=metadata
        )
        self.current_session_id = session_id
        logger.info(f"Created and activated session: {session_id}")
        return session_id

    def use_session(self, session_id: str) -> bool:
        """
        Set an existing session as the current active session.
        
        Validates that the session exists and sets it as the active session
        for future conversation interactions.
        
        Args:
            session_id (str): Session identifier to activate
            
        Returns:
            bool: True if session was successfully activated, False if not found
        """
        session_data = self.session_manager.load_session(session_id)
        if not session_data:
            logger.warning(f"Cannot activate session {session_id}: not found")
            return False
            
        self.current_session_id = session_id
        logger.info(f"Activated session: {session_id}")
        return True

    def list_sessions(self, limit: int = 50, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List recent chat sessions with metadata and optional filtering.
        
        Args:
            limit (int): Maximum number of sessions to return
            filter_metadata (Optional[Dict[str, Any]]): Optional metadata filters
                to only return sessions matching specific criteria
        
        Returns:
            List[Dict[str, Any]]: List of session metadata dictionaries containing:
                - id: Unique session identifier
                - title: Session title
                - created_at: Creation timestamp
                - updated_at: Last update timestamp
                - message_count: Number of messages in the session
                - metadata: Additional session metadata
        """
        return self.session_manager.list_sessions(limit, filter_metadata)

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
        
    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update or add metadata to a session.
        
        Updates existing metadata fields or adds new fields without
        affecting other session data.
        
        Args:
            session_id (str): Session identifier
            metadata (Dict[str, Any]): Metadata fields to update or add
            
        Returns:
            bool: True if metadata was updated successfully
        """
        return self.session_manager.update_session_metadata(session_id, metadata)
        
    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific session.
        
        Retrieves just the metadata fields for a session without
        loading the entire conversation.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            Optional[Dict[str, Any]]: Session metadata if found
        """
        return self.session_manager.get_session_metadata(session_id)
        
    async def export_session(self, session_id: str, export_path: str) -> bool:
        """
        Export a session to a file.
        
        Exports a complete session to a JSON file that can be shared
        or backed up.
        
        Args:
            session_id (str): Session identifier
            export_path (str): Path to save the exported session
            
        Returns:
            bool: True if export was successful
        """
        return self.session_manager.export_session(session_id, export_path)
        
    async def import_session(self, import_path: str, new_id: bool = True, use_immediately: bool = False) -> Optional[str]:
        """
        Import a session from a file.
        
        Imports a session from a JSON file, optionally assigning a new ID
        to avoid conflicts with existing sessions.
        
        Args:
            import_path (str): Path to the JSON file to import
            new_id (bool): Whether to assign a new ID (True) or keep original ID (False)
            use_immediately (bool): Whether to set the imported session as the current session
            
        Returns:
            Optional[str]: Session ID of the imported session, or None if import failed
        """
        session_id = self.session_manager.import_session(import_path, new_id)
        if session_id and use_immediately:
            self.current_session_id = session_id
            logger.info(f"Imported and activated session: {session_id}")
        return session_id
        
    async def merge_sessions(self, source_ids: List[str], title: Optional[str] = None, use_immediately: bool = False) -> Optional[str]:
        """
        Merge multiple sessions into a new session.
        
        Creates a new session containing all messages from the source sessions,
        properly ordered by timestamp.
        
        Args:
            source_ids (List[str]): List of session IDs to merge
            title (Optional[str]): Optional title for the merged session
            use_immediately (bool): Whether to set the merged session as the current session
            
        Returns:
            Optional[str]: ID of the newly created merged session, or None if merge failed
        """
        session_id = self.session_manager.merge_sessions(source_ids, title)
        if session_id and use_immediately:
            self.current_session_id = session_id
            logger.info(f"Created and activated merged session: {session_id}")
        return session_id
        
    async def copy_session(self, session_id: str, new_title: Optional[str] = None, use_immediately: bool = False) -> Optional[str]:
        """
        Create a copy of an existing session.
        
        Creates a new session with the same content as an existing session
        but with a new ID.
        
        Args:
            session_id (str): Session identifier to copy
            new_title (Optional[str]): Optional new title for the copied session
            use_immediately (bool): Whether to set the copied session as the current session
            
        Returns:
            Optional[str]: ID of the new copy, or None if copy failed
        """
        new_session_id = self.session_manager.copy_session(session_id, new_title)
        if new_session_id and use_immediately:
            self.current_session_id = new_session_id
            logger.info(f"Copied and activated session: {new_session_id}")
        return new_session_id

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
        try:
            # Validate all requirements are met
            await self._validate_requirements()
        except Exception as e:
            # Handle validation errors
            print_error(f"Error starting chat: {e}")
            return None
        
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
                
                # Execute prompt with streaming but don't display metrics (we'll show them later)
                print("Thinking...")
                result = await self.execute_streaming(user_input, display_metrics=False)
                
                if result:
                    if "error" in result:
                        print(f"\nError: {result['error']}")
                    else:
                        # Don't print the response again - it was already printed by execute_streaming
                        # Skip directly to the JSON metrics
                        import json
                        
                        # Format metrics using helper function
                        enhanced_metrics = self.format_metrics(result.get("metrics", {}))
                        
                        model_id = result.get("model_id", "")
                        model_type = model_id.split(":")[0] if ":" in model_id else ""
                        model_name = model_id.split(":")[1] if ":" in model_id else model_id
                        
                        # Create a response object with the required fields
                        response_json = {
                            "response": result.get("response", ""),
                            "model_name": model_name,
                            "model_type": model_type,
                            "time_to_first_token": enhanced_metrics.get("time_to_first_token"),
                            "tokens_per_second": enhanced_metrics.get("tokens_per_second"),
                            "latency": enhanced_metrics.get("total_latency"),
                            "input_tokens": enhanced_metrics.get("input_tokens"),
                            "output_tokens": enhanced_metrics.get("output_tokens")
                        }
                        
                        # Remove null values
                        response_json = {k: v for k, v in response_json.items() if v is not None}
                            
                        print(f"\nJSON response:\n{json.dumps(response_json, indent=2)}")
                        
                        # Print routing decisions if available
                        if result.get('routing_decision'):
                            resource_routing = result['routing_decision'].get('resource_routing')
                            connectivity_routing = result['routing_decision'].get('connectivity_routing')
                            
                            print("\nRouting decisions:")
                            if resource_routing:
                                print(f"Resource: {{\n  'target': '{resource_routing.get('target')}',\n  'state': '{resource_routing.get('state')}',\n  'reason': '{resource_routing.get('reason')}'\n}}")
                            if connectivity_routing:
                                print(f"Connectivity: {{\n  'state': '{connectivity_routing.get('state')}',\n  'target': '{connectivity_routing.get('target')}',\n  'reason': '{connectivity_routing.get('reason')}'\n}}")
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
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Format response as pretty-printed JSON.
        
        Args:
            result (Dict[str, Any]): Raw result from execute() method
            
        Returns:
            str: Formatted response text
        """
        # Get the response content
        response_data = result.get("response", "")
        
        # Handle nested response objects
        if isinstance(response_data, dict) and "response" in response_data:
            # Extract the actual response from the nested structure
            response_text = response_data["response"]
        else:
            response_text = response_data
            
        # No longer displaying routing decisions here, moved to the calling functions
        
        return response_text
        
    def format_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics dictionary to ensure consistent output across streaming and non-streaming modes.
        
        Args:
            metrics (Dict[str, Any]): Raw metrics dictionary
            
        Returns:
            Dict[str, Any]: Enhanced metrics dictionary with all required fields
        """
        # Check if essential metrics are present
        has_metrics = (metrics and 
                      metrics.get("total_latency") is not None)
        
        if not has_metrics:
            # Return metrics with all fields set to None if no metrics are available yet
            return {
                "total_latency": None,
                "tokens_per_second": None,
                "start_time": None,
                "end_time": None,
                "input_tokens": None,
                "output_tokens": None,
                "model_name": None,
                "model_type": None,
                "time_to_first_token": None
            }
            
        # Create enhanced metrics dictionary with all requested fields
        enhanced_metrics = {
            "total_latency": metrics.get("total_latency"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "start_time": metrics.get("start_time"),
            "end_time": metrics.get("end_time"),
            "input_tokens": metrics.get("input_tokens"),
            "output_tokens": metrics.get("output_tokens"),
            "model_name": metrics.get("model_name"),
            "model_type": metrics.get("model_type"),
            "time_to_first_token": metrics.get("time_to_first_token")
        }
        
        return enhanced_metrics
    
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
        
    async def cleanup(self):
        """
        Clean up all resources used by the client.
        
        This method ensures proper cleanup of all async resources, 
        including HTTP sessions, connections, and any pending tasks.
        
        Call this method when shutting down the client to prevent 
        resource leaks and event loop warnings.
        """
        logger.info("Cleaning up client resources...")
        
        try:
            # Close all model connections if available
            for model_key, model in self.models.items():
                try:
                    if hasattr(model, 'close') and callable(model.close):
                        await model.close()
                    elif hasattr(model, 'cleanup') and callable(model.cleanup):
                        await model.cleanup()
                    logger.info(f"Closed model: {model_key}")
                except Exception as e:
                    logger.error(f"Error closing model {model_key}: {e}")
            
            # Close execution manager if available
            if hasattr(self, 'execution_manager'):
                try:
                    if hasattr(self.execution_manager, 'close') and callable(self.execution_manager.close):
                        await self.execution_manager.close()
                    logger.info("Closed execution manager")
                except Exception as e:
                    logger.error(f"Error closing execution manager: {e}")
            
            # Auth has been removed
            
            # Find and close any other aiohttp sessions - do this safely to avoid proxy object issues
            try:
                import inspect
                for obj in gc.get_objects():
                    # Only check objects that are actually aiohttp.ClientSession instances
                    # This avoids issues with proxy objects or other objects that might raise errors
                    if not inspect.isclass(obj) and obj.__class__.__name__ == 'ClientSession':
                        if hasattr(obj, 'closed') and not obj.closed:
                            try:
                                await obj.close()
                                # Silent success
                            except Exception:
                                # Silent failure
                                pass
            except Exception:
                # Silent failure
                pass
            
            # Cancel any pending tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                for task in tasks:
                    task.cancel()
                
                # Wait for tasks to be cancelled
                await asyncio.gather(*tasks, return_exceptions=True)
                
            # Silent completion
            
        except Exception:
            # Silent failure
            pass
            
    async def shutdown(self):
        """Alias for cleanup() - ensures compatibility with other frameworks"""
        await self.cleanup()