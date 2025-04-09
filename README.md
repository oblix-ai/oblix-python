# Oblix: AI Orchestration SDK

Oblix is an AI orchestration SDK that intelligently routes between local and cloud models based on system conditions, providing a unified interface regardless of the underlying model provider.

For complete documentation, please visit: [documentation.oblix.ai](https://documentation.oblix.ai)

## Features

- **Unified Model Interface**: Consistent API across different model providers (OpenAI, Claude, Ollama)
- **Intelligent Orchestration**: Automatic switching between local and cloud models based on system resources and connectivity
- **Session Management**: Create, load, export, and import conversation sessions
- **Streaming Support**: Real-time token streaming
- **Extensible Agent System**: Custom monitoring and decision-making policies
- **OpenAI-Compatible API**: Drop-in compatibility with applications built for OpenAI's API

## Getting Started

### Installation

```bash
pip install oblix
```

### Basic Usage

```python
from oblix import OblixClient, ModelType

# Initialize client
client = OblixClient()

# Connect to models
await client.hook_model(ModelType.OLLAMA, "llama2", endpoint="http://localhost:11434")
await client.hook_model(ModelType.OPENAI, "gpt-3.5-turbo", api_key="your_openai_key")

# Add monitoring agents
from oblix.agents.resource_monitor import ResourceMonitor
from oblix.agents.connectivity import ConnectivityAgent
client.hook_agent(ResourceMonitor())
client.hook_agent(ConnectivityAgent())

# Execute prompts (with automatic model selection)
response = await client.execute("Your prompt here")

# Stream responses
await client.execute_streaming("Your prompt here")

# Manage sessions
session_id = await client.create_session("Chat Title")
```

## OpenAI-Compatible API

Oblix provides an OpenAI-compatible endpoint that allows using the standard OpenAI Python client with Oblix's orchestration capabilities:

### Starting the Server

```bash
# Start the Oblix server
uvicorn oblix.main:app --host 0.0.0.0 --port 8140
```

### Client Usage

```python
from openai import OpenAI

# Point to local Oblix server
client = OpenAI(
    base_url="http://localhost:8140/v1",
    api_key="any-value"  # Authentication not required
)

# Use just like the OpenAI client
completion = client.chat.completions.create(
    model="ollama:llama2",  # Use format "provider:model" or just "model_name"
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello world!"}
    ],
    temperature=0.7,
)

# Access response
print(completion.choices[0].message.content)

# Streaming also works
stream = client.chat.completions.create(
    model="openai:gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Orchestration Features

The OpenAI-compatible endpoint supports:

- Specifying models in "provider:name" format (e.g., "openai:gpt-3.5-turbo", "ollama:llama2")
- Using a generic model identifier to let Oblix automatically select the best model
- Streaming responses
- Full context handling for conversations

## Examples

Check the `examples/` directory for more usage examples:

- `demo.py`: Basic usage of the Oblix client
- `connectivity_demo.py`: Demonstrates automatic switching based on network conditions
- `session_management_demo.py`: Shows session creation and management
- `openai_compatibility_demo.py`: Shows how to use the OpenAI client with Oblix
