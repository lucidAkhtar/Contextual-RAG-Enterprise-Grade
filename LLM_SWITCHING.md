# LLM Endpoint Switching - Quick Reference

## How to Switch LLMs

### Option 1: Configuration File (Easiest)
**File**: `config/settings.py` (lines 25-29)

```python
llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"  # ← Change this
llm_model: str = "mistral:latest"                                   # ← Or this
llm_base_url: str = "http://localhost:11434"                       # ← Or this
```

**Examples**:
- Ollama Mistral: `llm_provider = "ollama"`, `llm_model = "mistral:latest"`
- Ollama Llama2: `llm_provider = "ollama"`, `llm_model = "llama2:latest"`
- OpenAI GPT-4: `llm_provider = "openai"`, `llm_model = "gpt-4"`

### Option 2: Environment Variables
Create `.env` file:
```bash
LLM_PROVIDER=ollama
LLM_MODEL=mistral:latest
LLM_BASE_URL=http://localhost:11434
```

## Implementation Details

### 1. Factory Pattern Implementation
**File**: `src/core/llm_factory.py`

- **Lines 16-27**: Abstract `LLMProvider` interface
- **Lines 29-65**: `OllamaProvider` implementation
- **Lines 68-104**: `OpenAIProvider` implementation  
- **Lines 186-197**: `get_llm()` method - auto-creates correct provider

```python
class LLMFactory:
    _providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
    }
    
    @classmethod
    def get_llm(cls) -> LLM:
        # Reads config and creates right provider automatically
```

### 2. Usage in System
**File**: `src/core/query_engine.py` (line ~160)
```python
self.llm = llm or LLMFactory.get_llm()  # Auto-configures based on settings
```

**File**: `src/core/retrievers.py` (line ~71)
```python
llm = LLMFactory.get_llm()  # Used for contextual enrichment
```

## Key Benefits
- **Zero code changes** - just edit config
- **Factory pattern** - clean abstraction
- **Multiple providers** - Ollama, OpenAI (extensible)
- **Same interface** - all providers use `LLMProvider` base class
- **SOLID principles** - Open/Closed (open for extension)

## To Apply Changes
Restart the API server:
```bash
python src/main.py
```
