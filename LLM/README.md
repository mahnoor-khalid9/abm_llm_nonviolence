# LLM Module

## File structure

```
llm/
├── schemas.py                  # All Pydantic output models (shared, no ML deps)
├── base.py                     # Abstract BaseProvider + shared utilities
├── chat_model.py               # Public factory — the ONLY file simulations import
└── providers/
    ├── __init__.py
    ├── gemini.py               # Google Gemini (closed, API)
    ├── openai_provider.py      # OpenAI (closed, API)
    ├── ollama.py               # Ollama (local REST server)
    └── huggingface.py          # HuggingFace local inference (GPU / Unity cluster)
```

## How to use

```python
# Option A — factory function (recommended)
from chat_model import build_provider

provider = build_provider("gemini", temperature=0.0)
output   = provider.ask_activist_position(context, memory, use_memory=True)

# Option B — ChatModel wrapper (drop-in replacement for old code)
from chat_model import ChatModel

model  = ChatModel(provider="gemini", use_memory=True)
output = model.ask_activist_position(context, memory)
```

## Adding a new provider

1. Create `providers/my_provider.py` with a class that inherits `BaseProvider`.
2. Implement `_call_api(self, prompt, response_model, **kwargs) -> str`.
3. Add an `elif provider == "myprovider":` branch in `chat_model.build_provider()`.

That's it — the shared prompt rendering, position validation, and
`ask_activist_position()` are all inherited for free.

## Dependency isolation

| File                       | Heavy deps loaded?          |
|----------------------------|-----------------------------|
| `schemas.py`               | None (pydantic only)        |
| `base.py`                  | None (jinja2 only)          |
| `chat_model.py`            | None at import time         |
| `providers/gemini.py`      | google-genai (only if used) |
| `providers/openai_provider.py` | openai (only if used)   |
| `providers/ollama.py`      | requests (only if used)     |
| `providers/huggingface.py` | torch + transformers — **only if used** |

HuggingFace is imported lazily, so running the simulation with Gemini
will never load torch or transformers.
