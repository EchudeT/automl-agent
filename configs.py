import os

class Configs:
    # API Keys - can be overridden by environment variables
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    HF_KEY = os.getenv("HF_KEY", "")
    PWC_KEY = os.getenv("PWC_KEY", "")
    SEARCHAPI_API_KEY = os.getenv("SEARCHAPI_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

AVAILABLE_LLMs = {
    # From apikey.md - Gemini via PoloAI proxy (corrected model name)
    "gemini": {
        "api_key": os.getenv("OPENAI_API_KEY", "key待填"),
        "model": os.getenv("MODEL_NAME", "gemini-3-flash-preview"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://poloai.top/v1"),
    },
    # Default model (points to gemini for compatibility)
    "qwen": {
        "api_key": os.getenv("OPENAI_API_KEY", "key待填"),
        "model": os.getenv("MODEL_NAME", "gemini-3-flash-preview"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://poloai.top/v1"),
    },
    # Legacy OpenAI configurations (can be activated via env vars)
    "gpt-4.1": {
        "api_key": os.getenv("OPENAI_API_KEY", Configs.OPENAI_KEY),
        "model": "gpt-4.1"
    },
    "gpt-4": {
        "api_key": os.getenv("OPENAI_API_KEY", Configs.OPENAI_KEY),
        "model": "gpt-4o"
    },
    "gpt-3.5": {
        "api_key": os.getenv("OPENAI_API_KEY", Configs.OPENAI_KEY),
        "model": "gpt-3.5-turbo"
    },
    # Local model placeholder
    "prompt-llm": {
        "api_key": "empty",
        "model": "prompt-llama",
        "base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    },
}

TASK_METRICS = {
    "image_classification": "accuracy",
    "text_classification": "accuracy",
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
    "node_classification": "accuracy",
    "ts_forecasting": "RMSLE",
}
