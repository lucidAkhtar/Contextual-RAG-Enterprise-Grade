"""
LLM Factory implementing Strategy and Factory patterns.
Allows easy switching between LLM providers (Ollama, OpenAI, Anthropic, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_llm(self) -> LLM:
        """Get configured LLM instance."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if LLM provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        logger.info(f"Initialized OllamaProvider with model: {model}")
    
    def get_llm(self) -> LLM:
        """Get configured Ollama LLM instance."""
        return Ollama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            request_timeout=120.0,
            **self.kwargs
        )
    
    def health_check(self) -> bool:
        """Check Ollama availability."""
        try:
            llm = self.get_llm()
            response = llm.complete("test")
            return response is not None
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation (extensible for future)."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        logger.info(f"Initialized OpenAIProvider with model: {model}")
    
    def get_llm(self) -> LLM:
        """Get configured OpenAI LLM instance."""
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.kwargs
        )
    
    def health_check(self) -> bool:
        """Check OpenAI availability."""
        try:
            llm = self.get_llm()
            response = llm.complete("test")
            return response is not None
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


class LLMFactory:
    """
    Factory class for creating LLM providers.
    Implements Factory pattern for provider instantiation.
    """
    
    _providers: Dict[str, type] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a new LLM provider.
        
        Args:
            name: Provider identifier
            provider_class: Provider class implementing LLMProvider
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered new LLM provider: {name}")
    
    @classmethod
    def create_provider(
        cls,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """
        Create LLM provider based on configuration.
        
        Args:
            provider_name: Optional provider name override
            **kwargs: Additional provider-specific arguments
        
        Returns:
            Configured LLM provider instance
        
        Raises:
            ValueError: If provider is not supported
        """
        settings = get_settings()
        provider_name = provider_name or settings.llm_provider
        
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. "
                f"Available: {list(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider_name]
        
        # Prepare provider-specific configuration
        if provider_name == "ollama":
            config = {
                "model": kwargs.get("model", settings.llm_model),
                "base_url": kwargs.get("base_url", settings.llm_base_url),
                "temperature": kwargs.get("temperature", settings.llm_temperature),
                "max_tokens": kwargs.get("max_tokens", settings.llm_max_tokens),
            }
        elif provider_name == "openai":
            config = {
                "model": kwargs.get("model", settings.llm_model),
                "api_key": kwargs.get("api_key"),
                "temperature": kwargs.get("temperature", settings.llm_temperature),
                "max_tokens": kwargs.get("max_tokens", settings.llm_max_tokens),
            }
        else:
            config = kwargs
        
        logger.info(f"Creating LLM provider: {provider_name}")
        return provider_class(**config)
    
    @classmethod
    def get_llm(cls, provider_name: Optional[str] = None, **kwargs) -> LLM:
        """
        Convenience method to get LLM instance directly.
        
        Args:
            provider_name: Optional provider name override
            **kwargs: Additional provider-specific arguments
        
        Returns:
            Configured LLM instance
        """
        provider = cls.create_provider(provider_name, **kwargs)
        return provider.get_llm()
