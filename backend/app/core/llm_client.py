"""Multi-backend LLM client supporting Groq, OpenAI, Ollama, and HuggingFace."""

import time
from typing import Optional, Tuple
import httpx
import torch

from app.config import get_settings


class LLMClient:
    """Unified client for invoking various LLM backends with latency measurement."""

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.backend = (backend or settings.DEFAULT_LLM_BACKEND).lower()

        if self.backend == "groq":
            self.model = model or settings.DEFAULT_GROQ_MODEL
            self.api_key = api_key or settings.GROQ_API_KEY
            self.base_url = base_url or ""
        elif self.backend == "openai":
            self.model = model or "gpt-3.5-turbo"
            self.api_key = api_key or settings.OPENAI_API_KEY
            self.base_url = base_url or ""
        elif self.backend == "ollama":
            self.model = model or "llama3"
            self.api_key = api_key or ""
            self.base_url = base_url or settings.OLLAMA_BASE_URL
        elif self.backend == "huggingface":
            self.model = model or settings.HF_MODEL_ID
            self.api_key = api_key or ""
            self.base_url = base_url or ""
            self._hf_pipeline = None
        else:
            raise ValueError(f"Unsupported LLM backend: '{self.backend}'. Supported: groq, openai, ollama, huggingface")

    def _generate_groq(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Execute text generation via Groq Cloud API."""
        if not self.api_key:
            raise ValueError(
                "Groq API key is missing. Please provide GROQ_API_KEY in your environment or .env file."
            )
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"Groq API error ({self.model}): {str(e)}") from e

    def _generate_openai(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Execute text generation via OpenAI or compatible endpoints."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. Please provide OPENAI_API_KEY in your environment or .env file."
            )
        try:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            client = OpenAI(**kwargs)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI API error ({self.model}): {str(e)}") from e

    def _generate_ollama(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Execute text generation via Ollama local REST API."""
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at '{self.base_url}'. Ensure Ollama service is running."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}") from e

    def _generate_huggingface(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """Execute text generation via local HuggingFace transformers pipeline."""
        try:
            from transformers import pipeline

            if getattr(self, "_hf_pipeline", None) is None:
                device = 0 if torch.cuda.is_available() else -1
                self._hf_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    device=device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )

            full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:" if system_prompt else prompt
            outputs = self._hf_pipeline(
                full_prompt,
                max_new_tokens=max_tokens,
                temperature=max(0.01, temperature),
                do_sample=temperature > 0.0,
                pad_token_id=self._hf_pipeline.tokenizer.eos_token_id,
            )
            raw_text = outputs[0]["generated_text"]
            if raw_text.startswith(full_prompt):
                return raw_text[len(full_prompt):].strip()
            return raw_text.strip()
        except Exception as e:
            raise RuntimeError(f"HuggingFace local pipeline error ({self.model}): {str(e)}") from e

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Tuple[str, float]:
        """
        Generate completion text and record elapsed wall-clock latency in milliseconds.
        Returns: (answer_string, latency_ms)
        """
        start_time = time.perf_counter()

        if self.backend == "groq":
            answer = self._generate_groq(prompt, system_prompt, max_tokens, temperature)
        elif self.backend == "openai":
            answer = self._generate_openai(prompt, system_prompt, max_tokens, temperature)
        elif self.backend == "ollama":
            answer = self._generate_ollama(prompt, system_prompt, max_tokens, temperature)
        elif self.backend == "huggingface":
            answer = self._generate_huggingface(prompt, system_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return answer.strip(), round(elapsed_ms, 2)


def get_llm_client(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """Instantiate and return an LLMClient for requested provider configuration."""
    return LLMClient(backend=backend, model=model, api_key=api_key, base_url=base_url)
