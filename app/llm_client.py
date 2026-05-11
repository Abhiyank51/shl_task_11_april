import concurrent.futures
import os

from openai import OpenAI

from app.prompts import FALLBACKS, SYSTEM_PROMPT


class LLMClient:
    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN", "").strip()
        self.model = os.environ.get(
            "HF_MODEL",
            "meta-llama/Llama-3.1-8B-Instruct",
        ).strip()

        self.client = None

        if self.hf_token:
            try:
                self.client = OpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=self.hf_token,
                    timeout=12,
                )
            except Exception as exc:
                print(f"Failed to initialize Hugging Face LLM client: {exc}")
                self.client = None

    def _fallback(self, intent: str) -> str:
        return FALLBACKS.get(intent, FALLBACKS["recommend"])

    def generate_reply(self, prompt: str, intent: str) -> str:
        if not self.client:
            return self._fallback(intent)

        def _call_llm() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )

            if not response or not response.choices:
                return self._fallback(intent)

            content = response.choices[0].message.content

            if not content or not content.strip():
                return self._fallback(intent)

            return content.strip()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_call_llm)

        try:
            return future.result(timeout=10)

        except concurrent.futures.TimeoutError:
            print("LLM generation timed out after 10s.")
            future.cancel()
            return self._fallback(intent)

        except Exception as exc:
            print(f"LLM generation failed: {exc}")
            return self._fallback(intent)

        finally:
            executor.shutdown(wait=False, cancel_futures=True)


# Backward-compatible alias so graph.py does not need to change.
GeminiClient = LLMClient