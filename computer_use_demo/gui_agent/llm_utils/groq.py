"""
Groq API integration for Computer Use OOTB.
Groq provides an OpenAI-compatible API, making integration straightforward.

Groq free tier limits (see https://console.groq.com/docs/rate-limits for current limits):
- Llama 3.3 70B: 1,000 requests/day, 12,000 tokens/minute
- Llama 3.1 8B: 14,400 requests/day, 6,000 tokens/minute
- Llama 4 Scout: 1,000 requests/day, 30,000 tokens/minute
- Qwen3 32B: 1,000 requests/day, 6,000 tokens/minute

Note: Groq models are text-only (no vision support).
For GUI automation, use Groq as planner with ShowUI as actor.
"""

import os
import requests
from computer_use_demo.gui_agent.llm_utils.llm_utils import is_image_path


# Groq API base URL (OpenAI-compatible)
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Groq model mappings
GROQ_MODELS = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen3-32b": "qwen/qwen3-32b",
}


def run_groq_interleaved(
    messages: list,
    system: str,
    llm: str,
    api_key: str,
    max_tokens: int = 256,
    temperature: float = 0,
):
    """
    Send chat completion request to Groq API.

    Note: Groq does not support vision models, so images in messages
    will be described as "[Screenshot attached]" placeholder.

    Args:
        messages: List of message content (text and image paths)
        system: System prompt
        llm: Model name (will be mapped to Groq model ID)
        api_key: Groq API key
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        tuple: (response_text, token_usage)
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Get your free API key at https://console.groq.com/"
        )

    # Map model name to Groq model ID
    model_id = GROQ_MODELS.get(llm, llm)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    final_messages = [{"role": "system", "content": system}]

    # Process messages - Groq doesn't support images, so we handle them specially
    if isinstance(messages, list):
        for item in messages:
            contents = []
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            # Groq doesn't support vision - add placeholder
                            content = {
                                "type": "text",
                                "text": "[Screenshot attached - analyzing current screen state]",
                            }
                        else:
                            content = {"type": "text", "text": cnt}
                        contents.append(content)
                message = {"role": item["role"], "content": contents}

            elif isinstance(item, str):
                if is_image_path(item):
                    # Groq doesn't support vision - add placeholder
                    contents.append(
                        {
                            "type": "text",
                            "text": "[Screenshot attached - analyzing current screen state]",
                        }
                    )
                    message = {"role": "user", "content": contents}
                else:
                    contents.append({"type": "text", "text": item})
                    message = {"role": "user", "content": contents}
            else:
                contents.append({"type": "text", "text": str(item)})
                message = {"role": "user", "content": contents}

            final_messages.append(message)
    elif isinstance(messages, str):
        final_messages.append({"role": "user", "content": messages})

    # Flatten content arrays to strings for Groq (simpler format)
    for msg in final_messages:
        if isinstance(msg["content"], list):
            msg["content"] = " ".join(
                c["text"] if isinstance(c, dict) and "text" in c else str(c)
                for c in msg["content"]
            )

    print(f"[groq] sending messages to model: {model_id}")

    payload = {
        "model": model_id,
        "messages": final_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(
            f"{GROQ_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        result = response.json()

        if response.status_code == 200:
            text = result["choices"][0]["message"]["content"]
            token_usage = int(result.get("usage", {}).get("total_tokens", 0))
            return text, token_usage
        else:
            error_msg = result.get("error", {}).get("message", str(result))
            print(f"[groq] Error: {error_msg}")
            raise Exception(f"Groq API error: {error_msg}")

    except requests.exceptions.Timeout:
        raise Exception("Groq API request timed out")
    except Exception as e:
        print(f"[groq] Error in Groq API call: {e}")
        raise


if __name__ == "__main__":
    # Test the Groq API integration
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY environment variable to test")
    else:
        text, token_usage = run_groq_interleaved(
            messages=["Hello, what model are you?"],
            system="You are a helpful assistant.",
            llm="llama-3.3-70b",
            api_key=api_key,
            max_tokens=100,
            temperature=0,
        )
        print(f"Response: {text}")
        print(f"Token usage: {token_usage}")
