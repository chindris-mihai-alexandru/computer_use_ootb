"""
Google Gemini API integration for Computer Use OOTB.
Uses the new google-genai SDK (NOT deprecated google.generativeai).

Gemini free tier limits (as of Jan 2026):
- Gemini 2.5 Flash: 500 requests/day, 10 requests/min
- Gemini 2.0 Flash: 1,500 requests/day, 15 requests/min
- Gemini 2.0 Flash Lite: 1,500 requests/day, 30 requests/min
- Gemini 1.5 Flash: 1,500 requests/day, 15 requests/min
- Gemini 1.5 Flash-8B: 1,500 requests/day, 15 requests/min
- All models: 1 million tokens/min input

Note: Gemini models support vision (multimodal input).
Get your free API key at: https://aistudio.google.com/apikey
"""

import os
import base64
from computer_use_demo.gui_agent.llm_utils.llm_utils import is_image_path, encode_image

# Gemini model mappings
GEMINI_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
}


def run_gemini_interleaved(
    messages: list,
    system: str,
    llm: str,
    api_key: str,
    max_tokens: int = 256,
    temperature: float = 0,
):
    """
    Send chat completion request to Google Gemini API.

    Args:
        messages: List of message content (text and image paths)
        system: System prompt
        llm: Model name (will be mapped to Gemini model ID)
        api_key: Google API key
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        tuple: (response_text, token_usage)
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai package is required. Install it with: pip install google-genai"
        )

    api_key = (
        api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Get your free API key at https://aistudio.google.com/apikey"
        )

    # Map model name to Gemini model ID
    model_id = GEMINI_MODELS.get(llm, llm)

    # Create client
    client = genai.Client(api_key=api_key)

    # Build contents list
    contents = []

    # Process messages
    if isinstance(messages, list):
        for item in messages:
            parts = []

            if isinstance(item, dict):
                for cnt in item.get("content", []):
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            # Encode image as base64
                            image_base64 = encode_image(cnt)
                            # Determine mime type from extension
                            ext = cnt.lower().split(".")[-1]
                            mime_type = {
                                "jpg": "image/jpeg",
                                "jpeg": "image/jpeg",
                                "png": "image/png",
                                "gif": "image/gif",
                                "webp": "image/webp",
                            }.get(ext, "image/png")

                            parts.append(
                                types.Part.from_bytes(
                                    data=base64.b64decode(image_base64),
                                    mime_type=mime_type,
                                )
                            )
                        else:
                            parts.append(types.Part.from_text(text=cnt))

                role = "user" if item.get("role") == "user" else "model"
                if parts:
                    contents.append(types.Content(role=role, parts=parts))

            elif isinstance(item, str):
                if is_image_path(item):
                    # Encode image as base64
                    image_base64 = encode_image(item)
                    ext = item.lower().split(".")[-1]
                    mime_type = {
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "png": "image/png",
                        "gif": "image/gif",
                        "webp": "image/webp",
                    }.get(ext, "image/png")

                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(image_base64),
                            mime_type=mime_type,
                        )
                    )
                else:
                    parts.append(types.Part.from_text(text=item))

                if parts:
                    contents.append(types.Content(role="user", parts=parts))
            else:
                # Handle other types by converting to string
                parts.append(types.Part.from_text(text=str(item)))
                contents.append(types.Content(role="user", parts=parts))

    elif isinstance(messages, str):
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=messages)])
        )

    print(f"[gemini] sending messages to model: {model_id}")

    try:
        # Generate content
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        text = response.text
        token_usage = getattr(response.usage_metadata, "total_token_count", 0) or 0

        return text, token_usage

    except Exception as e:
        print(f"[gemini] Error in Gemini API call: {e}")
        raise


if __name__ == "__main__":
    # Test the Gemini API integration
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY environment variable to test")
    else:
        text, token_usage = run_gemini_interleaved(
            messages=["Hello, what model are you?"],
            system="You are a helpful assistant.",
            llm="gemini-2.0-flash",
            api_key=api_key,
            max_tokens=100,
            temperature=0,
        )
        print(f"Response: {text}")
        print(f"Token usage: {token_usage}")
