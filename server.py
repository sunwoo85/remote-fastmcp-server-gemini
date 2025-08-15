# server.py
"""
Minimal MCP server for Gemini text generation (FastMCP Cloud/Claude Desktop)
"""

import time
from fastmcp import FastMCP
from google import genai
from google.genai import types as gt

mcp = FastMCP(name="Gemini")
client = genai.Client()  # relies on environment auth (e.g., GOOGLE_API_KEY)

@mcp.tool
def gemini_generate_text(
    prompt: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    grounding: bool = True,
    delay_seconds: float = 5,
) -> str:
    """
    Generate text with Gemini.
    """
    try:
        if model == "gemini-2.5-pro":
            delay_seconds = 25
            
        tools = [gt.Tool(google_search=gt.GoogleSearch())] if grounding else None
        config = gt.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
        )
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        text = getattr(resp, "text", None)
        if text:
            time.sleep(delay_seconds)
            return text
        out = []
        for c in getattr(resp, "candidates", []) or []:
            for p in getattr(getattr(c, "content", None), "parts", []) or []:
                if getattr(p, "text", None):
                    out.append(p.text)
        time.sleep(delay_seconds)
        return "".join(out)
    except Exception as e:
        return f"Error generating text: {e}"

if __name__ == "__main__":
    mcp.run()
