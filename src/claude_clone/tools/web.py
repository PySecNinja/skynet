"""Web search and fetch tools."""

import asyncio
import warnings
from typing import Any

from claude_clone.tools.base import Tool, ToolResult


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo."""

    name = "web_search"
    description = (
        "Search the web for current information. Use this to find up-to-date information, "
        "documentation, tutorials, news, or answers to questions about topics you're unsure about."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "The search query",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return. Default: 5",
        },
    }
    required = ["query"]

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        """Search the web and return results."""
        try:
            # Run the synchronous DuckDuckGo search in a thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._search(query, max_results)
            )

            if not results:
                return ToolResult.ok(f"No results found for: {query}")

            # Format results
            output_lines = [f"Search results for: {query}\n"]
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("href", result.get("link", ""))
                snippet = result.get("body", result.get("snippet", ""))

                output_lines.append(f"{i}. {title}")
                if url:
                    output_lines.append(f"   URL: {url}")
                if snippet:
                    # Clean up snippet
                    snippet = snippet.replace("\n", " ").strip()
                    output_lines.append(f"   {snippet[:300]}")
                output_lines.append("")

            return ToolResult.ok("\n".join(output_lines), result_count=len(results))

        except Exception as e:
            return ToolResult.fail(f"Search error: {str(e)}")

    def _search(self, query: str, max_results: int) -> list[dict]:
        """Perform the actual search (synchronous)."""
        # Suppress the deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from duckduckgo_search import DDGS

            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                    return results
            except Exception:
                # Try alternative method
                ddgs = DDGS()
                results = list(ddgs.text(query, max_results=max_results))
                return results


class WebFetchTool(Tool):
    """Fetch and extract text content from a URL."""

    name = "web_fetch"
    description = (
        "Fetch the content of a web page and extract its text. "
        "Use this to read documentation, articles, or any web page content."
    )
    parameters = {
        "url": {
            "type": "string",
            "description": "The URL to fetch",
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum characters to return. Default: 10000",
        },
    }
    required = ["url"]

    async def execute(
        self,
        url: str,
        max_length: int = 10000,
        **kwargs: Any,
    ) -> ToolResult:
        """Fetch and extract text from a URL."""
        try:
            import httpx
            from html.parser import HTMLParser

            # Simple HTML text extractor
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts = []
                    self.skip_tags = {"script", "style", "nav", "footer", "header", "aside"}
                    self.current_skip = 0

                def handle_starttag(self, tag, attrs):
                    if tag in self.skip_tags:
                        self.current_skip += 1

                def handle_endtag(self, tag):
                    if tag in self.skip_tags and self.current_skip > 0:
                        self.current_skip -= 1

                def handle_data(self, data):
                    if self.current_skip == 0:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)

                def get_text(self) -> str:
                    return " ".join(self.text_parts)

            # Fetch the URL with httpx (async)
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SkyNet/1.0)"}
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text

            # Extract text
            extractor = TextExtractor()
            extractor.feed(html)
            text = extractor.get_text()

            # Clean up whitespace
            import re
            text = re.sub(r'\s+', ' ', text).strip()

            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + "\n\n[Content truncated...]"

            if not text.strip():
                return ToolResult.fail("Could not extract text content from the page")

            return ToolResult.ok(f"Content from {url}:\n\n{text}", url=url)

        except httpx.HTTPStatusError as e:
            return ToolResult.fail(f"HTTP error {e.response.status_code}: {url}")
        except httpx.RequestError as e:
            return ToolResult.fail(f"Request error: {str(e)}")
        except Exception as e:
            return ToolResult.fail(f"Fetch error: {str(e)}")
