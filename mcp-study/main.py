from fastmcp import FastMCP

mcp = FastMCP("Demo")


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Generate a greeting for the given name."""
    return f"Hello, {name}"


@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt with specified style."""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }
    return f"{styles.get(style, styles['friendly'])} for someone named {name}"


if __name__ == "__main__":
    # mcp.run(transport="stdio")
    # mcp.run(transport="sse")
    mcp.run(transport="streamable-http")
