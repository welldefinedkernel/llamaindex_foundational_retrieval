import argparse

from mcp.server.fastmcp import FastMCP
from pipeline.run_pipeline import run_pipeline

# Initialize FastMCP server
mcp = FastMCP(
    "RAG Retrieval Server", 
    host = "127.0.0.1",
    port = 8001
)

@mcp.tool("retrieve_chunks")
def retrieve_chunks(query: str) -> list[str]:
    """Tool to retrieve relevant chunks for a given query."""
    return run_pipeline(query)

def main():
    parser = argparse.ArgumentParser(description="Start the RAG Pipeline MCP Server.")
    parser.add_argument(
        "--port", type=int, default=9999, help="Port to run the MCP server on (default: 9999)"
    )
    args = parser.parse_args()
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
