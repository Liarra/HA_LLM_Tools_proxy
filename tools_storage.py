"""Read, write and find tools in the storage."""


def encode_tool(tool: dict) -> int:
    """Encode a tool dictionary into a string."""
    tool_name= tool.get("function", {}).get("name", "")
    tool_description = tool.get("function", {}).get("description", "")
    return hash(f"{tool_name}:{tool_description}")

storage=dict()
def store_tool(tool: dict) -> None:
    """Store a tool in the memory pseudo-storage."""
    if not isinstance(tool, dict):
        raise ValueError("Tool must be a dictionary.")
    storage[encode_tool(tool)]=tool

def exists_in_storage(tool: dict) -> bool:
    """Check if a tool exists in the storage."""
    if not isinstance(tool, dict):
        raise ValueError("Tool must be a dictionary.")
    encoded_tool = encode_tool(tool)
    return encoded_tool in storage.keys()

def dump_tools_into_storage(tools: list) -> None:
    """Dump a list of tools into the storage."""
    if not isinstance(tools, list):
        raise ValueError("Tools must be a list.")
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Each tool must be a dictionary.")
        if not exists_in_storage(tool):
            store_tool(tool)

def get_3_random_tools() -> list:
    """Get 3 random tools from the storage."""
    if len(storage) < 3:
        return list(storage.values())
    import random
    return random.sample(list(storage.values()), 3)