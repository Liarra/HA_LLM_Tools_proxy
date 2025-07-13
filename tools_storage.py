"""Read, write and find tools in the storage."""
import json
import logging
import os

import embedding
import sqlite3
import hashlib

# Configure logging only if not already configured
if not logging.getLogger().handlers:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)

def crude_hash(s: str) -> int:
    """Return a crude hash of the string."""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:10], 16)


def get_tool_label(tool: dict) -> str:
    """Encode a tool dictionary into a string."""
    tool_name= tool.get("function", {}).get("name", "")
    tool_description = tool.get("function", {}).get("description", "")
    return f"{tool_name}:{tool_description}"

def get_tool_label_by_name(tool_name:str) -> str:
    """Get the tool label by its name. Assumes every tool has a unique name."""
    if not isinstance(tool_name, str):
        raise ValueError("Tool name must be a string.")
    # Find the tool in the storage by its name
    for tool in tool_definition_storage.values():
        if tool.get("function", {}).get("name") == tool_name:
            return get_tool_label(tool)
    raise ValueError(f"Tool with name '{tool_name}' not found in storage.")


tool_definition_storage=dict()
def store_tool(tool: dict) -> None:
    """Store a tool in the memory pseudo-storage."""
    if not isinstance(tool, dict):
        raise ValueError("Tool must be a dictionary.")

    tool_label=get_tool_label(tool)
    logger.debug("Storing tool: %s", tool_label)
    embedding.store_text_into_faiss(tool_label)
    h=crude_hash(tool_label)
    tool_definition_storage[h]=tool

def exists_in_storage(tool: dict) -> bool:
    """Check if a tool exists in the storage."""
    if not isinstance(tool, dict):
        raise ValueError("Tool must be a dictionary.")
    encoded_tool = get_tool_label(tool)
    return crude_hash(encoded_tool) in tool_definition_storage.keys()

def dump_tools_into_storage(tools: list) -> None:
    """Adds any new tools to the storage."""
    if not isinstance(tools, list):
        raise ValueError("Tools must be a list.")
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Each tool must be a dictionary.")
        if not exists_in_storage(tool):
            store_tool(tool)

def get_3_random_tools() -> list:
    """Get 3 random tools from the storage."""
    if len(tool_definition_storage) < 3:
        return list(tool_definition_storage.values())
    import random
    return random.sample(list(tool_definition_storage.values()), 3)

def get_whitelisted_labels() -> list:
    """Get the labels of the whitelisted tools."""
    ret= []
    for wt in whitelisted_tool_names:
        if not isinstance(wt, str):
            raise ValueError("Whitelisted tool names must be strings.")
        try:
            label = get_tool_label_by_name(wt)
            ret.append(label)
        except ValueError as e:
            logger.warning("Error getting label for whitelisted tool '%s': %s", wt, e)
            continue
    return ret

def get_blacklisted_labels() -> list:
    """Get the labels of the blacklisted tools."""
    ret= []
    for bt in blacklisted_tool_names:
        if not isinstance(bt, str):
            raise ValueError("Blacklisted tool names must be strings.")
        try:
            label = get_tool_label_by_name(bt)
            ret.append(label)
        except ValueError as e:
            logger.warning("Error getting label for blacklisted tool '%s': %s", bt, e)
            continue
    return ret


# Get whitelisted and blacklisted tool names from environment variables
# Default values maintain backward compatibility
_default_whitelist = ['GetLiveContext']
_default_blacklist = ['HassHumidifierMode', 'HassHumidifierSetPoint']

def _parse_tool_names_from_env(env_var: str, default_list: list) -> list:
    """Parse comma-separated tool names from environment variable."""
    env_value = os.getenv(env_var, "")
    if not env_value.strip():
        return default_list
    # Split by comma and strip whitespace from each tool name
    parsed_list = [name.strip() for name in env_value.split(',') if name.strip()]
    # If after parsing we have an empty list, fall back to defaults
    return parsed_list if parsed_list else default_list

whitelisted_tool_names = _parse_tool_names_from_env('WHITELISTED_TOOLS', _default_whitelist)
blacklisted_tool_names = _parse_tool_names_from_env('BLACKLISTED_TOOLS', _default_blacklist)
def get_n_most_relevant_tools(request:str, n: int=3) -> list:
    """Get n most relevant tools from the storage."""
    labels=get_whitelisted_labels()

    # Increase the number of tools to retrieve to be able to remove the blacklisted tools if they are present
    # in the results. Whitelisted tools may appear in the results, so no need to account for them.
    number_of_tools_to_retrieve= n+  len(blacklisted_tool_names)

    if number_of_tools_to_retrieve==0:
        logger.info("No more tools to retrieve, only whitelisted tools will be returned.")

    else:
        embedding_labels = embedding.retrieve_similar(request, k=number_of_tools_to_retrieve)
        blacklisted_labels = get_blacklisted_labels()
        for el in embedding_labels:
            if len(labels)==n:
                logger.debug("Reached the limit of tools to return, stopping.")
                break
            # If the label is blacklisted, skip it
            if el[2] in blacklisted_labels:
                logger.debug("Skipping blacklisted tool: %s", el[2])
                continue
            # If the label is already in the whitelisted tools, skip it
            if el[2] in labels:
                logger.debug("Skipping already included tool: %s", el[2])
                continue
            # Otherwise, add the label to the list
            labels.append(el[2])
            logger.debug("Tool %s with score %f was picked for this query", el[2], el[1])

    # Return the tools corresponding to the labels
    tools = []
    for label in labels:
        h = crude_hash(label)
        if h in tool_definition_storage:
            tools.append(tool_definition_storage[h])

    return tools

def save_tools(sqlite_file_path: str) -> None:
    """Save the tools to a SQLite database."""
    conn = sqlite3.connect(sqlite_file_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash INTEGER UNIQUE,
            tool_name TEXT,
            tool_description TEXT,
            tool_data TEXT
        );''')

    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS embedding (
            id   INTEGER PRIMARY KEY CHECK (id = 1),
            file_path TEXT NOT NULL
        );''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings_labels (
            id   INTEGER PRIMARY KEY,
            text TEXT NOT NULL
        );
    ''')

    # Insert tools into the table
    cursor.execute('DELETE FROM tools')  # Clear existing tools
    for h, tool in tool_definition_storage.items():
        tool_name = tool.get("function", {}).get("name", "")
        tool_description = tool.get("function", {}).get("description", "")
        tool_data = json.dumps(tool)  # Convert dict to string for storage
        cursor.execute('''
            INSERT INTO tools (hash, tool_name, tool_description, tool_data)
            VALUES (?, ?, ?, ?)
        ''', (h, tool_name, tool_description, tool_data))

    # Save embeddings
    embedding.save_to_file('data/embedding.bin')
    cursor.execute('''
        INSERT OR REPLACE INTO embedding (id, file_path) VALUES (?, ?)
    ''', (1, 'data/embedding.bin'))

    # Save labels
    labels = embedding.get_labels()
    cursor.execute('DELETE FROM embeddings_labels')  # Clear existing labels
    for idx, label in enumerate(labels):
        cursor.execute('''
            INSERT INTO embeddings_labels (id, text) VALUES (?, ?)
        ''', (idx, label))

    conn.commit()
    conn.close()


def load_tools(sqlite_file_path: str) -> None:
    """Load tools from a SQLite database into the storage."""
    conn = sqlite3.connect(sqlite_file_path)
    cursor = conn.cursor()

    # Fetch all tools from the table
    cursor.execute('SELECT hash, tool_name, tool_description, tool_data FROM tools')
    rows = cursor.fetchall()

    for h, tool_name, tool_description, tool_data in rows:
        tool = {
            "function": {
                "name": tool_name,
                "description": tool_description
            }
        }
        # Convert string back to dict if necessary
        if isinstance(tool_data, str):
            tool.update(json.loads(tool_data))
        tool_definition_storage[h] = tool

    # Load embeddings
    embed_filepath = cursor.execute('SELECT file_path FROM embedding WHERE id = 1').fetchone()
    if embed_filepath:
        embedding.load_from_file(embed_filepath[0])

    # Load labels
    cursor.execute('SELECT text FROM embeddings_labels')
    rows = cursor.fetchall()
    labels = [row[0] for row in rows]
    embedding.load_labels(labels)

    conn.close()