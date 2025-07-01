"""Read, write and find tools in the storage."""
import json

import embedding
import sqlite3
import hashlib

def crude_hash(s: str) -> int:
    """Return a crude hash of the string."""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:10], 16)


def get_tool_label(tool: dict) -> str:
    """Encode a tool dictionary into a string."""
    tool_name= tool.get("function", {}).get("name", "")
    tool_description = tool.get("function", {}).get("description", "")
    return f"{tool_name}:{tool_description}"

tool_definition_storage=dict()
def store_tool(tool: dict) -> None:
    """Store a tool in the memory pseudo-storage."""
    if not isinstance(tool, dict):
        raise ValueError("Tool must be a dictionary.")

    tool_label=get_tool_label(tool)
    print (f"Storing tool: {tool_label}")
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

def get_n_most_relevant_tools(request:str, n: int=3) -> list:
    """Get n most relevant tools from the storage."""
    labels = embedding.retrieve_similar(request, k=n)

    # Return the tools corresponding to the labels
    tools = []
    for _, score, label in labels:
        print (f"Tool {label} with score {score} was picked for this query")
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