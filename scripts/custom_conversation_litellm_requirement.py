"""Print the LiteLLM requirement from Custom Conversation's current manifest."""

import json
from urllib.request import urlopen

MANIFEST_URL = (
    "https://raw.githubusercontent.com/michelle-avery/custom-conversation/"
    "main/custom_components/custom_conversation/manifest.json"
)

with urlopen(MANIFEST_URL, timeout=30) as response:
    manifest = json.load(response)

try:
    print(next(item for item in manifest["requirements"] if item.startswith("litellm")))
except StopIteration as error:
    raise SystemExit(
        "Custom Conversation does not currently declare LiteLLM"
    ) from error
