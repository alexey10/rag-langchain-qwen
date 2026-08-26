import sys
import json
from pathlib import Path

# Ensure project root is on the path regardless of where script is run from
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api.main import app

with open("docs/openapi.json", "w") as f:
    json.dump(app.openapi(), f, indent=2)

print("OpenAPI specification exported to docs/openapi.json")
