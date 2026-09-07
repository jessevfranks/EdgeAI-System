"""Start the local Streamlit UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    dashboard = project / "src" / "edge_ai" / "dashboard.py"
    sys.path.insert(0, str(project / "src"))
    os.chdir(project)

    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(dashboard), "--server.address=localhost", "--server.headless=true"]
    cli.main()


if __name__ == "__main__":
    main()
