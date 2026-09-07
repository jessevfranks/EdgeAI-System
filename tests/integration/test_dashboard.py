from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_initial_page_renders() -> None:
    dashboard = Path(__file__).resolve().parents[2] / "src" / "edge_ai" / "dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "EdgeAI YOLOv8 Experiments"
    for page in ["Jobs", "Experiment Detail", "Compare", "Promote / Evaluate", "Exports"]:
        app.sidebar.radio[0].set_value(page)
        app.run(timeout=20)
        assert not app.exception
