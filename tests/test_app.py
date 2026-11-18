"""Smoke tests for the Streamlit dashboard structure."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from config import APP_TITLE

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"


def test_dashboard_renders_expected_controls() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    assert not app.exception, [item.value for item in app.exception]
    assert app.title[0].value == APP_TITLE
    assert any("Minta bizonytalanság" in item.value for item in app.sidebar.header)
    assert [item.label for item in app.sidebar.toggle] == [
        "Bootstrap felhők megjelenítése"
    ]
    assert len(app.dataframe) >= 2


def test_dashboard_simulation_toggle_rerenders() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    app.sidebar.toggle[0].set_value(True).run()

    assert not app.exception, [item.value for item in app.exception]
    assert app.caption  # chart annotation and simulation note are present


def test_reset_button_restores_equal_weights() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    app.sidebar.slider[0].set_value(0.70).run()
    assert app.sidebar.slider[0].value == 0.70

    app.sidebar.button[0].click().run()

    assert not app.exception, [item.value for item in app.exception]
    assert [slider.value for slider in app.sidebar.slider] == [0.25] * 4
