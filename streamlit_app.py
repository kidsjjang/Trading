"""Streamlit Community Cloud entrypoint.

Streamlit reruns this file in a persistent Python process. Running the dashboard
script explicitly avoids Python import caching leaving the app blank on reruns.
"""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).with_name("dashboard.py")), run_name="__main__")
