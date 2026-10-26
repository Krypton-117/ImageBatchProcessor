"""GUI entry point with startup diagnostics even when runtime imports fail."""

import argparse, os, sys, traceback
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--web", action="store_true")
    args = p.parse_args()
    try:
        from ibp_agent.server import serve

        serve(args.no_browser, args.port, studio=not args.web)
    except Exception:
        text = traceback.format_exc()
        data = Path(
            os.environ.get(
                "IBP_AGENT_DATA",
                str(
                    Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
                    / "IBPPhotoAgent"
                ),
            )
        )
        data.mkdir(parents=True, exist_ok=True)
        (data / "error.log").write_text(text, encoding="utf-8")
        if sys.stderr:
            print(text, file=sys.stderr)
        elif not args.no_browser:
            try:
                import ctypes

                ctypes.windll.user32.MessageBoxW(0, text[-1500:], "IBP Agent", 16)
            except Exception:
                pass
        sys.exit(1)
