# MCP server for coding tasks with the following tools:
# - code.list_csvs
# - code.inspect_csv
# - code.run
# - code.validate

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import os, json, subprocess
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("code-server")

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "resources"
DATA_DIR = RES_DIR / "data"
RUNS_DIR = RES_DIR / "runs"


# --- helpers ---

def _now() -> str:
    """Return current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# --------- tools ---------

@mcp.tool("code.list_csvs")
def list_csvs(dir: str = str(DATA_DIR)) -> Dict[str, Any]:
    """List all CSV files in a directory."""
    base = Path(dir)
    if not base.exists():
        return {"csvs": [], "note": f"Directory not found: {base}"}
    items = []
    for p in sorted(base.glob("*.csv")):
        try:
            size_kb = round(p.stat().st_size / 1024.0, 1)
        except Exception:
            size_kb = None
        items.append({
            "path": str(p),
            "filename": p.name,
            "size_kb": size_kb,
        })
    return {"csvs": items}

@mcp.tool("code.inspect_csv")
def inspect_csv(path: str, n_head: int = 5) -> Dict[str, Any]:
    """Inspect CSV: rows, columns, dtypes, and head."""
    p = Path(path)
    if not p.exists():
        return {"path": path, "error": "file_not_found"}
    df = pd.read_csv(p)
    return {
        "path": str(p),
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns.tolist()],
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "head": json.loads(df.head(max(1, int(n_head))).to_json(orient="records")),
    }

@mcp.tool("code.run")
def run_code(
    code: str,
    csv_path: Optional[str] = None,
    attempt: int = 1,
    timeout_s: int = 20,
    run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run user code in isolated directory and capture outputs."""
    if run_dir is None:
        run_dir_path = RUNS_DIR / f"{_now()}"
    else:
        run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    script_path = run_dir_path / f"attempt_{attempt}.py"
    script_path.write_text(code, encoding="utf-8")

    env = os.environ.copy()
    if csv_path:
        env["CSV_PATH"] = str(csv_path)

    try:
        proc = subprocess.run(
            [os.sys.executable, str(script_path)],
            cwd=str(run_dir_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_s),
        )
        stdout, stderr = proc.stdout, proc.stderr
        success = (proc.returncode == 0)
    except Exception as e:
        stdout, stderr, success = False, f"{type(e).__name__}: {e}", False

    (run_dir_path / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (run_dir_path / "stderr.txt").write_text(stderr or "", encoding="utf-8")

    # Optional previews
    preview = None
    plot_path = None
    res_csv = run_dir_path / "result.csv"
    if res_csv.exists():
        try:
            preview = json.loads(pd.read_csv(res_csv).head(10).to_json(orient="records"))
        except Exception:
            preview = None
    if (run_dir_path / "plot.png").exists():
        plot_path = str(run_dir_path / "plot.png")

    return {
        "success": bool(success),
        "stdout_tail": (stdout or "")[-2000:],
        "stderr_tail": (stderr or "")[-2000:],
        "result_preview": preview,
        "plot_path": plot_path,
        "run_dir": str(run_dir_path),
    }

@mcp.tool("code.validate")
def validate(
    run_dir: str,
    required_columns: Optional[List[str]] = None,
    non_empty: bool = True,
) -> Dict[str, Any]:
    """Validate result.csv against basic conditions and required columns."""

    required_columns = required_columns or []
    res_csv = Path(run_dir) / "result.csv"
    if not res_csv.exists():
        return {"passed": False, "messages": ["result.csv not found"]}

    # Fast pre-check
    try:
        if res_csv.stat().st_size == 0:
            return {"passed": False, "messages": ["result.csv is empty (0 bytes)."]}
    except Exception:
        pass  # fall through to pandas read

    try:
        df = pd.read_csv(res_csv)
    except EmptyDataError:
        return {"passed": False, "messages": ["result.csv has no columns/rows (empty)."]}
    except ParserError as e:
        return {"passed": False, "messages": [f"Could not parse result.csv: {e}"]}
    except Exception as e:
        return {"passed": False, "messages": [f"Error reading result.csv: {type(e).__name__}: {e}"]}

    msgs, passed = [], True
    if non_empty and len(df) == 0:
        passed = False
        msgs.append("Result is empty.")
    else:
        msgs.append(f"Result has {len(df)} rows.")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        passed = False
        msgs.append(f"Missing columns: {missing}, Got: {list(df.columns)}")
    elif required_columns:
        msgs.append("Required columns present.")

    return {"passed": passed, "messages": msgs}


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    mcp.run(transport="stdio")
