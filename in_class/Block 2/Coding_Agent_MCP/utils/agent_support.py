# Here are helpers for LLM interaction, code extraction, printing steps,
# They are in in no particular order and are not grouped by functionality.

from __future__ import annotations
import os, re, asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import time

# --- display (notebook-friendly, safe fallback) ---
try:
    from IPython.display import display, Image
except Exception:  # non-notebook fallback
    display = lambda *a, **k: None
    class _Img: 
        def __init__(self, filename): self.filename = filename
    Image = _Img


# -------- LLM setup and utilities (Azure or OpenAI) --------

try:
    from openai import AzureOpenAI, OpenAI
except Exception as e:
    raise RuntimeError("Please `pip install -U openai` to use the LLM.") from e

def _make_llm_client():
    """Return (LLM client, model/deployment) based on env vars."""
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        ), os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_LLM_CLIENT, _LLM_MODEL = _make_llm_client()

async def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    """Call the LLM with messages and return the text content."""
    def _call():
        if isinstance(_LLM_CLIENT, AzureOpenAI):
            return _LLM_CLIENT.chat.completions.create(
                model=_LLM_MODEL, messages=messages, temperature=temperature
            ).choices[0].message.content
        else:
            return _LLM_CLIENT.chat.completions.create(
                model=_LLM_MODEL, messages=messages, temperature=temperature
            ).choices[0].message.content
    return await asyncio.to_thread(_call)


# -------- Printing helpers --------

def print_step(title: str, level: int = 1, delay: float = 0.5, width: int = 80):
    """Print a formatted step banner (major/sub-step) with slight delay."""
    time.sleep(delay)
    inner = f" {title} "
    if level == 1:  # major step
        side_len = max(0, (width - len(inner)) // 2)
        line = "=" * side_len + inner + "=" * (width - side_len - len(inner))
    else:  # sub-step
        width -= 20
        side_len = max(0, (width - len(inner)) // 2)
        line = "-" * side_len + inner + "-" * (width - side_len - len(inner))
        line = " " * 10 + line  # indent to make it visually nested
    print("\n" + line, flush=True)
    time.sleep(delay)

def print_plan(plan):
    """Print a simple numbered plan with optional fields."""
    print("\n--- PLAN ---")
    for i, step in enumerate(plan, 1):
        print(f"Step {i}: intent='{step['intent']}'")
        if step.get("params"):
            print(f"  params: {step['params']}")
        if step.get("use"):
            print(f"  use:    {step['use']}")
        if step.get("save"):
            print(f"  save:   {step['save']}")
        if step.get("output"):
            print(f"  output:   {step['output']}")
    print("------------\n")

def compact_df_preview(rows: List[Dict[str, Any]], max_rows: int = 10):
    """Display a compact DataFrame preview of at most max_rows."""
    if not rows:
        print("(no preview rows)"); return
    df = pd.DataFrame(rows[:max_rows])
    display(df)
    
def compact_preview(out: Dict[str, Any]):
    """Show brief, readable previews of text, stdout, CSV head, and plot."""
    if not out:
        print("(no outputs)")
        return
    if "answer_text" in out and out["answer_text"]:
        print("\n[answer_text]\n", out["answer_text"][:500])
    if "stdout" in out and out["stdout"]:
        tail = out["stdout"][-500:]
        print("\n[stdout tail]\n", tail if tail else "(empty)")
    if "result_csv" in out and out["result_csv"] and os.path.exists(out["result_csv"]):
        print("\n[result_csv head]")
        try:
            df = pd.read_csv(out["result_csv"])
            display(df.head(10))
        except Exception as e:
            print(f"(could not preview CSV: {e})")
    if "plot_png" in out and out["plot_png"] and os.path.exists(out["plot_png"]):
        print("\n[plot_png]")
        display(Image(filename=out["plot_png"], width=800, height=400))

def intent_specs_text(registry: dict) -> str:
    """Render an INTENT_REGISTRY into a readable, prompt-ready string."""
    lines = ["INTENTS:"]
    for name, d in registry.items():
        params = ", ".join(d.get("params", {}).keys()) or "(none)"
        uses   = ", ".join(d.get("uses", [])) or "(none)"
        saves  = ", ".join(d.get("saves", [])) or "(none)"
        desc   = d.get("description", "")
        lines.append(
            f"- {name}: params [{params}] | can optionally use [{uses}] | can optionally save [{saves}] — {desc}"
        )
    return "\n".join(lines)

def show_final_outputs(ctx: Dict[str, Any], max_final: int = 5):
    """Display final outputs by priority: answer_text, result_csv, plot_png."""
    priority = ["answer_text", "result_csv", "plot_png"]
    print_step(f"FINAL OUTPUT", level=2)
    shown = 0
    for key in priority:
        if key in ctx and ctx[key]:
            val = ctx[key]
            compact_preview({key: val} if not isinstance(val, dict) else val)
            shown += 1
            if shown >= max_final:
                break
            

# -------- Further notebook utilities --------

def extract_code(text: str) -> str:
    """Extract Python code block (```python ... ```) or return stripped text."""
    if not text: return ""
    m = re.search(r"```python(.*?)```", text, flags=re.DOTALL|re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()

def load_prompts() -> dict[str, str]:
    """Load *.md prompt files from the local 'prompts' directory."""
    base = Path("prompts")  # same base as your existing code
    if not base.is_dir():
        return {}

    prompts: dict[str, str] = {}
    for p in sorted(base.glob("*.md")):
        try:
            prompts[p.stem] = p.read_text(encoding="utf-8")
        except OSError:
            # Keep the key but fall back to empty string on read error
            prompts[p.stem] = ""
    return prompts

def discover_run_artifacts(run_dir: Optional[str]) -> Dict[str, Any]:
    """Find common artifacts (result.csv, plot.png) in a run directory."""
    out: Dict[str, Any] = {}
    if not run_dir: 
        return out
    rd = Path(run_dir)
    rcsv = rd / "result.csv"
    rpng = rd / "plot.png"
    if rcsv.exists(): out["result_csv"] = str(rcsv)
    if rpng.exists(): out["plot_png"]  = str(rpng)
    return out

__all__ = [
    "llm_chat",
    "extract_code",
    "print_step",
    "compact_df_preview",
    "load_prompts",
    "Image",
    "show_final_outputs",
    "compact_preview",
    "print_plan",
    "intent_specs_text",
    "discover_run_artifacts",
]