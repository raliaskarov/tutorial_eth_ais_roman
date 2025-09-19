# MCP server for fridge tasks with the following tools:
# - fridge.list      -> list current inventory (sorted by soonest expiry)
# - fridge.add       -> add/update an item (merge quantities if same expiry)
# - fridge.remove    -> remove all rows matching an item (case-insensitive)
# - fridge.cost      -> compute total cost using a price catalog if available
#
# Assumes:
# CSV schema (resources/data/fridge_inventory.csv):
#   Item,Quantity,Expiry Date   (Expiry Date in ISO: YYYY-MM-DD)

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fridge-server")

ROOT = Path(os.getenv("FRIDGE_ROOT", Path(__file__).resolve().parent.parent)).expanduser().resolve()
RES_DIR = ROOT / "resources"
DATA_DIR = RES_DIR / "data"

INV_PATH = DATA_DIR / "fridge_inventory.csv"
ISO_FMT = "%Y-%m-%d"


# --- helpers ---

def _ensure_inventory_file() -> None:
    """Ensure inventory file exists with correct header."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not INV_PATH.exists():
        INV_PATH.write_text("Item,Quantity,Expiry Date\n", encoding="utf-8")

def _norm_item(s: str) -> str:
    """Normalize an item name string (trim, lowercase, singularize basic plurals)."""
    s = (s or "").strip().lower()
    if not s:
        return s
    # very basic plural handling
    if s.endswith("ies") and len(s) > 3:
        s = s[:-3] + "y"        # e.g. "berries" -> "berry"
    elif s.endswith("oes") and len(s) > 3:
        s = s[:-2]              # e.g. "tomatoes" -> "tomato"
    elif s.endswith("ses") and len(s) > 3:
        s = s[:-2]              # e.g. "buses" -> "bus"
    elif s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]              # e.g. "apples" -> "apple"
    return s

def _norm_mask(df: pd.DataFrame, item: str):
    """Case/whitespace/plural-insensitive match for Item."""
    key = _norm_item(item)
    return df["Item"].map(_norm_item).eq(key)

def _parse_iso_date(s: str) -> Optional[datetime]:
    """Parse ISO date string to datetime object."""
    s = (s or "").strip()
    try:
        return datetime.strptime(s, ISO_FMT)
    except Exception:
        return None

def _read_df(path: Path):
    """Read CSV into DataFrame with cleaned types."""
    _ensure_inventory_file()
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["Item", "Quantity", "Expiry Date"])
    # normalize dtypes
    if "Quantity" in df.columns:
        try:
            df["Quantity"] = df["Quantity"].fillna(0).astype(int)
        except Exception:
            df["Quantity"] = df["Quantity"].apply(lambda x: int(float(x)) if str(x).strip() != "" else 0)
    for col in ["Item", "Expiry Date"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").map(lambda x: x.strip())
    return df

def _write_df_atomic(df, path: Path) -> None:
    """Write DataFrame to CSV atomically."""
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def _sort_by_expiry(df):
    """Sort DataFrame by expiry date then item."""
    if "Expiry Date" not in df.columns:
        return df
    tmp = df.copy()
    tmp["_expiry"] = pd.to_datetime(tmp["Expiry Date"], errors="coerce", format=ISO_FMT)
    tmp = tmp.sort_values(by=["_expiry", "Item"], na_position="last").drop(columns=["_expiry"])
    return tmp


# --------- tools ---------

@mcp.tool("fridge.list")
def list_fridge() -> Dict[str, Any]:
    """Return current inventory as records, sorted by soonest expiry."""
    _ensure_inventory_file()
    df = _read_df(INV_PATH)
    df = _sort_by_expiry(df)
    records: List[Dict[str, Any]] = json.loads(df.to_json(orient="records"))
    earliest = None
    if len(df) and "Expiry Date" in df.columns:
        dt = _parse_iso_date(str(df.iloc[0]["Expiry Date"]))
        earliest = dt.strftime(ISO_FMT) if dt else None
    total_qty = int(df["Quantity"].sum()) if "Quantity" in df.columns and len(df) else 0
    return {
        "path": str(INV_PATH),
        "n_rows": int(len(df)),
        "total_quantity": total_qty,
        "earliest_expiry": earliest,
        "rows": records,
    }


@mcp.tool("fridge.add")
def add_to_fridge(item: str, quantity: int, expiry_date: str) -> Dict[str, Any]:
    """Add or increase quantity for an item with a given expiry date."""
    item_n = _norm_item(item)
    if not item_n:
        return {"ok": False, "error": "invalid_item", "message": "Item name is required."}
    try:
        q = int(quantity)
    except Exception:
        return {"ok": False, "error": "invalid_quantity", "message": "Quantity must be an integer."}
    if q < 0:
        return {"ok": False, "error": "invalid_quantity", "message": "Quantity must be >= 0."}

    dt = _parse_iso_date(expiry_date)
    if not dt:
        return {"ok": False, "error": "invalid_date", "message": f"Use ISO date YYYY-MM-DD (got '{expiry_date}')."}
    exp_s = dt.strftime(ISO_FMT)

    df = _read_df(INV_PATH)
    if len(df) == 0:
        df = pd.DataFrame(columns=["Item", "Quantity", "Expiry Date"])

    # Merge by same item (case-insensitive) AND same expiry date
    mask = _norm_mask(df, item) & (df["Expiry Date"] == exp_s)
    if mask.any():
        df.loc[mask, "Quantity"] = df.loc[mask, "Quantity"].astype(int) + q
        action = "updated"
    else:
        df = pd.concat([df, pd.DataFrame([{"Item": item_n, "Quantity": int(q), "Expiry Date": exp_s}])], ignore_index=True)
        action = "inserted"

    df = _sort_by_expiry(df)
    _write_df_atomic(df, INV_PATH)

    return {
        "ok": True,
        "action": action,
        "path": str(INV_PATH),
        "item": item_n,
        "quantity_added": int(q),
        "expiry_date": exp_s,
        "n_rows": int(len(df)),
    }

@mcp.tool("fridge.remove")
def remove_from_fridge(item: str, quantity: int) -> Dict[str, Any]:
    """Remove up to `quantity` units of an item (case-insensitive, trimmed).
    Removes from the soonest-expiring rows first (FEFO). Fails if not enough stock.
    """
    item_n = _norm_item(item)
    if not item_n:
        return {"ok": False, "error": "invalid_item", "message": "Item name is required."}

    # validate quantity
    try:
        q = int(quantity)
    except Exception:
        return {"ok": False, "error": "invalid_quantity", "message": "Quantity must be an integer."}
    if q <= 0:
        return {"ok": False, "error": "invalid_quantity", "message": "Quantity must be > 0."}

    df = _read_df(INV_PATH)
    if len(df) == 0:
        return {
            "ok": False,
            "error": "not_found",
            "message": f"No inventory yet. Cannot remove {q} × {item_n}.",
            "path": str(INV_PATH),
        }

    mask = _norm_mask(df, item)
    if not mask.any():
        return {
            "ok": False,
            "error": "not_found",
            "message": f"Item '{item_n}' not found in the fridge.",
            "path": str(INV_PATH),
        }

    avail_total = int(df.loc[mask, "Quantity"].sum())
    if q > avail_total:
        return {
            "ok": False,
            "error": "insufficient_quantity",
            "message": f"Requested {q} × {item_n}, but only {avail_total} available.",
            "available_quantity": avail_total,
            "path": str(INV_PATH),
        }

    # Remove from soonest-expiring first (FEFO)
    sub = _sort_by_expiry(df.loc[mask].copy())
    to_remove = q
    removed_rows_zeroed = 0

    for idx in sub.index:
        if to_remove <= 0:
            break
        have = int(sub.at[idx, "Quantity"])
        take = min(have, to_remove)
        new_q = have - take
        sub.at[idx, "Quantity"] = int(new_q)
        if new_q == 0:
            removed_rows_zeroed += 1
        to_remove -= take

    # Write back: merge the adjusted subset with the untouched rows, drop zero-qty rows
    df_updated = pd.concat([df.loc[~mask], sub], ignore_index=True)
    df_updated = df_updated[df_updated["Quantity"] > 0].copy()
    df_updated = _sort_by_expiry(df_updated)
    _write_df_atomic(df_updated, INV_PATH)

    return {
        "ok": True,
        "item": item_n,
        "removed_quantity": int(q),
        "removed_rows": int(removed_rows_zeroed),
        "n_rows": int(len(df_updated)),
        "path": str(INV_PATH),
    }

@mcp.tool("fridge.cost")
def calculate_total_fridge_cost(prices: Dict[str, float]) -> Dict[str, Any]:
    """Compute total cost using UnitPrice from a price CSV (or fallback)."""
    df = _read_df(INV_PATH)
    if len(df) == 0:
        return {"total_cost": 0.0, "currency": "CHF", "items": [], "missing_prices": [], "path": str(INV_PATH)}

    items = []
    missing = set()
    total = 0.0

    for _, row in df.iterrows():
        name = row["Item"]
        qty = int(row["Quantity"])
        if name in prices:
            unit = float(prices[name])
            subtotal = unit * qty
            total += subtotal
            items.append({
                "Item": name,
                "Quantity": int(qty),
                "UnitPrice": round(unit, 4),
                "Subtotal": round(subtotal, 4),
            })
        else:
            missing.add(name)

    return {
        "path": str(INV_PATH),
        "currency": "CHF",
        "total_cost": round(float(total), 2),
        "items": items,
        "missing_prices": sorted(missing),
    }

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_inventory_file()
    mcp.run(transport="stdio")
