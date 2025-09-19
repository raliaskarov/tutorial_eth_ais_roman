# The sweep_stale_servers_in_dir function in this file is necessary to avoid
# leaving behind orphaned server processes which are generated in our client-server architecture.

from pathlib import Path
import os
import sys

def sweep_stale_servers_in_dir(servers_dir, *, include_subdirs=False, timeout=3, verbose=True):
    """
    Kill leftover Python processes whose script path is inside `servers_dir`.

    - `servers_dir`: path-like (absolute or relative)
    - `include_subdirs`: if True, match any script under the tree; if False, only direct children
    - `timeout`: seconds to wait after terminate() before kill()
    - Returns: number of processes terminated

    Note: This detects processes started like `python -u /path/to/server.py`.
          If servers are launched via `python -m package.module`, there's no file path
          to match and they won't be found.
    """
    try:
        import psutil
    except ImportError:
        raise RuntimeError("psutil is required. Install with `pip install psutil`.")

    base = Path(servers_dir).resolve()
    me = os.getpid()
    killed = 0

    for proc in psutil.process_iter(["pid", "cmdline", "cwd"]):
        try:
            if proc.info["pid"] == me:
                continue

            cmd = proc.info.get("cmdline") or []
            if len(cmd) < 2:
                continue

            # Heuristic: is this a Python process?
            exe0 = os.path.basename(cmd[0]).lower()
            is_python = (
                "python" in exe0 or
                (sys.executable and Path(cmd[0]) == Path(sys.executable))
            )
            if not is_python:
                continue

            # Does any argument point to a script within servers_dir?
            found = False
            for arg in cmd[1:]:
                # Skip flags like -u, -m, etc.
                if arg.startswith("-"):
                    continue
                try:
                    p = Path(arg)
                    if not p.is_absolute():
                        # Try resolving relative to the process' cwd first
                        try:
                            ap = (Path(proc.info.get("cwd") or proc.cwd()) / p).resolve()
                        except Exception:
                            ap = p.resolve()
                    else:
                        ap = p.resolve()
                except Exception:
                    continue

                # Check tree membership
                if include_subdirs:
                    if ap == base or base in ap.parents:
                        found = True
                        break
                else:
                    if ap.parent == base:
                        found = True
                        break

            if not found:
                continue

            # Terminate the process
            try:
                proc.terminate()
                proc.wait(timeout=timeout)
                killed += 1
                if verbose:
                    print(f"swept PID {proc.pid}: {' '.join(cmd)}")
            except Exception:
                try:
                    proc.kill()
                    killed += 1
                    if verbose:
                        print(f"killed PID {proc.pid}: {' '.join(cmd)}")
                except Exception:
                    if verbose:
                        print(f"warning: could not kill PID {proc.pid}")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            if verbose:
                print("sweep warning:", e)

    if verbose:
        print(f"sweep complete: {killed} process(es) terminated")
    return