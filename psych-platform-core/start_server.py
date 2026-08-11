"""
Startup wrapper that sets OpenMP env vars before any native libs load,
then hands off to uvicorn. Run with:
  python start_server.py
"""
import asyncio
import os
import sys

# Must be set before PyTorch/HuggingFace import any OpenMP/BLAS DLLs
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Force uvicorn onto SelectorEventLoop on Windows.
#
# This is NOT a policy-based fix (asyncio.set_event_loop_policy has no
# effect here — see below) — it patches uvicorn's own loop factory directly.
# uvicorn >=0.36 creates its loop via
#   asyncio.run(..., loop_factory=config.get_loop_factory())
# and Python's loop_factory parameter constructs the loop directly,
# completely bypassing the process-wide event loop policy regardless of what
# it's set to. uvicorn.loops.asyncio.asyncio_loop_factory() then
# unconditionally returns asyncio.ProactorEventLoop on win32 (verified by
# reading its source directly) — a deliberate, hardcoded choice with no
# exposed uvicorn config flag to change it, presumably because Proactor
# supports subprocess handling that Selector doesn't.
#
# The problem: psycopg's async mode hard-refuses to run under
# ProactorEventLoop at all. This was diagnosed live while migrating the
# LangGraph Postgres checkpointer — the failure didn't surface directly; it
# was buried inside a connection pool's background retry logging, so it
# looked like a silent 30s startup hang instead of the real, immediate
# "cannot use ProactorEventLoop" error underneath.
if sys.platform == "win32":
    import uvicorn.loops.asyncio as _uvicorn_asyncio_loop

    _uvicorn_asyncio_loop.asyncio_loop_factory = (
        lambda use_subprocess=False: asyncio.SelectorEventLoop
    )

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("app.api.server:app", host="127.0.0.1", port=port, log_level="info")
