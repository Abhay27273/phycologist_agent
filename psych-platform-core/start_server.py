"""
Startup wrapper that sets OpenMP env vars before any native libs load,
then hands off to uvicorn. Run with:
  python start_server.py
"""
import os
import sys

# Must be set before PyTorch/HuggingFace import any OpenMP/BLAS DLLs
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("app.api.server:app", host="127.0.0.1", port=port, log_level="info")
