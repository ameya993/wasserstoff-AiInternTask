from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
import os

from app.api.routes import router  # Adjust import if needed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register your API router
app.include_router(router)

# Path resolution:
current_dir = Path(__file__).parent      # backend/app
backend_dir = current_dir.parent         # backend
frontend_dir = backend_dir / "frontend"  # backend/frontend

if not frontend_dir.exists():
    raise RuntimeError(f"Frontend directory not found at {frontend_dir}")

# Mount frontend folder at root
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

# Optional explicit routes (not necessary if mounted at "/")
@app.get("/app.js")
async def get_js():
    js_path = frontend_dir / "app.js"
    if not js_path.exists():
        return Response(status_code=404)
    return FileResponse(js_path, media_type="application/javascript")

@app.get("/style.css")
async def get_css():
    css_path = frontend_dir / "style.css"
    if not css_path.exists():
        return Response(status_code=404)
    return FileResponse(css_path, media_type="text/css")

# Support HEAD requests on root (health checks)
@app.api_route("/", methods=["GET", "HEAD"])
async def read_root(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse(content="index.html not found", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Run app locally
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
