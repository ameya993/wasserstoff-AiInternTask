# Copyright (c) 2025 Ameya Gawande. All rights reserved.
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You may not use this file except in compliance with the License.
# Non-commercial use only. Commercial use is strictly prohibited without written permission.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import os
from app.api.routes import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register your router
app.include_router(router)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get to the project root
project_root = os.path.dirname(os.path.dirname(current_dir))
# Path to the frontend directory
frontend_dir = os.path.join(project_root, "frontend")

# Serve static files (app.js and style.css)
@app.get("/app.js")
async def get_js():
    js_path = os.path.join(frontend_dir, "app.js")
    return FileResponse(js_path, media_type="application/javascript")

@app.get("/style.css")
async def get_css():
    css_path = os.path.join(frontend_dir, "style.css")
    return FileResponse(css_path, media_type="text/css")

# Serve the index.html at the root
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(frontend_dir, "index.html")
    with open(index_path, "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

# For Vercel deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
