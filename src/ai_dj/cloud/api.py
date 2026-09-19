"""ASGI cloud API: no audio device, desktop process, or localhost dependency."""
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from urllib.parse import unquote
import uuid

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ai_dj.cloud.service import Library, plan
import json


class PlanRequest(BaseModel):
    current: str
    candidates: list[str] = Field(min_length=1, max_length=100)
    position: float = Field(default=0, ge=0, allow_inf_nan=False)
    rate: float = Field(default=1, ge=.92, le=1.08, allow_inf_nan=False)
    cues: dict[str, float] = {}
    point: float | None = Field(default=None, ge=0, allow_inf_nan=False)


def create_app():
    root = Path(os.environ.get("SYNC_DATA_DIR", "data/cloud")).resolve()
    @asynccontextmanager
    async def lifespan(app):
        app.state.library = Library(root, os.environ.get("SYNC_VOCALS", "0") == "1")
        yield
        app.state.library.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    origins = [o.strip() for o in os.environ.get("SYNC_FRONTEND_ORIGINS", "").split(',') if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "X-Filename"])

    def authorized(request: Request):
        header = request.headers.get('Authorization', '')
        if not header.startswith('Bearer ') or len(header) > 200:
            raise HTTPException(401, 'Browser session required')
        try:
            return request.app.state.library.scoped(header[7:])
        except KeyError:
            raise HTTPException(401, 'Browser session expired')

    @app.post('/api/session', status_code=201)
    def session(request: Request):
        return JSONResponse({'token': request.app.state.library.session()}, status_code=201,
                            headers={'Cache-Control': 'no-store'})

    @app.get("/health")
    def health():
        return {"ok": True, "playback": "browser"}

    @app.get("/api/library")
    def library(store=Depends(authorized)):
        return store.list()

    @app.post("/api/upload", status_code=202)
    async def upload(request: Request, store=Depends(authorized)):
        filename = unquote(request.headers.get("X-Filename", ""))
        suffix = Path(filename).suffix.lower()
        if not filename or '/' in filename or '\\' in filename or suffix not in {".mp3", ".wav", ".flac"}:
            raise HTTPException(400, "Choose an MP3, WAV or FLAC file")
        key = uuid.uuid4().hex
        source = root / (key + suffix)
        limit = 250 * 1024 * 1024
        size = 0
        try:
            with source.open("xb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, "File exceeds 250 MB")
                    await asyncio.to_thread(output.write, chunk)
            if not size:
                raise HTTPException(400, "Empty file")
            store.register(key, filename, source)
        except BaseException:
            source.unlink(missing_ok=True)
            raise
        return {"id": key, "status": "analyzing"}

    @app.get("/api/waveform/{key}")
    def waveform(key: str, store=Depends(authorized)):
        try:
            row = store.row(key)
        except KeyError:
            raise HTTPException(404, "Unknown track")
        if not row["peaks"]:
            return JSONResponse({"pending": True}, status_code=202)
        return json.loads(row["peaks"])

    @app.get("/api/audio/{key}")
    def audio(key: str, rate: float = 1, store=Depends(authorized)):
        if not .92 <= rate <= 1.08:
            raise HTTPException(400, "Playback rate out of range")
        try:
            path = store.prepare(key, round(rate, 6))
        except KeyError:
            raise HTTPException(404, "Unknown track")
        except ValueError as error:
            raise HTTPException(409, str(error))
        if not path:
            return JSONResponse({"pending": True}, status_code=202)
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/plan")
    async def transition(body: PlanRequest, store=Depends(authorized)):
        try:
            ids = set([body.current, *body.candidates])
            analyses = {key: json.loads(store.row(key)["analysis"]) for key in ids}
            if body.position >= analyses[body.current]["duration"]:
                raise ValueError("Current position is outside the track")
            if body.point is not None and not body.position <= body.point < analyses[body.current]["duration"]:
                raise ValueError("Choose a future exit within the track")
            for key, cue in body.cues.items():
                if key in analyses and not 0 <= cue < analyses[key]["duration"]:
                    raise ValueError("Cue is outside the track")
            future = store.pool.submit(plan, analyses, body.current, body.candidates,
                                       body.position, body.rate, body.cues, body.point)
            return await asyncio.wrap_future(future)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(400, str(error))

    assets = Path(os.environ.get("SYNC_FRONTEND_DIST", "frontend/dist"))
    if assets.is_dir():
        app.mount("/", StaticFiles(directory=assets, html=True), name="frontend")
    return app


app = create_app()
