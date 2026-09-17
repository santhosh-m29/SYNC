# SYNC web frontend

This directory is the independently deployable Vite/React frontend. It does not
contain the Python audio engine and cannot access the user's `music/` directory.

For Vercel, set the project **Root Directory** to `frontend` and use the existing
`frontend/vercel.json`. The committed root `vercel.json` also supports deployments
started from the repository root by building only this directory.

The local engine URL is configured with `VITE_ENGINE_URL`; it defaults to
`http://127.0.0.1:8765`. A Vercel page can therefore render normally while showing
the disconnected state until the user starts `python -m ai_dj ui` on the playback
computer. The browser never sends music files to Vercel.

The header's **+ Add Songs** button imports multiple MP3, WAV or FLAC files into
the local engine's library (250 MB per file). Analysis status appears beneath the
header. When ready, select the song, **Load track**, then **Play**.

For a deployed website, allow its exact origin before starting the local engine:

```powershell
$env:SYNC_FRONTEND_ORIGINS = "https://your-sync-site.vercel.app"
.\.venv\Scripts\python.exe -m ai_dj ui
```

Local Vite development origins on port 5173 are allowed by default. Multiple
origins can be comma-separated. If the browser requests permission to connect to
the local network, allow it for your SYNC website. An offline or blocked local
engine produces an import error; songs are never silently uploaded to Vercel.
