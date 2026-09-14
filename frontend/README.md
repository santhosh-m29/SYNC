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
