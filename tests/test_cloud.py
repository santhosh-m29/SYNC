"""Hosted API checks: real upload, analysis, persistent library and private audio."""
from io import BytesIO
import time
import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from ai_dj.cloud.api import create_app


def test_cloud_upload_analysis_playable_audio_and_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNC_DATA_DIR", str(tmp_path))
    samples=np.sin(np.arange(22050*8)*2*np.pi*220/22050)*.2
    stream=BytesIO()
    sf.write(stream,samples,22050,format="WAV")
    with TestClient(create_app()) as client:
        token=client.post('/api/session').json()['token']
        headers={'Authorization':'Bearer '+token}
        other={'Authorization':'Bearer '+client.post('/api/session').json()['token']}
        assert client.get('/health').status_code==200
        assert client.get('/api/library').status_code==401
        assert client.post('/api/upload',content=stream.getvalue(),headers={"X-Filename":"song.wav"}).status_code==401
        assert client.post('/api/upload',content=b'x',headers={**headers,"X-Filename":"../escape.wav"}).status_code==400
        response=client.post('/api/upload',content=stream.getvalue(),headers={**headers,"X-Filename":"song.wav"})
        assert response.status_code==202
        key=response.json()['id']
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            rows=client.get('/api/library',headers=headers).json()
            if rows[0]['status']!='analyzing':break
            time.sleep(.1)
        assert client.get('/api/library',headers=other).json()==[]
        for route in ['audio','waveform']:
            assert client.get(f'/api/{route}/{key}',headers=other).status_code==404
        assert client.post('/api/plan',headers=other,json={'current':key,'candidates':[key]}).status_code==400
        assert rows[0]['status']=='ready',rows
        assert rows[0]['duration']==pytest.approx(8,abs=.1)
        peaks=client.get(f'/api/waveform/{key}',headers=headers).json()
        assert max(peaks['peaks'])>.1
        assert client.get(f'/api/audio/{key}').status_code==401
        audio=client.get(f'/api/audio/{key}',headers=headers)
        decoded,sr=sf.read(BytesIO(audio.content))
        assert sr==44100 and len(decoded)>300000
        assert np.max(np.abs(decoded))>0
        assert client.get(f'/api/audio/{key}?rate=nan',headers=headers).status_code==400
    with TestClient(create_app()) as client:
        assert client.get('/api/library',headers=headers).json()[0]['id']==key


def test_sessions_need_no_shared_key_and_legacy_library_is_not_exposed(tmp_path, monkeypatch):
    monkeypatch.setenv('SYNC_DATA_DIR',str(tmp_path))
    monkeypatch.delenv('SYNC_ACCESS_TOKEN',raising=False)
    import sqlite3
    # Old ownerless rows stay private, rather than becoming visible to every visitor.
    with sqlite3.connect(tmp_path/'library.sqlite') as db:
        db.execute('CREATE TABLE tracks (id TEXT PRIMARY KEY, filename TEXT, source TEXT, status TEXT, error TEXT, analysis TEXT, peaks TEXT, audio TEXT)')
        db.execute("INSERT INTO tracks(id,status) VALUES('legacy','error')")
    with TestClient(create_app()) as client:
        response=client.post('/api/session')
        assert response.status_code==201
        assert response.headers['cache-control']=='no-store'
        headers={'Authorization':'Bearer '+response.json()['token']}
        assert client.get('/api/library',headers=headers).json()==[]
        assert client.get('/api/audio/legacy',headers=headers).status_code==404
        assert client.get('/api/library',headers={'Authorization':'Bearer invented'}).status_code==401
