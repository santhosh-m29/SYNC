import { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { listTracks, waveform, importSong, type Track } from './local-library';
import { BrowserEngine } from './browser-engine';
import { TimelineEditor } from './TimelineEditor';
import './style.css';

const time = (n: number) => `${Math.floor(Math.max(0, n) / 60).toString().padStart(2, '0')}:${Math.floor(Math.max(0, n) % 60).toString().padStart(2, '0')}`;
function App() {
    const [engine] = useState(() => new BrowserEngine());
    const [tracks, setTracks] = useState<Track[]>([]), [selected, setSelected] = useState('');
    const [notice, setNotice] = useState('Ready — add songs from your device'), [connected, setConnected] = useState(true);
    const [uploading, setUploading] = useState(false), [uploadNotice, setUploadNotice] = useState('');
    const [wave, setWave] = useState<number[]>([]), [cue, setCue] = useState(0);
    const [search, setSearch] = useState('');
    const [exit, setExit] = useState(0), [incomingCue, setIncomingCue] = useState(0);
    const [, render] = useState(0); const files = useRef<HTMLInputElement>(null);
    const current = tracks.find(t => t.id === engine.current?.id), chosen = tracks.find(t => t.id === selected);
    async function refresh() {
        try { const result: Track[] = listTracks(); setTracks(result); engine.updateLibrary(result); setConnected(true); setNotice('On-device library ready'); }
        catch (error) { setConnected(false); setNotice(error instanceof Error ? error.message : 'Could not read this session’s library'); }
    }
    useEffect(() => { void refresh(); const poll = setInterval(refresh, 2500), clock = setInterval(() => render(n => n + 1), 50); return () => { clearInterval(poll); clearInterval(clock); engine.dispose(); }; }, []);
    useEffect(() => { if (!selected) return; let live = true; setWave([]); setCue(engine.cues[selected] || 0); waveform(selected).then(data => { if (live) setWave(data.peaks || []) }).catch(() => { }); return () => { live = false; }; }, [selected]);

    async function perform(action: () => void | Promise<void>) { try { await action(); } catch (error) { setNotice(error instanceof Error ? error.message : 'Playback failed'); } }
    async function upload(list: File[]) {
        setUploading(true); const errors: string[] = []; let successCount = 0;
        for (let i = 0; i < list.length; i++) {
            const file = list[i];
            if (!/\.(mp3|wav|flac)$/i.test(file.name) || !file.size || file.size > 250 * 1024 * 1024) { errors.push(`${file.name}: choose MP3/WAV/FLAC under 250 MB.`); continue; }
            setUploadNotice(`Analyzing ${file.name} (${i + 1}/${list.length})…`);
            try { const pending = importSong(file); void refresh(); await pending; successCount++; }
            catch (error) { errors.push(`${file.name}: ${error instanceof Error ? error.message : 'Import failed'}`); }
        }
        setUploading(false);
        if (errors.length) { setUploadNotice(errors.join('. ')); }
        else { setUploadNotice(successCount > 0 ? `Added ${successCount} track${successCount > 1 ? 's' : ''} to DJ library.` : ''); setTimeout(() => setUploadNotice(''), 4000); }
        void refresh();
    }
    const event = engine.event, position = engine.position;
    return <main>
        <header><h1>SYNC <small>LIVE DJ</small></h1><div className="header-actions"><span className={connected ? 'connected' : ''}>{connected ? '● Your library' : notice}</span>{!engine.recording ? <button className="record-btn" title="Record live master audio mix" onClick={() => { try { engine.startRecording(); } catch (e) { setNotice(e instanceof Error ? e.message : 'Recording failed'); } }}>● REC MIX</button> : <button className="record-btn recording" title="Click to stop recording" onClick={() => engine.stopRecording()}><span className="rec-dot" /> STOP ({time(engine.recordingDuration)})</button>}{!!engine.recordedBlob && <button className="download-btn" title="Download your recorded DJ mix" onClick={() => engine.downloadRecording()}>⬇ Download Mix</button>}<button className="add-songs" disabled={uploading} onClick={() => files.current?.click()}>{uploading ? 'Importing…' : '+ Add Songs'}</button><input ref={files} type="file" accept=".mp3,.wav,.flac" multiple hidden onChange={e => { void upload(Array.from(e.target.files || [])); e.target.value = ''; }} /></div></header>
        {!connected && <div className="connection-notice" role="status">{notice} <button onClick={() => void refresh()}>Retry connection</button></div>}
        <div className="import-status" role="status">{uploadNotice && <span>{uploadNotice}</span>}{tracks.filter(t => t.status === 'error').map(t => <div key={t.id}>{t.filename} — {t.error}</div>)}{connected && !tracks.length && <p>Add songs from your device, then select a track and press Play. Audio plays on this device.</p>}</div>
        <section className="player"><div className="track-heading"><span className="artwork" aria-hidden="true">♫</span><div><label>NOW PLAYING</label><h2>{current?.title || 'Choose a track to begin'}</h2><p>{current?.filename || 'Live audio on this device'}</p></div></div><div className="transport"><button aria-label="Previous" onClick={() => void perform(() => engine.previous())}>Ⅰ◀</button><button className="play" aria-label={engine.playing ? 'Pause' : 'Play'} onClick={() => void perform(() => engine.playing ? engine.pause() : engine.play())}>{engine.playing ? 'Ⅱ' : '▶'}</button><button aria-label="Next" disabled={!engine.order.length} onClick={() => void perform(() => engine.next())}>▶Ⅰ</button><button aria-label="Stop" onClick={() => engine.stop()}>■</button></div><label className="volume">VOL <input aria-label="Volume" type="range" min="0" max="1" step=".01" value={engine.volume} onChange={e => engine.setVolume(+e.target.value)} /><strong>{Math.round(engine.volume * 100)}%</strong></label><div className="seek"><time>{time(position)}</time><input aria-label="Seek" type="range" min="0" max={current?.duration || 1} step=".01" value={position} onChange={e => engine.seek(+e.target.value)} /><time>−{time((current?.duration || 0) - position)}</time></div></section>
        <section className="timeline"><div className="section-title"><h2>∿ Timeline</h2><span>{event ? `${event.strategy.startsWith('manual_') ? 'MANUAL' : 'AUTO'} · EXIT ${time(event.outgoing_transition_timestamp)} · CUE ${time(event.incoming_start_timestamp)} · ${event.transition_duration.toFixed(2)}s` : engine.preparing ? 'Preparing next deck…' : engine.auto ? 'AUTO' : 'MANUAL'}</span><button onClick={() => engine.setAuto(!engine.auto)}>AUTO DJ · {engine.auto ? 'ON' : 'OFF'}</button></div>
            <TimelineEditor engine={engine} tracks={tracks} />
            <div className="manual-fields"><label>EXIT (seconds) <input type="number" min={position} max={current?.duration || 0} step=".01" value={exit} onChange={e => setExit(+e.target.value)} /></label><button disabled={!current} onClick={() => engine.setPoint(exit)}>Set exit</button><label>INCOMING CUE <input type="number" min="0" step=".01" value={incomingCue} onChange={e => setIncomingCue(+e.target.value)} /></label><button disabled={!event} onClick={() => event && engine.setCue(event.next_track, incomingCue)}>Set incoming cue</button></div>
        </section>
        <div className="columns"><section className="library-panel"><div className="section-title"><h2>♫ Library <small>{tracks.length}</small></h2><input className="search" aria-label="Search songs" placeholder="Search songs…" value={search} onChange={e => setSearch(e.target.value)} /></div><div className="track-list">{tracks.filter(t => `${t.title} ${t.filename}`.toLowerCase().includes(search.toLowerCase())).map(t => <div className={`row ${selected === t.id ? 'selected' : ''}`} key={t.id}><span className="artwork small" aria-hidden="true">♫</span><button disabled={t.status !== 'ready'} onClick={() => setSelected(t.id)}>{t.title}</button><span>{t.status === 'ready' ? `${(t.bpm ? t.bpm.toFixed(1) + ' BPM' : 'Tempo uncertain')} · ${t.key || '—'}` : t.status}</span><time>{time(t.duration)}</time><button aria-label={'Load ' + t.title} disabled={t.status !== 'ready'} onClick={() => void perform(() => engine.load(t.id))}>▶</button></div>)}</div></section><section className="queue-panel"><div className="section-title"><h2>≡ Queue <small>{engine.order.length}</small></h2><div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}><button className="ai-order-btn" title="Re-order queue using AI DJ analysis for optimal set flow" onClick={() => engine.aiSequence()}>✨ AI Sequence</button><span>Drag to reorder</span></div></div><div className="track-list">{engine.order.map((id, i) => <div className={`row queue ${id === current?.id ? 'selected' : ''}`} key={id} draggable onDragStart={e => e.dataTransfer.setData('text/plain', id)} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); const moved = e.dataTransfer.getData('text/plain'); if (!engine.order.includes(moved) || moved === id) return; const order = engine.order.filter(t => t !== moved); order.splice(order.indexOf(id), 0, moved); engine.reorder(order); }}><small>{String(i + 1).padStart(2, '0')}</small><span className="artwork small" aria-hidden="true">♫</span><b>{tracks.find(t => t.id === id)?.title}</b><time>{time(tracks.find(t => t.id === id)?.duration || 0)}</time><button aria-label={'Skip to ' + id} onClick={() => void perform(() => engine.next(id))}>▶</button></div>)}</div></section></div>
        <section className="cue"><div className="section-title"><div><label>CUE EDITOR</label><h2>{chosen?.title || 'Select a track'}</h2></div><button disabled={!selected} onClick={() => void perform(() => engine.load(selected))}>Load track</button></div><Wave peaks={wave} position={chosen ? cue / chosen.duration : undefined} label={chosen?.filename || 'Select a track'} onSeek={fraction => { if (chosen) { const point = fraction * chosen.duration; setCue(point); engine.setCue(chosen.id, point); } }} /><p>Cue {time(cue)} · Click waveform to choose the start position</p></section>
        <footer role="status">{engine.error || notice}</footer>
    </main>;
}
function Wave({ peaks, position, stamp, marker, label, onSeek }: { peaks: number[]; position?: number; stamp?: string; marker?: number; label: string; onSeek?: (fraction: number) => void }) {
    const ref = useRef<HTMLCanvasElement>(null);
    useEffect(() => { const canvas = ref.current; if (!canvas) return; const draw = () => { const width = canvas.clientWidth; canvas.width = Math.round(width * devicePixelRatio); canvas.height = 80 * devicePixelRatio; const ctx = canvas.getContext('2d')!; ctx.scale(devicePixelRatio, devicePixelRatio); ctx.fillStyle = canvas.closest('.incoming') ? '#bc9cee' : '#a5e9d7'; for (let x = 0; x < width; x += 5) { const start = Math.floor(x / width * peaks.length), end = Math.ceil((x + 5) / width * peaks.length); let peak = 0; for (let i = start; i < end; i++)peak = Math.max(peak, peaks[i] || 0); ctx.globalAlpha = position === undefined || x / width <= position ? 1 : .35; ctx.fillRect(x, 40 - peak * 32, 3, Math.max(2, peak * 64)); } ctx.globalAlpha = 1; if (marker !== undefined) { ctx.strokeStyle = '#c4a4f5'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(marker * width, 0); ctx.lineTo(marker * width, 80); ctx.stroke(); ctx.setLineDash([]); } if (position !== undefined) { ctx.fillStyle = '#fff'; ctx.fillRect(position * width, 0, 1, 80); } }; draw(); const observer = new ResizeObserver(draw); observer.observe(canvas); return () => observer.disconnect(); }, [peaks, position, marker, label]);
    return <div className="wave-container">{stamp && position !== undefined && <span className="wave-stamp" style={{ left: `${Math.max(6, Math.min(94, position * 100))}%` }}>{stamp}</span>}<canvas className="audio-wave" ref={ref} aria-label={label + ' waveform'} onClick={e => { const rect = e.currentTarget.getBoundingClientRect(); onSeek?.(Math.max(0, Math.min(.999, (e.clientX - rect.left) / rect.width))); }} /></div>;
}
createRoot(document.getElementById('root')!).render(<App />);
