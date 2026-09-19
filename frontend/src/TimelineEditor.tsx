import { useEffect, useRef, useState, type PointerEvent } from 'react';
import type { BrowserEngine } from './browser-engine';
import { waveform, type Track } from './local-library';
import type { TimelineClip } from './session-timeline';
const label = (seconds: number) => `${Math.floor(Math.max(0, seconds) / 60).toString().padStart(2, '0')}:${(Math.max(0, seconds) % 60).toFixed(1).padStart(4, '0')}`;
type Drag = { id: string; kind: 'move' | 'cue'; x: number; scroll: number; initial: number; value: number };
export function TimelineEditor({ engine, tracks }: { engine: BrowserEngine; tracks: Track[] }) {
  const [zoom, setZoom] = useState(8), [message, setMessage] = useState(''), [drag, setDrag] = useState<Drag | null>(null);
  const [peaks, setPeaks] = useState<Record<string, number[]>>({});
  const scroll = useRef<HTMLDivElement>(null);
  const ids = [...new Set(engine.timeline.map(c => c.id))].join(',');
  useEffect(() => { let live = true; for (const id of ids.split(',').filter(Boolean)) void waveform(id).then(data => { if (live) setPeaks(p => ({ ...p, [id]: data.peaks })); }); return () => { live = false; }; }, [ids]);
  const clips = engine.timeline;
  const duration = Math.max(90, ...clips.map(c => c.end)) + 15;
  const width = Math.max(900, 120 + duration * zoom), step = zoom < 6 ? 30 : zoom < 14 ? 10 : 5;
  const ticks = Array.from({ length: Math.ceil(duration / step) }, (_, i) => i * step);
  function begin(e: PointerEvent<HTMLButtonElement>, clip: TimelineClip, kind: 'move' | 'cue') {
    if (clip.past) return;
    e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId);
    const initial = kind === 'move' ? clip.start : clip.cue;
    setDrag({ id: clip.id, kind, x: e.clientX, scroll: scroll.current?.scrollLeft || 0, initial, value: initial }); setMessage('');
  }
  function move(e: PointerEvent<HTMLButtonElement>) {
    if (!drag) return;
    const viewport = scroll.current;
    if (viewport) { const box = viewport.getBoundingClientRect(); if (e.clientX > box.right - 35) viewport.scrollLeft += 15; if (e.clientX < box.left + 35) viewport.scrollLeft -= 15; }
    const delta = (e.clientX - drag.x + (viewport?.scrollLeft || 0) - drag.scroll) / zoom;
    setDrag({ ...drag, value: Math.max(0, Math.round((drag.initial + delta) * 10) / 10) });
  }
  function apply(id: string, value: number, kind: 'move' | 'cue') {
    try { engine.editTimeline(id, value, kind); setMessage('Manual edit applied. Later clips shift with the transition.'); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'Could not move clip.'); }
  }
  function end(e: PointerEvent<HTMLButtonElement>) { if (drag) { apply(drag.id, drag.value, drag.kind); setDrag(null); } if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId); }
  const [scrubbing, setScrubbing] = useState<number | null>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  function handlePlayheadDown(e: PointerEvent<HTMLDivElement>) {
    e.preventDefault(); e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    const canvas = e.currentTarget.closest('.session-canvas') as HTMLElement;
    if (!canvas) return;
    const box = canvas.getBoundingClientRect();
    const time = Math.max(0, (e.clientX - box.left - 120) / zoom);
    setScrubbing(time);
  }
  function handlePlayheadMove(e: PointerEvent<HTMLDivElement>) {
    if (scrubbing === null) return;
    const canvas = e.currentTarget.closest('.session-canvas') as HTMLElement;
    if (!canvas) return;
    const box = canvas.getBoundingClientRect();
    const time = Math.max(0, (e.clientX - box.left - 120) / zoom);
    setScrubbing(time);
  }
  function handlePlayheadUp(e: PointerEvent<HTMLDivElement>) {
    if (scrubbing !== null) {
      const time = scrubbing;
      setScrubbing(null);
      void engine.seekSession(time);
      setMessage(`Jumped to ${label(time)} · Re-hearing playback`);
    }
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  }
  function handleRulerClick(e: React.MouseEvent<HTMLDivElement>) {
    const canvas = e.currentTarget.closest('.session-canvas') as HTMLElement;
    if (!canvas) return;
    const box = canvas.getBoundingClientRect();
    const clickX = e.clientX - box.left;
    if (clickX < 120) return;
    const time = Math.max(0, (clickX - 120) / zoom);
    void engine.seekSession(time);
    setMessage(`Jumped to ${label(time)} · Re-hearing playback`);
  }
  const displayPosition = scrubbing !== null ? scrubbing : engine.sessionPosition;
  return <div className="session-editor">
    <div className="session-scroll" ref={scroll} tabIndex={0} aria-label="Session timeline, scroll horizontally">
      <div className="session-canvas" style={{ width, height: Math.max(240, 38 + clips.length * 100) }}>
        <div className="session-ruler" onClick={handleRulerClick} title="Click ruler to jump playhead"><span className="ruler-origin">TRACK / TIME</span>{ticks.map(t => <span key={t} style={{ left: 120 + t * zoom }}>{label(t)}</span>)}</div>
        {clips.map((clip, index) => {
          const track = tracks.find(t => t.id === clip.id), moving = drag?.id === clip.id && !clip.past ? drag : null;
          const start = moving?.kind === 'move' ? moving.value : clip.start;
          const cue = moving?.kind === 'cue' ? moving.value : clip.cue;
          const current = !clip.past && clip.id === engine.current?.id;
          const editable = !clip.past && (!current || !engine.playing);
          return <div className="session-lane" key={`${clip.id}-${clip.start}-${index}`} style={{ top: 38 + index * 100 }}>
            <span className="lane-name">{String(index + 1).padStart(2, '0')} {clip.past ? 'PLAYED' : current ? 'PLAYING' : 'UPCOMING'}</span>
            <article className={`session-clip ${current ? 'active' : ''} ${clip.past ? 'past' : ''} ${moving ? 'dragging' : ''}`} style={{ left: 120 + start * zoom, width: Math.max(12, (clip.end - clip.start) * zoom) }}>
              <button className="clip-title" aria-label={`Move ${track?.title || 'track'} on timeline`} disabled={clip.past || current} onPointerDown={e => begin(e, clip, 'move')} onPointerMove={move} onPointerUp={end} onPointerCancel={() => setDrag(null)} onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); apply(clip.id, clip.start + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 5 : 1), 'move'); } }}><b>{track?.title}</b><span>{label(start)} → {label(start + clip.end - clip.start)} · {clip.manual || moving ? 'MANUAL' : 'AUTO'}</span></button>
              <ClipWave peaks={peaks[clip.id] || []} cue={cue} length={clip.end - clip.start} duration={track?.duration || 1} />
              {clip.fade > 0 && <div className="clip-overlap" title={`Crossfade ${clip.fade.toFixed(2)} seconds`} style={{ width: Math.max(2, clip.fade * zoom) }} />}
              <button className="clip-cue" aria-label={`Adjust cue for ${track?.title || 'track'}`} disabled={!editable} onPointerDown={e => begin(e, clip, 'cue')} onPointerMove={move} onPointerUp={end} onPointerCancel={() => setDrag(null)} onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); apply(clip.id, clip.cue + (e.key === 'ArrowRight' ? .5 : -.5), 'cue'); } }}>IN {label(cue)}</button>
              <button className="clip-seek" aria-label={`Seek to position in ${track?.title || 'track'}`} onClick={e => { const box = e.currentTarget.getBoundingClientRect(); const trackSec = clip.cue + (e.clientX - box.left) / box.width * (clip.end - clip.start); const sessionSec = clip.start + (trackSec - clip.cue); void engine.seekSession(sessionSec); }} title="Click waveform to jump playhead here" />
            </article>
          </div>;
        })}
        {!clips.length && <p className="timeline-empty">Load a song to plan its timeline and the upcoming tracks.</p>}
        {!!clips.length && <div ref={playheadRef} className={`session-playhead ${scrubbing !== null ? 'scrubbing' : ''}`} style={{ left: 120 + Math.max(0, displayPosition) * zoom }} onPointerDown={handlePlayheadDown} onPointerMove={handlePlayheadMove} onPointerUp={handlePlayheadUp} onPointerCancel={() => setScrubbing(null)} title="Drag playhead line back or forward to scrub / re-hear completed transitions"><div className="playhead-handle" /><span>{label(displayPosition)}</span></div>}
      </div>
    </div>

  </div>;
}
function ClipWave({ peaks, cue, length, duration }: { peaks: number[]; cue: number; length: number; duration: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return; const draw = () => {
      const width = Math.max(1, Math.min(8192, Math.ceil(canvas.clientWidth))); canvas.width = width; canvas.height = 42;
      const ctx = canvas.getContext('2d')!; ctx.fillStyle = '#9fe1ce';
      for (let x = 0; x < width; x += 3) { let max = 0; const start = Math.floor((cue + x / width * length) / duration * peaks.length), end = Math.ceil((cue + (x + 3) / width * length) / duration * peaks.length); for (let i = start; i < Math.min(end, peaks.length); i++)max = Math.max(max, peaks[i] || 0); ctx.fillRect(x, 21 - max * 20, 2, Math.max(1, max * 40)); }
    }; draw(); const observer = new ResizeObserver(draw); observer.observe(canvas); return () => observer.disconnect();
  }, [peaks, cue, length, duration]);
  return <canvas ref={ref} className="clip-wave" aria-label="Audio waveform" />;
}
