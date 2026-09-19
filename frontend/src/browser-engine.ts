import {decodeFile, planTransition, sequenceTrackIds, type Track, type Transition} from './local-library';
import {buildTimeline,boundedStart,type TimelineClip} from './session-timeline';

type Deck = {id:string;buffer:AudioBuffer;rate:number;offset:number;at:number;node?:AudioBufferSourceNode;gain?:GainNode};

/** Two live decks on the device's audio clock; Files and analysis stay in this browser session. */
export class BrowserEngine {
  context: AudioContext | null = null;
  master: GainNode | null = null;
  tracks: Track[] = [];
  order: string[] = [];
  cues: Record<string, number> = {};
  points: Record<string, number> = {};
  current: Deck | null = null;
  incoming: Deck | null = null;
  event: Transition | null = null;
  prepared: {event:Transition;buffer:AudioBuffer} | null = null;
  playing = false;
  auto = true;
  manualOrder = false;
  volume = 1;
  preparing = false;
  error = '';
  history: string[] = [];
  heardFrom = 0;

  // Recording
  recorder: MediaRecorder | null = null;
  recordedChunks: Blob[] = [];
  recording = false;
  recordingStartTime = 0;
  recordingDuration = 0;
  recordedBlob: Blob | null = null;
  recDestination: MediaStreamAudioDestinationNode | null = null;
  sessionStart = 0;
  timeline: TimelineClip[] = [];
  completed: TimelineClip[] = [];
  get sessionPosition(){return this.sessionStart+this.position-this.heardFrom;}
  rebuildTimeline(){
    if(!this.current){this.timeline=[];return;}
    const index=this.order.indexOf(this.current.id);
    const remaining=[...this.order.slice(index+1),...this.order.slice(0,index)];
    this.timeline=[...this.completed,...buildTimeline(this.tracks,this.current.id,this.sessionStart,this.heardFrom,this.event,remaining,planTransition,this.cues,this.points,this.auto,this.manualOrder)];
  }
  editTimeline(id:string,value:number,kind:'move'|'cue'){
    const active=this.timeline.filter(c=>!c.past),index=active.findIndex(c=>c.id===id);
    if(index<0)throw Error('This clip is no longer editable.');
    const clip=active[index],track=this.tracks.find(t=>t.id===id)!;
    if(this.incoming&&this.audio().currentTime>=this.incoming.at)throw Error('Wait until the active crossfade finishes before editing.');
    if(kind==='move'){
      if(index===0)throw Error('Move an upcoming clip to change its start.');
      const previous=active[index-1],source=this.tracks.find(t=>t.id===previous.id)!;
      this.points[previous.id]=boundedStart(previous,source.audioEnd??source.duration,value,index===1?this.position:previous.cue+previous.fade);
    }else{
      if(index===0&&this.playing)throw Error('Pause to change the current song’s starting cue.');
      this.cues[id]=Math.max(0,Math.min((clip.exit??track.audioEnd??track.duration)-.5,value));
      if(index===0){this.heardFrom=this.cues[id];if(this.current)this.current.offset=this.cues[id];}
    }
    // Editing fixes the visible order, so a later automatic choice cannot replace this clip.
    const ids=active.map(c=>c.id);this.order=[...ids,...this.order.filter(x=>!ids.includes(x))];this.manualOrder=true;
    this.invalidate();void this.plan();
  }
  generation = 0;
  transitionEnd = Infinity;
  cache = new Map<string, AudioBuffer>();
  timer: ReturnType<typeof setInterval>;

  constructor() { this.timer = setInterval(() => this.tick(), 50); }
  audio() {
    if (!this.context) {
      this.context = new AudioContext();
      this.master = this.context.createGain();
      this.master.gain.value = this.volume;
      this.master.connect(this.context.destination);
      if (typeof this.context.createMediaStreamDestination === 'function') {
        try {
          this.recDestination = this.context.createMediaStreamDestination();
          this.master.connect(this.recDestination);
        } catch {}
      }
    }
    return this.context;
  }
  get position() {
    if (!this.current) return 0;
    return this.current.offset + (this.playing ? Math.max(0, this.audio().currentTime-this.current.at)*this.current.rate : 0);
  }
  updateLibrary(tracks: Track[]) {
    this.tracks = tracks.filter(t=>t.status==='ready');
    const readyIds = this.tracks.map(t => t.id);
    if (!this.manualOrder) {
      const sequencer = typeof sequenceTrackIds === 'function' ? sequenceTrackIds : (ids: string[]) => ids;
      this.order = sequencer(readyIds, this.current?.id || readyIds[0]);
    } else {
      for (const track of this.tracks) if (!this.order.includes(track.id)) this.order.push(track.id);
    }
    if(this.current&&!this.event&&!this.preparing&&this.auto&&this.tracks.length>1)void this.plan();
  }
  aiSequence() {
    const readyIds = this.tracks.map(t => t.id);
    if (readyIds.length <= 1) return;
    this.manualOrder = false;
    const sequencer = typeof sequenceTrackIds === 'function' ? sequenceTrackIds : (ids: string[]) => ids;
    this.order = sequencer(readyIds, this.current?.id || readyIds[0]);
    this.invalidate();
    void this.plan();
  }
  async buffer(id: string, rate=1): Promise<AudioBuffer> {
    const key = `${id}:${rate.toFixed(6)}`;
    const existing = this.cache.get(key);
    if (existing) return existing;
    const decoded = await decodeFile(id,this.audio());
    this.cache.set(key,decoded);
    while(this.cache.size>2)this.cache.delete(this.cache.keys().next().value!);
    return decoded;
  }

  halt(deck: Deck|null) {
    if (!deck) return;
    if(deck.node) { deck.node.onended=null; try {deck.node.stop();} catch {} deck.node.disconnect(); }
    deck.gain?.disconnect(); deck.node=undefined;deck.gain=undefined;
  }
  invalidate() {
    this.tick();
    const position=this.position;
    this.generation++;
    this.preparing=false;
    this.halt(this.current);this.halt(this.incoming);
    this.incoming=null;this.event=null;this.prepared=null;this.transitionEnd=Infinity;
    if(this.current) {this.current.offset=position;this.current.at=this.audio().currentTime;}
    if(this.playing&&this.current)this.start(this.current,this.audio().currentTime);
  }
  start(deck: Deck, at: number, level=1) {
    const ctx=this.audio(), node=ctx.createBufferSource(), gain=ctx.createGain();
    node.buffer=deck.buffer;node.connect(gain);gain.connect(this.master!);
    gain.gain.setValueAtTime(level,at);
    deck.node=node;deck.gain=gain;deck.at=at;
    node.start(at,Math.min(deck.buffer.duration-.001,deck.offset/deck.rate));
    node.onended=()=>{if(this.current===deck&&!this.incoming){this.playing=false;deck.offset=deck.buffer.duration*deck.rate;}};
  }
  async load(id: string, play=this.playing) {
    if(!this.tracks.some(track=>track.id===id))throw Error('Choose an analyzed track first.');
    const ctx=this.audio(); if(play)await ctx.resume();
    const generation=++this.generation;
    const buffer=await this.buffer(id);
    if(generation!==this.generation)return;
    if(this.current&&this.current.id!==id)this.history.push(this.current.id);
    this.halt(this.current);this.halt(this.incoming);
    this.incoming=null;this.event=null;this.prepared=null;this.transitionEnd=Infinity;
    this.current={id,buffer,rate:1,offset:this.cues[id]||0,at:ctx.currentTime};
    this.heardFrom=this.current.offset;
    this.sessionStart=0;this.completed=[];this.rebuildTimeline();
    this.playing=play;
    if(play)this.start(this.current,ctx.currentTime);
    void this.plan();
  }
  async play() {
    await this.audio().resume(); // Called directly from a user gesture.
    if(!this.current) {if(this.order[0])await this.load(this.order[0],true);return;}
    if(this.playing)return;
    if(this.current.offset>=this.current.buffer.duration*this.current.rate)this.current.offset=0;
    this.playing=true;this.start(this.current,this.audio().currentTime);
    if(this.prepared)this.arm(this.prepared.event,this.prepared.buffer);else void this.plan();
  }
  pause() { const position=this.position;this.playing=false;this.invalidate();if(this.current)this.current.offset=position;void this.plan(); }
  stop() { this.pause();if(this.current)this.current.offset=0; }
  seek(seconds: number) {
    if(!this.current)return;
    const playing=this.playing;this.pause();
    this.current.offset=Math.max(0,Math.min(seconds,this.current.buffer.duration*this.current.rate-.01));
    if(playing)void this.play();else void this.plan();
  }
  async seekSession(targetTime: number) {
    if (!this.timeline.length) return;
    const ctx = this.audio();
    if (this.playing) await ctx.resume();
    const clamped = Math.max(0, targetTime);
    let foundIndex = -1;
    for (let i = 0; i < this.timeline.length; i++) {
      const clip = this.timeline[i];
      if (clamped >= clip.start && clamped <= clip.end) {
        foundIndex = i;
        break;
      }
    }
    if (foundIndex < 0) {
      if (clamped < this.timeline[0].start) foundIndex = 0;
      else foundIndex = this.timeline.length - 1;
    }

    const clip = this.timeline[foundIndex];
    const nextClip = this.timeline[foundIndex + 1];
    const xfadeStart = clip.exit !== undefined ? clip.start + (clip.exit - clip.cue) : clip.end;
    const inCrossfade = !!(nextClip && clip.fade > 0 && clamped >= xfadeStart && clamped <= clip.end);

    const generation = ++this.generation;
    this.preparing = false;
    this.halt(this.current);
    this.halt(this.incoming);
    this.incoming = null;
    this.event = null;
    this.prepared = null;
    this.transitionEnd = Infinity;

    this.completed = this.timeline.slice(0, foundIndex).map(c => ({ ...c, past: true }));
    this.sessionStart = clip.start;
    this.heardFrom = clip.cue;

    if (inCrossfade && nextClip) {
      const outBuffer = await this.buffer(clip.id);
      const inBuffer = await this.buffer(nextClip.id);
      if (generation !== this.generation) return;

      const outOffset = clip.cue + (clamped - clip.start);
      const inOffset = nextClip.cue + (clamped - xfadeStart);
      const remainingXfade = Math.max(0.01, clip.end - clamped);
      const progress = Math.min(0.99, Math.max(0, (clamped - xfadeStart) / Math.max(0.01, clip.fade)));

      const outLevel = Math.cos(progress * Math.PI / 2);
      const inLevel = Math.sin(progress * Math.PI / 2);

      const now = ctx.currentTime;
      const outgoingDeck: Deck = { id: clip.id, buffer: outBuffer, rate: 1, offset: outOffset, at: now };
      const incomingDeck: Deck = { id: nextClip.id, buffer: inBuffer, rate: 1, offset: inOffset, at: now };

      this.current = outgoingDeck;
      this.incoming = incomingDeck;
      this.transitionEnd = now + remainingXfade;

      if (this.playing) {
        this.start(outgoingDeck, now, outLevel);
        this.start(incomingDeck, now, inLevel);

        const count = 128;
        const fadeOut = new Float32Array(count);
        const fadeIn = new Float32Array(count);
        for (let i = 0; i < count; i++) {
          const frac = progress + (1 - progress) * (i / (count - 1));
          fadeOut[i] = Math.cos(frac * Math.PI / 2);
          fadeIn[i] = Math.sin(frac * Math.PI / 2);
        }
        outgoingDeck.gain?.gain.setValueCurveAtTime(fadeOut, now, remainingXfade);
        incomingDeck.gain?.gain.setValueCurveAtTime(fadeIn, now, remainingXfade);
        outgoingDeck.node?.stop(this.transitionEnd);
      }
      this.rebuildTimeline();
    } else {
      const buffer = await this.buffer(clip.id);
      if (generation !== this.generation) return;

      const offset = Math.max(0, clip.cue + (clamped - clip.start));
      const now = ctx.currentTime;
      this.current = { id: clip.id, buffer, rate: 1, offset, at: now };
      this.heardFrom = clip.cue;
      this.rebuildTimeline();

      if (this.playing) {
        this.start(this.current, now);
      }
      void this.plan();
    }
  }
  next(id?: string) {
    const index=this.order.indexOf(this.current?.id||'');
    return this.load(id||this.event?.next_track||this.order[(index+1)%this.order.length],this.playing);
  }
  previous() {const id=this.history.pop();if(id)return this.load(id,this.playing);}
  setVolume(value:number){this.volume=value;if(this.master)this.master.gain.setTargetAtTime(value,this.audio().currentTime,.02);}
  setAuto(value:boolean){this.auto=value;this.invalidate();if(value)void this.plan();}
  setCue(id:string,seconds:number){this.cues[id]=seconds;this.invalidate();if(this.current?.id===id&&!this.playing)this.current.offset=seconds;void this.plan();}
  setPoint(seconds:number){if(!this.current)return;this.points[this.current.id]=seconds;this.invalidate();void this.plan();}
  reorder(ids:string[]){this.order=ids;this.manualOrder=true;this.invalidate();void this.plan();}
  async plan() {
    if(!this.current||(!this.auto&&this.points[this.current.id]===undefined)){this.rebuildTimeline();return;}
    const current=this.current;
    const index=this.order.indexOf(current.id);
    let candidates = index >= 0 ? this.order.slice(index + 1) : [];
    if (!candidates.length && this.order.length > 1) {
      candidates = [this.order[0]];
    }
    if(!candidates.length){this.rebuildTimeline();return;}
    candidates = candidates.slice(0, 1);
    const generation=++this.generation;
    this.preparing=true;this.error='';
    try {
      let event:Transition=planTransition(current.id,candidates,this.position,this.cues,this.points[current.id],this.heardFrom);
      if(generation!==this.generation)return;
      const buffer=await this.buffer(event.next_track,event.tempo_ratio);
      if(generation!==this.generation)return;
      if(event.outgoing_transition_timestamp<this.position+.1){event=planTransition(current.id,[event.next_track],this.position,this.cues,this.points[current.id],this.heardFrom);}
      this.prepared={event,buffer};this.event=event;
      this.rebuildTimeline();
      if(this.playing)this.arm(event,buffer);
    } catch(error) {if(generation===this.generation)this.error=error instanceof Error?error.message:'Planning failed';}
    finally{if(generation===this.generation)this.preparing=false;}
  }
  arm(event:Transition,buffer:AudioBuffer) {
    if(!this.current||!this.playing)return;
    const ctx=this.audio(), outgoing=this.current;
    const at=ctx.currentTime+(event.outgoing_transition_timestamp-this.position)/outgoing.rate;
    if(at<ctx.currentTime)return;
    const incoming:Deck={id:event.next_track,buffer,rate:event.tempo_ratio,offset:event.incoming_start_timestamp,at};
    this.incoming=incoming;this.event=event;
    this.start(incoming,at,0);
    const count=256, fadeIn=new Float32Array(count),fadeOut=new Float32Array(count);
    for(let i=0;i<count;i++){fadeIn[i]=Math.sin(i/(count-1)*Math.PI/2);fadeOut[i]=Math.cos(i/(count-1)*Math.PI/2);}
    outgoing.gain!.gain.setValueCurveAtTime(fadeOut,at,event.transition_duration);
    incoming.gain!.gain.setValueCurveAtTime(fadeIn,at,event.transition_duration);
    this.transitionEnd=at+event.transition_duration;
    outgoing.node!.onended=null;outgoing.node!.stop(this.transitionEnd);
  }
  tick() {
    if (this.recording) {
      this.recordingDuration = (Date.now() - this.recordingStartTime) / 1000;
    }
    if(!this.playing||!this.incoming||!this.context||this.context.currentTime<this.transitionEnd)return;
    if(this.current)this.history.push(this.current.id);
    const outgoing=this.timeline.find(c=>!c.past&&c.id===this.current?.id);
    if(outgoing)this.completed.push({...outgoing,past:true});
    if(this.event)this.sessionStart+=(this.event.outgoing_transition_timestamp-this.heardFrom);
    this.halt(this.current);this.current=this.incoming;this.heardFrom=this.current.offset;this.incoming=null;
    this.event=null;this.prepared=null;this.transitionEnd=Infinity;this.generation++;
    void this.plan();
  }
  startRecording() {
    const ctx = this.audio();
    if (!this.recDestination && typeof ctx.createMediaStreamDestination === 'function') {
      try {
        this.recDestination = ctx.createMediaStreamDestination();
        this.master?.connect(this.recDestination);
      } catch {}
    }
    if (!this.recDestination || typeof MediaRecorder === 'undefined') {
      throw Error('Audio recording is not supported in this browser environment.');
    }
    let mimeType = 'audio/webm;codecs=opus';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      if (MediaRecorder.isTypeSupported('audio/webm')) mimeType = 'audio/webm';
      else if (MediaRecorder.isTypeSupported('audio/mp4')) mimeType = 'audio/mp4';
      else if (MediaRecorder.isTypeSupported('audio/ogg')) mimeType = 'audio/ogg';
      else mimeType = '';
    }
    this.recordedChunks = [];
    this.recordedBlob = null;
    const options = mimeType ? { mimeType } : undefined;
    this.recorder = new MediaRecorder(this.recDestination.stream, options);
    this.recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) this.recordedChunks.push(e.data);
    };
    this.recorder.onstop = () => {
      const type = this.recorder?.mimeType || 'audio/webm';
      this.recordedBlob = new Blob(this.recordedChunks, { type });
      this.recording = false;
    };
    this.recorder.start(100);
    this.recording = true;
    this.recordingStartTime = Date.now();
    this.recordingDuration = 0;
  }
  stopRecording() {
    if (this.recorder && this.recorder.state !== 'inactive') {
      this.recorder.stop();
    }
    this.recording = false;
  }
  downloadRecording() {
    if (!this.recordedBlob || typeof document === 'undefined') return;
    const isMp4 = this.recordedBlob.type.includes('mp4');
    const ext = isMp4 ? 'm4a' : 'webm';
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `SYNC-Live-Mix-${timestamp}.${ext}`;
    const url = URL.createObjectURL(this.recordedBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 15000);
  }
  dispose(){clearInterval(this.timer);this.halt(this.current);this.halt(this.incoming);void this.context?.close();}
}
