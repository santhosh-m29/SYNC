export type Section = {time:number;energy:number;tone:number;rhythm:number;change:number;repeat:number};
export type Analysis = {duration:number;bpm:number;confidence:number;start:number;end:number;peaks:number[];sections:Section[]};

/** Lightweight signal analysis. No vocal-separation or key-certainty claims. */
export function analyze(samples:Float32Array,sr:number):Analysis {
  const hop=Math.max(1,Math.round(sr*.02)), energy:number[]=[], tone:number[]=[], peaks:number[]=[];
  for(let i=0;i<samples.length;i+=hop){
    let power=0,difference=0;
    for(let j=i;j<Math.min(i+hop,samples.length);j++){power+=samples[j]**2;if(j)difference+=(samples[j]-samples[j-1])**2;}
    energy.push(Math.sqrt(power/hop));tone.push(Math.sqrt(difference/(power+1e-12)));
  }
  const stride=Math.max(1,Math.ceil(samples.length/6000));
  for(let i=0;i<samples.length;i+=stride){let max=0;for(let j=i;j<Math.min(i+stride,samples.length);j++)max=Math.max(max,Math.abs(samples[j]));peaks.push(max);}
  const maximum=energy.reduce((m,v)=>Math.max(m,v),.00001), threshold=maximum*.025;
  const first=energy.findIndex(x=>x>threshold);let last=energy.length-1;while(last>=0&&energy[last]<=threshold)last--;
  const duration=samples.length/sr,start=Math.max(0,first*.02),end=last<0?duration:Math.min(duration,(last+1)*.02);
  const onset=energy.map((e,i)=>Math.max(0,e-(energy[i-1]??e)));
  let best=0,bpm=0;
  for(let candidate=70;candidate<=180;candidate++){
    const lag=Math.round(3000/candidate);let numerator=0,denominator=0;
    for(let i=lag;i<onset.length;i++){numerator+=onset[i]*onset[i-lag];denominator+=onset[i]**2;}
    const score=numerator/(denominator+1e-12);if(score>best){best=score;bpm=candidate;}
  }
  const beat=best>.12?60/bpm:.5;
  let phase=start,phaseScore=-1;
  for(let offset=0;offset<beat;offset+=.02){let score=0;for(let t=start+offset;t<end;t+=beat)score+=onset[Math.round(t/.02)]||0;if(score>phaseScore){phaseScore=score;phase=start+offset;}}
  const average=(a:number[],lo:number,hi:number)=>{const values=a.slice(Math.max(0,Math.floor(lo/.02)),Math.max(1,Math.ceil(hi/.02)));return values.reduce((s,v)=>s+v,0)/Math.max(1,values.length);};
  const sections:Section[]=[];
  for(let t=phase;t<end-.1;t+=beat*16){
    const after=average(energy,t,Math.min(end,t+beat*16))/maximum;
    const before=average(energy,Math.max(start,t-beat*16),t)/maximum;
    const section={time:t,energy:after,tone:average(tone,t,t+beat*16),rhythm:average(onset,t,t+beat*16)/maximum,change:Math.abs(after-before),repeat:0};
    section.repeat=sections.length<2?0:Math.max(0,...sections.slice(0,-1).map(s=>1-Math.min(1,Math.abs(s.energy-section.energy)*2+Math.abs(s.tone-section.tone)+Math.abs(s.rhythm-section.rhythm)*5)));
    sections.push(section);
  }
  if(!sections.length)sections.push({time:start,energy:0,tone:0,rhythm:0,change:0,repeat:0});
  return {duration,bpm:best>.12?bpm:0,confidence:Math.min(1,best),start,end,peaks,sections};
}

export type LocalPlan = {current_track:string;next_track:string;outgoing_transition_timestamp:number;incoming_start_timestamp:number;transition_duration:number;tempo_ratio:number;strategy:string;score:number};
export function chooseTransition(current:string,source:Analysis,candidates:{id:string;analysis:Analysis}[],position:number,cues:Record<string,number>,point?:number,heardFrom=source.start):LocalPlan {
  if(!Number.isFinite(position)||position<0||position>=source.end)throw Error('Seek within the audible part of this track.');
  if(point!==undefined&&(!Number.isFinite(point)||point<position+.1||point>=source.end))throw Error('Choose a future exit within the audible track.');
  const nearest=(a:Analysis,t:number)=>a.sections.reduce((p,s)=>Math.abs(s.time-t)<Math.abs(p.time-t)?s:p);
  const phrase=source.bpm?960/source.bpm:8;
  const minimumHeard=Math.min((source.end-heardFrom)*.55,Math.max((source.end-heardFrom)*.35,phrase*3));
  let exits=point!==undefined?[{...nearest(source,point),time:point}]:source.sections.filter(s=>s.time>Math.max(position+2,heardFrom+minimumHeard)&&s.time<source.end-.3);
  if(!exits.length)exits=[{...nearest(source,source.end),time:Math.max(position+.1,source.end-.3)}];
  let best:LocalPlan|undefined;
  for(const target of candidates){
    const a=target.analysis,manualCue=cues[target.id];
    if(manualCue!==undefined&&(!Number.isFinite(manualCue)||manualCue<0||manualCue>=a.end))throw Error('Incoming cue is outside the audible track.');
    const remaining=Math.min(a.duration*.5,Math.max(a.duration*.3,a.bpm?2880/a.bpm:24));
    const entries=manualCue!==undefined?[{...nearest(a,manualCue),time:manualCue}]:a.sections.filter(s=>s.time<a.end-remaining&&s.energy>.03);
    for(const exit of exits)for(const entry of entries){
      const tempo=source.bpm&&a.bpm?Math.min(1,Math.abs(source.bpm-a.bpm)/Math.max(source.bpm,a.bpm)*4):.5;
      const mismatch=Math.abs(exit.energy-entry.energy)*2+Math.abs(exit.tone-entry.tone)*.5+Math.abs(exit.rhythm-entry.rhythm)*3+tempo;
      const score=1-mismatch+exit.change*.2+exit.repeat*.35-(exit.time-heardFrom)/source.duration*.35;
      const duration=Math.min(tempo>.25?.25:1.5,source.end-exit.time,a.end-entry.time);
      if(duration<.02)continue;
      if(!best||score>best.score)best={current_track:current,next_track:target.id,outgoing_transition_timestamp:exit.time,incoming_start_timestamp:entry.time,transition_duration:duration,tempo_ratio:1,strategy:point!==undefined||manualCue!==undefined?'manual_local_crossfade':'local_crossfade',score};
    }
  }
  if(!best)throw Error('No playable transition found. Choose another track or cue.');
  return best;
}
