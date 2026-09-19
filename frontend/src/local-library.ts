import {chooseTransition,type Analysis,type LocalPlan} from './local-analysis';
export type Track={id:string;title:string;filename:string;duration:number;audioEnd?:number;bpm:number;key:string|null;status:string;error?:string};
export type Transition=LocalPlan;
type Entry={track:Track;file:File;analysis?:Analysis};
// Session-only File references and derived metadata. No uploads, storage APIs or persistence.
const entries=new Map<string,Entry>();
export const listTracks=()=>[...entries.values()].map(e=>({...e.track}));
export const waveform=async(id:string)=>({peaks:entries.get(id)?.analysis?.peaks||[]});
export async function decodeFile(id:string,context:AudioContext){
  const entry=entries.get(id);if(!entry)throw Error('Select this song from your device again.');
  return context.decodeAudioData(await entry.file.arrayBuffer());
}
export async function importSong(file:File){
  const id=crypto.randomUUID(),track:Track={id,title:file.name.replace(/\.[^.]+$/,''),filename:file.name,duration:0,bpm:0,key:null,status:'analyzing'};
  const entry:Entry={track,file};entries.set(id,entry);
  const context=new AudioContext();
  try{
    const audio=await context.decodeAudioData(await file.arrayBuffer());
    // Analysis uses one channel; copying avoids detaching the browser's decoded audio.
    const samples=new Float32Array(audio.getChannelData(0));
    const analysis=await new Promise<Analysis>((resolve,reject)=>{
      const worker=new Worker(new URL('./analysis.worker.ts',import.meta.url),{type:'module'});
      const timeout=setTimeout(()=>{worker.terminate();reject(Error('Analysis timed out. Try a shorter file.'));},120000);
      worker.onmessage=e=>{clearTimeout(timeout);worker.terminate();e.data.error?reject(Error(e.data.error)):resolve(e.data.analysis);};
      worker.onerror=()=>{clearTimeout(timeout);worker.terminate();reject(Error('Could not analyze this file in your browser.'));};
      worker.postMessage({samples,sr:audio.sampleRate},[samples.buffer]);
    });
    entry.analysis=analysis;Object.assign(track,{duration:analysis.duration,audioEnd:analysis.end,bpm:analysis.bpm,status:'ready'});
  }catch(error){track.status='error';track.error=error instanceof Error?error.message:'This browser cannot decode this audio file.';throw error;}
  finally{await context.close();}
  return track;
}
export function planTransition(current:string,candidates:string[],position:number,cues:Record<string,number>,point?:number,heardFrom?:number){
  const source=entries.get(current)?.analysis;if(!source)throw Error('Current song is not analyzed yet.');
  return chooseTransition(current,source,candidates.flatMap(id=>{const analysis=entries.get(id)?.analysis;return analysis?[{id,analysis}]:[];}),position,cues,point,heardFrom);
}

export function trackCompatibility(a:Analysis,b:Analysis):number {
  const bpmRatio=a.bpm&&b.bpm?Math.abs(a.bpm-b.bpm)/Math.max(a.bpm,b.bpm):.5;
  const bpmScore=1-Math.min(1,bpmRatio*3);
  const aEnergy=a.sections.length?a.sections.reduce((s,x)=>s+x.energy,0)/a.sections.length:.5;
  const bEnergy=b.sections.length?b.sections.reduce((s,x)=>s+x.energy,0)/b.sections.length:.5;
  const energyScore=1-Math.min(1,Math.abs(aEnergy-bEnergy)*2);
  const aTone=a.sections.length?a.sections.reduce((s,x)=>s+x.tone,0)/a.sections.length:.5;
  const bTone=b.sections.length?b.sections.reduce((s,x)=>s+x.tone,0)/b.sections.length:.5;
  const toneScore=1-Math.min(1,Math.abs(aTone-bTone));
  return bpmScore*.5+energyScore*.3+toneScore*.2;
}

export function sequenceTrackIds(ids:string[],startId?:string):string[] {
  if(ids.length<=1)return[...ids];
  const remaining=new Set(ids),result:string[]=[];
  let currentId=(startId&&remaining.has(startId))?startId:ids[0];
  result.push(currentId);remaining.delete(currentId);
  while(remaining.size>0){
    const currentAnalysis=entries.get(currentId)?.analysis;
    let bestNext='',bestScore=-Infinity;
    for(const candId of remaining){
      const candAnalysis=entries.get(candId)?.analysis;
      if(!currentAnalysis||!candAnalysis){if(!bestNext)bestNext=candId;continue;}
      const score=trackCompatibility(currentAnalysis,candAnalysis);
      if(score>bestScore){bestScore=score;bestNext=candId;}
    }
    if(!bestNext)bestNext=remaining.values().next().value!;
    result.push(bestNext);remaining.delete(bestNext);currentId=bestNext;
  }
  return result;
}
