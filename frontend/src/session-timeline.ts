import type {Track,Transition} from './local-library';
export type TimelineClip={id:string;start:number;cue:number;end:number;exit?:number;fade:number;manual:boolean;past?:boolean};
export type Planner=(id:string,candidates:string[],position:number,cues:Record<string,number>,point?:number,heardFrom?:number)=>Transition;
/** All coordinates use session seconds; cues/exits use source-file seconds. */
export function buildTimeline(tracks:Track[],current:string,start:number,cue:number,event:Transition|null,remaining:string[],planner:Planner,cues:Record<string,number>,points:Record<string,number>,auto:boolean,manualOrder:boolean):TimelineClip[]{
  const clips:TimelineClip[]=[];let id=current,at=start,entry=cue,next=event;
  const pending=remaining.filter(x=>x!==id);
  for(let i=0;i<=tracks.length;i++){
    const track=tracks.find(t=>t.id===id);if(!track)break;
    clips.push({id,start:at,cue:entry,end:at+((next?next.outgoing_transition_timestamp+next.transition_duration:track.duration)-entry),exit:next?.outgoing_transition_timestamp,fade:next?.transition_duration||0,manual:next?.strategy.startsWith('manual_')||false});
    if(!next)break;
    at+=next.outgoing_transition_timestamp-entry;entry=next.incoming_start_timestamp;id=next.next_track;
    const index=pending.indexOf(id);if(index<0)break;pending.splice(index,1);
    const position=entry+next.transition_duration;
    try{next=pending.length&&(auto||points[id]!==undefined)?planner(id,pending.slice(0,1),position,cues,points[id],entry):null;}catch{next=null;}
  }
  return clips;
}

export function boundedStart(previous:TimelineClip,sourceEnd:number,requested:number,minimumSource:number){
  const low=Math.max(previous.cue+.2,minimumSource+.2),high=sourceEnd-.3;
  if(high<low)throw Error('This transition is too close to playback to move.');
  return Math.max(low,Math.min(high,previous.cue+requested-previous.start));
}
