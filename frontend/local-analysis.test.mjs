import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {stripTypeScriptTypes} from 'node:module';
const source=readFileSync(new URL('./src/local-analysis.ts',import.meta.url),'utf8');
const code=stripTypeScriptTypes(source).replaceAll('export function ','function ');
const scope={Float32Array,Math};vm.createContext(scope);vm.runInContext(code,scope);
const sr=2000,samples=new Float32Array(sr*40);
for(let beat=2;beat<38;beat+=.5)for(let i=0;i<100;i++)samples[Math.floor(beat*sr)+i]=Math.sin(i*.4)*Math.exp(-i/20)*.5;
const result=scope.analyze(samples,sr);
assert.ok(result.bpm>=118&&result.bpm<=122,`Expected 120 BPM, got ${result.bpm}`);
assert.ok(result.start>=1.9&&result.end<39);
assert.ok(result.peaks.some(p=>p>.1));
const section=(time,energy,repeat=0)=>({time,energy,tone:.3,rhythm:.1,change:0,repeat});
const a={duration:180,bpm:120,confidence:1,start:0,end:178,peaks:[],sections:[section(0,.5),section(32,.8),section(64,.7,1),section(96,.7,1),section(160,.7,1)]};
const b={...a,sections:[section(0,.01),section(32,.2),section(64,.7),section(96,.4)]};
const bad={...b,bpm:75,sections:[section(0,.05),section(32,.05)]};
const plan=scope.chooseTransition('a',a,[{id:'bad',analysis:bad},{id:'b',analysis:b}],0,{});
assert.equal(plan.next_track,'b');assert.equal(plan.incoming_start_timestamp,64);assert.equal(plan.outgoing_transition_timestamp,64);
assert.ok(plan.outgoing_transition_timestamp+plan.transition_duration<=a.end);
const manual=scope.chooseTransition('a',a,[{id:'b',analysis:b}],10,{b:32},40);
assert.equal(manual.outgoing_transition_timestamp,40);assert.equal(manual.incoming_start_timestamp,32);
assert.throws(()=>scope.chooseTransition('a',a,[{id:'b',analysis:b}],50,{},40));
for(const file of ['local-library.ts','local-analysis.ts','analysis.worker.ts','browser-engine.ts','main.tsx']){
 const text=readFileSync(new URL('./src/'+file,import.meta.url),'utf8');
 assert.ok(!/\b(fetch|XMLHttpRequest|WebSocket|sendBeacon|localStorage|sessionStorage|indexedDB)\b/.test(text),`${file} must not upload or persist files`);
}
console.log('Local analysis: tempo, silence bounds, compatible song/entry/exit, manual cues and no-network/storage checks passed.');
