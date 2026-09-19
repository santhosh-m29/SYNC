import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {stripTypeScriptTypes} from 'node:module';

const sources=[];
class Context {
  currentTime=100;
  destination={};
  createGain(){return {connect(){},disconnect(){},gain:{value:1,setValueAtTime(){},setTargetAtTime(){},curves:[],setValueCurveAtTime(values,at,duration){this.curves.push({values,at,duration});}}};}
  createBufferSource(){const source={connect(){},disconnect(){},start(at,offset){this.started={at,offset};},stop(at){this.stopped=at;}};sources.push(source);return source;}
  async resume(){}
  async close(){}
}
const code=stripTypeScriptTypes(readFileSync(new URL('./src/browser-engine.ts',import.meta.url),'utf8')).replace(/^import .*;\s*/gm,'').replace('export class BrowserEngine','exports.BrowserEngine = class BrowserEngine');
const exports={};
const timeline=stripTypeScriptTypes(readFileSync(new URL('./src/session-timeline.ts',import.meta.url),'utf8')).replace(/^import .*;\s*/gm,'').replaceAll('export function ','function ');
vm.runInNewContext(timeline+'\n'+code,{exports,planTransition(){throw Error('Unexpected planning in manual test');},require(){return {};},AudioContext:Context,setInterval(){return 1;},clearInterval(){},Float32Array,console});
const engine=new exports.BrowserEngine();
engine.auto=false;
const ctx=engine.audio();
engine.current={id:'a',buffer:{duration:180},offset:20,rate:1,at:100};
engine.playing=true;
engine.start(engine.current,100);
const event={next_track:'b',tempo_ratio:1.05,incoming_start_timestamp:42,outgoing_transition_timestamp:30,transition_duration:4};
engine.arm(event,{duration:180});
assert.equal(engine.incoming.node.started.at,110);
assert.equal(engine.incoming.node.started.offset,40);
assert.equal(engine.current.node.stopped,114);
const outgoing=engine.current.gain.gain.curves[0],incoming=engine.incoming.gain.gain.curves[0];
assert.equal(outgoing.at,incoming.at);
assert.equal(outgoing.duration,incoming.duration);
for(let i=0;i<256;i++)assert.ok(Math.abs(outgoing.values[i]**2+incoming.values[i]**2-1)<1e-6);
ctx.currentTime=115;
engine.tick();
assert.equal(engine.current.id,'b');
assert.equal(engine.position,47.25);
engine.pause();
ctx.currentTime=130;
assert.equal(engine.position,47.25);
assert.equal(engine.incoming,null);
engine.seek(12);
assert.equal(engine.position,12);
engine.dispose();
console.log('Browser engine: synchronized overlap, equal-power curves, promotion, pause and seek passed.');
