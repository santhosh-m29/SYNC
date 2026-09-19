import {analyze} from './local-analysis';
self.onmessage=(event:MessageEvent<{samples:Float32Array;sr:number}>)=>{
  try {self.postMessage({analysis:analyze(event.data.samples,event.data.sr)});}
  catch(error){self.postMessage({error:error instanceof Error?error.message:'Audio analysis failed'});}
};
