import React,{useEffect,useState} from "react";
import RowCard from "./components/RowCard";
import UploadDropzone from "./components/UploadDropzone";
import SlideOverDiff from "./components/SlideOverDiff";

export default function App(){
  const [runs,setRuns]=useState<any[]>([]);
  const [current,setCurrent]=useState<any|null>(null);

  useEffect(()=>{
    fetch("/runs").then(r=>r.json()).then(setRuns);
  },[]);

  return(
    <div className="p-6 space-y-6">
      <UploadDropzone/>
      <div className="border rounded">
        {runs.map(run=><RowCard key={run.run_id} run={run} onOpen={setCurrent}/>)}
      </div>
      <SlideOverDiff open={!!current} run={current} onClose={()=>setCurrent(null)}/>
    </div>);
}
