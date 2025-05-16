import React from "react";
type Props={run:any,onOpen:(run:any)=>void};
export default function RowCard({run,onOpen}:Props){
  return(
    <div className="flex justify-between p-3 border-b cursor-pointer hover:bg-gray-50"
         onClick={()=>onOpen(run)}>
      <span className="font-medium">{run.file_name}</span>
      <span className="text-sm text-gray-500">{run.status}</span>
    </div>);
} 