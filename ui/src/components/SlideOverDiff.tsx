import React from "react";
import * as Dialog from "@radix-ui/react-dialog";
type Props={open:boolean,run:any,onClose:()=>void};
export default function SlideOverDiff({open,run,onClose}:Props){
  return(
    <Dialog.Root open={open} onOpenChange={v=>!v&&onClose()}>
      <Dialog.Portal>
        <Dialog.Content className="fixed inset-y-0 right-0 w-full max-w-xl bg-white shadow-xl p-6 overflow-y-auto">
          <h2 className="font-semibold text-lg mb-4">Diff Review</h2>
          <pre className="bg-gray-50 p-4 rounded text-xs overflow-x-auto">
            '''
{JSON.stringify(run?.diff_json,null,2)} 
            '''
          </pre>
          <button className="mt-4 px-4 py-2 bg-green-600 text-white rounded">Approve</button>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
} 