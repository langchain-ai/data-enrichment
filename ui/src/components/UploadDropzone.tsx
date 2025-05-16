import React, { useCallback, useRef } from "react";

export default function UploadDropzone() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = useCallback(async (file: File) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Only PDF files are accepted."); // Simple feedback
      return;
    }
    const form = new FormData();
    form.append("file", file);
    try {
      await fetch("/upload", { method: "POST", body: form });
      window.location.reload();   // simple refresh; can replace with state update
    } catch (error) {
      console.error("Upload error:", error);
      alert("An error occurred during upload.");
    }
  }, []);

  const onDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      const files = e.dataTransfer.files;
      if (!files.length) return;
      handleFileUpload(files[0]);
    },
    [handleFileUpload]
  );

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files || !files.length) return;
      handleFileUpload(files[0]);
      // Reset file input to allow selecting the same file again
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [handleFileUpload]
  );

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
      className="border-2 border-dashed rounded-lg p-8 text-center space-y-4"
    >
      <p>Drag & drop PDFs here</p>
      <p>or</p>
      <input
        type="file"
        ref={fileInputRef}
        onChange={onFileChange}
        accept=".pdf"
        style={{ display: "none" }}
      />
      <button
        onClick={onButtonClick}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Select PDF File
      </button>
    </div>
  );
} 