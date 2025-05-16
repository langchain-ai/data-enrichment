import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5174,                 // keep the port that was auto-picked
    proxy: {
      "/upload": "http://localhost:8000",
      "/runs":   "http://localhost:8000",
    },
  },
}); 