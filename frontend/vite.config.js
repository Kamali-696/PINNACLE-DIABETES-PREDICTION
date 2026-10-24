import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development, /api requests are proxied to the FastAPI server.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
