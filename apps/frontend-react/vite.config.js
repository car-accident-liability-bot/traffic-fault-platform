import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        // 수정 포인트:
        // Java 백엔드 모듈 포트를 8080 -> 28080으로 변경한다.
        target: 'http://localhost:28080',
        changeOrigin: true,
      },
    },
  },
});
