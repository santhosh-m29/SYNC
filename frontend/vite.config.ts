import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({ plugins: [react(), {
  name: 'local-only-privacy', apply: 'build',
  transformIndexHtml() {
    return [{tag:'meta',attrs:{'http-equiv':'Content-Security-Policy',content:"connect-src 'none'; worker-src 'self'; form-action 'none'; object-src 'none'; base-uri 'self'"},injectTo:'head-prepend'}];
  },
}] });
