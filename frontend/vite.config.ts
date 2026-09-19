import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['sync.png'],
      manifest: {
        name: 'SYNC — Live DJ',
        short_name: 'SYNC',
        description: 'Live AI DJ that syncs music to your session in real time.',
        theme_color: '#101112',
        background_color: '#101112',
        display: 'standalone',
        orientation: 'portrait',
        start_url: '/',
        icons: [
          {
            src: '/sync.png',
            sizes: 'any',
            type: 'image/png',
            purpose: 'any maskable',
          },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,png,svg,ico,woff,woff2}'],
        runtimeCaching: [],
      },
      devOptions: {
        enabled: true,
      },
    }),
    {
      name: 'local-only-privacy',
      apply: 'build' as const,
      transformIndexHtml() {
        return [
          {
            tag: 'meta',
            attrs: {
              'http-equiv': 'Content-Security-Policy',
              content: "connect-src 'none'; worker-src 'self'; form-action 'none'; object-src 'none'; base-uri 'self'",
            },
            injectTo: 'head-prepend' as const,
          },
        ];
      },
    },
  ],
});
