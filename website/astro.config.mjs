import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

const isVercel = process.env.VERCEL === '1';

export default defineConfig({
  site: isVercel ? 'https://aidotnet.vercel.app' : 'https://ooples.github.io',
  base: isVercel ? '/' : '/AiDotNet/',
  integrations: [mdx()],
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
