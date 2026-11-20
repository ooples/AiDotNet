import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

// AIDOTNET_SITE_ROOT=1 builds for https://www.aidotnet.dev (served from the domain root);
// otherwise the build targets the GitHub Pages project path /AiDotNet/.
const isSiteRoot = process.env.AIDOTNET_SITE_ROOT === '1';

export default defineConfig({
  site: isSiteRoot ? 'https://www.aidotnet.dev' : 'https://ooples.github.io',
  base: isSiteRoot ? '/' : '/AiDotNet/',
  integrations: [mdx()],
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
  },
});
