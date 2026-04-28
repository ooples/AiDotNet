import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Content-layer API (Astro 5+/6+). The legacy `type: 'content'` shape
// silently produces an empty collection in Astro 6 clean installs (the
// CI build emitted only 48 of the expected 78 pages and surfaced
// `The collection "docs" does not exist or is empty` during
// `generating static routes`). The new `loader: glob({...})` form is
// explicit about which files belong to the collection and works
// identically in cached and clean environments.
const docsCollection = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/docs' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    order: z.number().optional(),
    section: z.string().optional(),
  }),
});

export const collections = {
  docs: docsCollection,
};
