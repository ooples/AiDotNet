import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Astro Content Layer API (Astro 5+/6+): use an explicit glob loader
// so docs entry discovery is deterministic across clean and cached
// builds, independent of any legacy collection-detection heuristics.
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
