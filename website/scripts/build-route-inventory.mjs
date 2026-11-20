import { readdirSync, writeFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

const routes = [];
function walk(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) walk(path);
    else if (entry.name === 'index.html') {
      const slug = relative('dist', directory).split(sep).join('/');
      routes.push(slug ? `/${slug}/` : '/');
    }
  }
}
walk('dist');
writeFileSync('dist/verification-routes.json', JSON.stringify(routes.sort()));
console.log(`Recorded ${routes.length} built routes for deployed Playwright verification.`);
