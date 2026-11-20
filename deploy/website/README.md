# Hosting aidotnet.dev on our own VPS

This replaces Vercel. The website is a static Astro build (sign-in and billing run in the browser
against Supabase and Stripe), so it needs only a web server. The traffic path is:

```
visitor ──► Cloudflare (free plan) ──► VPS, port 443 only ──► Caddy ──► /srv/aidotnet/www/current
            DDoS protection, edge cache,    firewall admits only
            TLS, analytics                  Cloudflare's addresses
```

**What it costs:** $0 beyond the VPS. Cloudflare's free plan has no bandwidth or request
charges, so a flood of traffic can slow things down at worst but can never run up a bill.

**Why floods don't reach the server:** Cloudflare blocks DDoS traffic at its edge and serves
cached pages itself. The VPS firewall drops every HTTPS connection that doesn't come from
Cloudflare, so an attacker who finds the server's IP still can't go around Cloudflare.

| File | What it is |
|---|---|
| `setup-vps.sh` | One-time server setup. Safe to run again. |
| `aidotnet.caddy` | Web server config for `aidotnet.dev`, `www.aidotnet.dev` and the private preview `test.aidotnet.dev` |
| `cloudflare-firewall.sh` | Opens port 443 to Cloudflare's addresses only. Runs weekly via cron. |
| `check-preview-secret.sh` | Stops Caddy from starting with an empty or malformed preview secret. Runs before every start. |
| `publish.sh` | Called by GitHub Actions to upload a build and switch to it with no downtime |

The code playground's "Run" button doesn't use this server. It calls the existing Azure
Functions execution service, so the VPS never runs code that visitors submit.

---

## Moving over, step by step

Steps 1 to 6 can all happen while the old site still points at Vercel. Nothing visitors see
changes until step 7.

### Step 1: Add the domain to Cloudflare (no switch yet)

1. Sign in at <https://dash.cloudflare.com> (or create a free account).
2. **Add a domain** → enter `aidotnet.dev` → choose the **Free** plan.
3. Cloudflare imports the existing DNS records. **Check that every email record below was
   copied exactly.** If one is missing, email to and from `@aidotnet.dev` stops working after the
   switch.

   | Type | Name | Content | Proxy |
   |---|---|---|---|
   | MX | `aidotnet.dev` | `mx1.improvmx.com` (priority 10) | DNS only |
   | MX | `aidotnet.dev` | `mx2.improvmx.com` (priority 20) | DNS only |
   | TXT | `aidotnet.dev` | `v=spf1 include:spf.improvmx.com ~all` | DNS only |
   | TXT | `resend._domainkey` | the long `p=MIGf...` key (compare it character by character with Resend → Domains) | DNS only |
   | MX | `send` | `feedback-smtp.us-east-1.amazonses.com` (priority 10) | DNS only |
   | TXT | `send` | `v=spf1 include:amazonses.com ~all` | DNS only |

4. **Delete** the imported `A` and `CNAME` records for `aidotnet.dev`, `www` and `test` that
   point at Vercel (`76.76.21.21`, `64.29.17.x`, or `cname.vercel-dns.com`).
5. **Add** three `A` records pointing at the VPS's public IPv4 address. Set each one to
   **Proxied** (orange cloud):
   `aidotnet.dev`, `www`, `test`.
   Don't create any record that points at the VPS IP with the grey cloud (DNS only), because that
   would expose the server's address.
6. Don't change nameservers yet. Cloudflare will remind you about it; that happens in step 7.

### Step 2: Create the Cloudflare origin certificate

Cloudflare uses this certificate to talk to the server securely. It lasts 15 years, so there's
nothing to renew.

1. Cloudflare → `aidotnet.dev` → **SSL/TLS → Origin Server → Create Certificate**.
2. Keep "Generate private key and CSR with Cloudflare". Hostnames: `aidotnet.dev` and
   `*.aidotnet.dev`. Validity: 15 years. Format: **PEM**. Click **Create**.
3. Leave the page open. On the VPS, save the two boxes into these files:
   ```bash
   sudo mkdir -p /etc/caddy/certs
   sudo nano /etc/caddy/certs/aidotnet-origin.pem   # paste "Origin Certificate"
   sudo nano /etc/caddy/certs/aidotnet-origin.key   # paste "Private Key"
   ```
   Cloudflare shows the private key only once.

### Step 3: Make the GitHub deploy key (on your own computer)

```bash
ssh-keygen -t ed25519 -N "" -C "github-actions-aidotnet-website" -f aidotnet-deploy
```

This creates `aidotnet-deploy` (private, goes into a GitHub secret in step 5) and
`aidotnet-deploy.pub` (public, goes onto the VPS in step 4).

### Step 4: Set up the VPS

The server needs Ubuntu 22.04+ or Debian 12+, and nothing else using port 443. From your
computer, in a checkout of this repository:

```bash
ssh YOUR_USER@VPS_IP mkdir -p aidotnet-website
scp -r deploy/website aidotnet-deploy.pub YOUR_USER@VPS_IP:aidotnet-website/
ssh YOUR_USER@VPS_IP
sudo bash ~/aidotnet-website/website/setup-vps.sh --deploy-key ~/aidotnet-website/aidotnet-deploy.pub
```

If SSH runs on a port other than 22, add `--ssh-port PORT`. The script keeps that port open
before it turns the firewall on, so you can't lock yourself out.

At the end, the script prints a `PREVIEW_BYPASS_SECRET` value. You'll need it in step 5.

If it warns that port 80 or 443 is "still open to everyone", an older firewall rule is letting
traffic bypass Cloudflare. Remove that rule unless another site on this server needs it.

### Step 5: Add the GitHub secrets

GitHub → repository → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `WEBSITE_SSH_HOST` | The VPS's public IP address |
| `WEBSITE_SSH_USER` | `aidotnet-deploy` |
| `WEBSITE_SSH_KEY` | The entire contents of the `aidotnet-deploy` file from step 3 |
| `WEBSITE_SSH_KNOWN_HOSTS` | Output of `ssh-keyscan VPS_IP` run on your computer (add `-p PORT` if SSH isn't on 22) |
| `WEBSITE_SSH_PORT` | Only if SSH isn't on port 22 |
| `CLOUDFLARE_ZONE_ID` | Cloudflare → `aidotnet.dev` → **Overview**, right-hand column, "Zone ID" |
| `CLOUDFLARE_API_TOKEN` | Cloudflare → **My Profile → API Tokens → Create Token → Custom token**. Permission **Zone / Cache Purge / Purge**, zone resources **Include / Specific zone / aidotnet.dev** |
| `PREVIEW_BYPASS_SECRET` | The value `setup-vps.sh` printed |

After that, delete the `aidotnet-deploy` private key file from your computer.

### Step 6: First deploy, and check it before anyone sees it

1. GitHub → **Actions → Deploy Marketing Website → Run workflow**. Wait for the
   "Publish to the VPS" step to go green. (The Cloudflare purge and smoke tests can fail until
   step 7; that's expected.)
2. On the VPS, check that the server returns the real site:
   ```bash
   curl -sk --resolve www.aidotnet.dev:443:127.0.0.1 https://www.aidotnet.dev/ | head -5
   curl -sk -o /dev/null -w '%{http_code}\n' --resolve test.aidotnet.dev:443:127.0.0.1 https://test.aidotnet.dev/   # 401 = preview is private
   ```

### Step 7: Switch DNS to Cloudflare

1. In Cloudflare, under `aidotnet.dev` → **DNS → Records**, note the two nameservers it asks for
   (for example `xxx.ns.cloudflare.com`).
2. Namecheap (where the domain is registered) → **Domain List → aidotnet.dev → Manage →
   Nameservers → Custom DNS** → enter the two Cloudflare nameservers → save.
3. Cloudflare emails you when the domain is **Active**. That usually takes minutes, rarely a
   few hours. Visitors move over gradually during that time, and both old and new hosting
   serve the same site.

### Step 8: Turn on Cloudflare's protection settings

All of these are free:

1. **SSL/TLS → Overview:** mode **Full (strict)**.
2. **SSL/TLS → Edge Certificates:** **Always Use HTTPS** on; **Minimum TLS Version** 1.2.
3. **Security → Bots:** leave **Bot Fight Mode off**. It serves JavaScript challenges that
   automated clients cannot pass, so it fails every post-deploy smoke test with HTTP 403, and
   the cache and rate limit below already keep floods off the server.
4. **Caching → Cache Rules → Create rule:** name it "Cache the public site". Set the
   **Hostname equals `www.aidotnet.dev`** → **Eligible for cache**. Set **Edge TTL → Ignore
   cache-control header and use this TTL → 1 day**. Under **Cache key**, turn on **Cache
   deception armor** and set **Query string → Exclude all**: the site is static, so a query
   string never changes a page, and without this an attacker defeats the cache by adding
   `?random` to every request. Cloudflare then answers page requests itself, which is what keeps
   a flood off the server. Every deploy purges this cache automatically. **Never add
   `test.aidotnet.dev` to this rule**, because the preview is private.
5. **Caching → Configuration:** **Browser Cache TTL → Respect Existing Headers**. The default
   (4 hours) overrides the server's `max-age=0` on pages, so visitors would keep seeing the old
   site for hours after a deploy.
6. **Security → WAF → Rate limiting rules → Create rule:** if requests from the **same IP** exceed
   **1000 per 10 seconds** on hostname `www.aidotnet.dev`, **Block** for 10 seconds. Set
   **Also apply rate limiting to cached assets → off** (count only requests that reach the
   server). Lower limits fail real traffic: the smoke tests fetch about 12,000 pages right after
   a deploy has emptied the cache, and one page view loads dozens of files.
7. **Analytics & Logs → Web Analytics:** enable it for `aidotnet.dev`. This replaces Vercel
   Analytics with cookieless analytics injected by Cloudflare, with no code change needed.
8. If you're under attack right now: **Overview → Under Attack Mode** on. Turn it off
   afterwards, because it shows every visitor a check page.

Then rerun **Deploy Marketing Website** once, so the cache purge and smoke tests pass.

### Step 9: Cancel Vercel (after about a day of the new site working)

1. Vercel → each project (`aidotnet_website`, the playground API at `aidotnet.vercel.app`) →
   **Settings → Delete Project**.
2. Vercel → team **Settings → Billing**: downgrade to Hobby or delete the team. This stops the
   $52/month.
3. GitHub → repository secrets: delete `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`
   and `VERCEL_AUTOMATION_BYPASS_SECRET`.

---

## Everyday operations

- **Deploy:** automatic on every push to `master` that touches `website/` (or run
  **Deploy Marketing Website** by hand).
- **Roll back:** on the VPS, `ls /srv/aidotnet/www/releases` lists the last three releases.
  To switch to one:
  `sudo -u aidotnet-deploy ln -sfn releases/RELEASE_NAME /srv/aidotnet/www/current`.
  Then purge the Cloudflare cache (**Caching → Configuration → Purge Everything**).
- **Logs:** `/var/log/caddy/aidotnet-www.log` and `aidotnet-test.log`. Visitor IPs show as
  Cloudflare addresses; the real IP is in the `Cf-Connecting-Ip` request header.
- **Change the preview secret:** edit `/etc/caddy/aidotnet.env` so it holds exactly one line,
  `AIDOTNET_PREVIEW_SECRET=` followed by 64 lowercase hex characters (e.g. from `openssl rand -hex 32`).
  Check it first: `sudo /usr/local/sbin/aidotnet-check-preview-secret && echo ok`.
  Then run `sudo systemctl restart caddy` (a reload does not re-read the file) and update the
  `PREVIEW_BYPASS_SECRET` GitHub secret. Caddy runs the same check before every start and refuses
  to start if the value is empty or malformed. Both sites are then down until the file is fixed;
  `systemctl status caddy` shows the reason.
- **Server updates:** `unattended-upgrades` installs security updates automatically; Caddy updates
  with `apt`.
