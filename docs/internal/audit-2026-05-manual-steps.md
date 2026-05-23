# audit-2026-05 — Manual steps required (you do these)

The audit-2026-05 PR closes most findings in code, but several items require
manual action outside the PR. Track each here and check off as completed.

## DNS configuration for aidotnet.dev (BLOCKS website value)

The Astro site at `website/` is built and deployed to Vercel
(`franklins-projects-02a0b5a0/website`) but DNS for aidotnet.dev points
nowhere, so every URL returns 404. Until DNS is fixed, every reference
this PR adds to `https://aidotnet.dev/*` (security, enterprise, federal-use,
pricing, license, privacy pages) is broken for end users.

Per `website/LAUNCH-PLAN.md`:

- [ ] **A record:** `aidotnet.dev` → `76.76.21.21`
- [ ] **CNAME:** `www.aidotnet.dev` → `cname.vercel-dns.com`
- [ ] **Add domain in Vercel dashboard:** Settings → Domains → Add → `aidotnet.dev` + `www.aidotnet.dev`
- [ ] Verify HTTPS certificate provisions cleanly (auto-issued by Vercel)
- [ ] Smoke-test the deployed pages:
      `curl -sL https://aidotnet.dev/security | head -5`
      `curl -sL https://aidotnet.dev/enterprise | head -5`
      `curl -sL https://aidotnet.dev/federal-use | head -5`
      `curl -sL https://aidotnet.dev/pricing | head -5`
      `curl -sL https://aidotnet.dev/license | head -5`
      `curl -sL https://aidotnet.dev/privacy | head -5`

## admin@aidotnet.dev mailbox + PGP key (SECURITY.md depends on this)

- [ ] Set up `admin@aidotnet.dev` mailbox (or alias) — ImprovMX is already
      configured per LAUNCH-PLAN.md; verify it works:
      `echo "test" | mail -s "test" admin@aidotnet.dev`
- [ ] Generate PGP key:
      `gpg --full-generate-key`
      (Type: RSA 4096 or ed25519; Expiry: 2y; Real name: AiDotNet Security; Email: admin@aidotnet.dev)
- [ ] Export public key:
      `gpg --armor --export admin@aidotnet.dev > docs/security/pgp.txt`
      Commit + push.
- [ ] Publish to public keyserver:
      `gpg --send-keys --keyserver hkps://keys.openpgp.org <KEY-ID>`
- [ ] Mirror at `https://aidotnet.dev/security` (already linked from the page)
- [ ] Store private key in Ooples password manager / hardware token. **Never commit.**

## Enable GitHub Security Advisories (SECURITY.md primary intake channel)

- [ ] **ooples/AiDotNet:** Settings → Security → Security advisories → enable + draft a placeholder advisory to confirm the form is reachable
- [ ] **ooples/AiDotNet.Tensors:** same
- [ ] Confirm the URL `https://github.com/ooples/AiDotNet/security/advisories/new` is reachable for external reporters

## Enterprise license signing key (Phase 4)

Phase 0's `build/AiDotNet.Tensors.Enterprise.targets` validates DISABLE_TELEMETRY /
DISABLE_LICENSE_GUARD flags against a placeholder marker
(`AiDotNet-Enterprise-License-v1` string). Phase 4 will replace with ed25519
signature verification. For that:

- [ ] Generate ed25519 enterprise-license signing key pair (Ooples private key)
- [ ] Embed public key in the targets file (or sibling .props file)
- [ ] Implement signature verifier as inline `RoslynCodeTaskFactory` task
      OR precompiled validator DLL shipped under `build/`
- [ ] License-file format: JSON document with fields:
      `{tenant, issued, expires, scope: [DISABLE_TELEMETRY, ...], signature}`
- [ ] Document license-request → signed-file workflow at `/enterprise`

## Stripe Payment Links (LAUNCH-PLAN.md item)

The pricing page references Stripe products that need to be created:

- [ ] Create Professional tier product ($29/mo per seat) — Stripe dashboard
- [ ] Create Enterprise tier product ($99/mo per seat)
- [ ] Update Vercel environment variables: STRIPE_PROFESSIONAL_LINK, STRIPE_ENTERPRISE_LINK
- [ ] Smoke test checkout flow end-to-end

## Backup maintainer onboarding (Phase 6)

Audit finding #16 (bus factor = 1). Per design discussion, document escalation
path now; identify second maintainer when feasible:

- [ ] Identify candidate(s) — established contributor with PR history, OR
      trusted external collaborator, OR contract hire
- [ ] Grant NuGet publish rights (NuGet API key share or organization
      membership)
- [ ] Grant GitHub admin on ooples/AiDotNet + ooples/AiDotNet.Tensors
- [ ] Update CODEOWNERS — assign second maintainer to subsystem(s) of their
      expertise (likely Tensors numerical core OR high-level model families)
- [ ] Update SECURITY.md "Maintainer Escalation" section once a real
      backup contact exists

## Audit issue closure

After all manual steps complete + the PR merges, update each audit issue:

- [ ] `gh issue comment 1425 --body "Phase 0 PR merged. Findings #1-#5 closed."`
- [ ] `gh issue comment 1426 --body "Phase 0 PR merged. Findings #6-#11 closed."`
- [ ] `gh issue comment 1427 --body "Phase 0 PR merged. Findings #12-#14 closed."`
- [ ] `gh issue comment 1428 --body "Phase 0 PR merged. Findings #15-#19 closed."`
- [ ] `gh issue close 1425 1426 1427 1428` after all checkboxes complete
