# Security Policy

## Reporting a Vulnerability

We take the security of AiDotNet seriously. If you believe you have found a
security vulnerability, please report it via one of the channels below.

### Preferred: GitHub Security Advisories (encrypted, audit-trailed)

Report privately through the [GitHub Security Advisory form](https://github.com/ooples/AiDotNet/security/advisories/new).
This is the fastest path and gives us a structured audit trail.

### Alternative: Email

If you cannot use GitHub Security Advisories, email **admin@aidotnet.dev**.
For sensitive disclosures we strongly recommend encrypting your report with
our PGP key (see `docs/security/pgp.txt` in this repository, or fetch from
https://aidotnet.dev/security).

Please do **not** report security vulnerabilities through public GitHub
issues, discussions, or any other public forum.

## Response SLA

We aim to meet the following response times:

| Stage                            | Target                    |
|----------------------------------|---------------------------|
| Acknowledge receipt              | 5 business days           |
| Triage + severity classification | 10 business days          |
| Fix for SEV-1 (critical)         | 30 days from triage       |
| Fix for SEV-2 (high)             | 60 days from triage       |
| Fix for SEV-3 (medium) or lower  | Next minor release        |

Severity uses the [CVSS 3.1](https://www.first.org/cvss/v3.1/specification-document)
calculator. We assign CVE identifiers via GitHub Security Advisories for any
vulnerability rated SEV-2 or higher.

## Embargo Policy

We follow a 90-day embargo from the date we acknowledge the report, or until
a fix is publicly released — whichever is sooner. Reporters who prefer a
shorter or longer embargo for coordinated disclosure should say so in their
initial report. We will not unilaterally extend embargoes beyond 90 days
without reporter agreement.

## Supported Versions

We provide security fixes for the following versions:

| Version Range  | Supported              | Notes                                |
|----------------|------------------------|--------------------------------------|
| Latest minor   | :white_check_mark:     | All security fixes                   |
| Previous minor | :white_check_mark:     | SEV-1 and SEV-2 fixes only           |
| All older 0.x  | :x:                    | Upgrade required                     |
| 1.0.0+ LTS     | (Future) 3-year window | Once 1.0 ships                       |

Until AiDotNet reaches 1.0, supported versions are the two most recent
minor releases. After 1.0, we will publish a long-term-support window and
update this table accordingly.

## Scope

Security policy applies to:

- `AiDotNet` (main library)
- `AiDotNet.Tensors` (numerical core)
- `AiDotNet.Serving` (inference server)
- `AiDotNet.Native.*` (native acceleration packages)
- Pre-built NuGet packages distributed via nuget.org under the `AiDotNet` org

Out of scope (report directly to the upstream maintainer):

- Vulnerabilities in third-party dependencies — please report to the upstream
  project first; we will coordinate downstream remediation once upstream has
  a fix or assigns a CVE.
- Vulnerabilities in Pre-trained Community Models that we distribute as
  metadata only — report to the model author per their model card.

## Coordinated Disclosure & Credit

We credit reporters in our advisory text and in release notes unless the
reporter requests anonymity. If you wish to publish your own write-up after
the embargo ends, please share the draft with us at least 48 hours in
advance so we can coordinate timing.

## Federal / Regulated Adoption

For U.S. government, military, national laboratory, and other regulated
deployments (FedRAMP, FISMA, NIST AI RMF / SP 800-218 SSDF, CMMC, HIPAA,
SOC 2 Type II, ISO 27001, etc.), please contact admin@aidotnet.dev for our
Enterprise security program. The Enterprise tier includes:

- Air-gapped deployment support (no telemetry, no license-server callout)
- FIPS 140-3 compatible cryptographic modules
- SBOM (CycloneDX 1.5) generation per release
- SLSA Level 3 build provenance attestations
- Signed NuGet packages
- Dedicated security contact + custom SLA
- NIST SP 800-218 SSDF compliance documentation

See https://aidotnet.dev/federal-use for the federal-use page or
https://aidotnet.dev/enterprise for general enterprise terms.

## Maintainer Escalation

This project is currently maintained by a single primary maintainer. If you
have not received an acknowledgement within 5 business days via the primary
channels above, please escalate to admin@aidotnet.dev with subject
`[ESCALATION] Security report not acknowledged`. We are actively working
to add a backup security contact and will update this section once that is
in place.

## Public Statements

We will not comment publicly on a vulnerability before the embargo expires
or a fix is released. Once the embargo lifts, our public statement will
appear on the GitHub Security Advisories page for the report.
