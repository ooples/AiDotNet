# Test-impact review controls

Keep maintained contracts beside the tooling. Record dated run results, commit
identities and before/after evidence in the pull request that ran them, not as an
ever-current claim in this inventory.

## Boundaries covered

| Entry point | Contract |
| --- | --- |
| `Test-CiImpactWorkflow.ps1` | Expensive-job gates, map/PR identity, bounded waits, named download/import steps, escalation-safe imports, and heavy survey/window configuration |
| `Test-CiImpactWorkflowReview.ps1` | Unsafe workflow and manifest mutations must fail for the intended reason; unrelated/comment-only evidence cannot satisfy an execution step |
| `Test-CertificateEvidenceReview.ps1` | Execute actual tree-binding and candidate-selection code; inconsistent certificates are ineligible for both exact and delta reuse |
| `Resolve-CiValidationReuse.ps1 -SelfTest` | Canonical named certificate scopes, required artifacts, exact-tree evidence, and typed delta decisions |
| `Test-CiValidationReuseReview.ps1` | Execute the production emission branch for complete, missing and malformed artifact inventories; observe the decline path, not only its fallback result |
| `Test-CiValidationReuseReviewControls.ps1` | Removing production decline notices must make every decline fixture fail even though fallback output remains correct |
| `Test-NoCoverageShardPolicy.ps1` | Retain legitimate growing lists, collapse duplicate names, and reject oversized invalid parses |
| `Test-ReviewFixtureCleanup.ps1` | Reject paths outside the temporary root or expected fixture prefix before recursive cleanup |
| `Test-TestImpactEndToEnd.ps1` | Real Git histories for selected PR tests, behind-master PRs, deleted tests, exact/delta reuse, and fail-closed invalid maps; runs the review controls above |
| `.github/scripts/New-CoverageRunSettings.ps1 -SelfTest` | Preserve coverage XML and write output in the PowerShell provider location even when the process directory differs |

Closed policy/test modes use enums. Artifact names, paths, GitHub expressions and
JSON values are boundary data, not internal dispatch modes. Tests must check both
valid reuse and fail-closed behavior; accepting no reuse at all is not a fix.

## Run the focused controls

From the repository root in PowerShell, stop on failure:

```powershell
$ErrorActionPreference = 'Stop'
foreach ($script in @('Select-Shards', 'Resolve-CiValidationReuse')) {
    pwsh -NoProfile -File "tools/TestImpact/$script.ps1" -SelfTest
    if ($LASTEXITCODE -ne 0) { throw "$script self-test failed" }
}
foreach ($script in @('Test-ValidationReuseModes', 'Test-CiGateModes', 'Test-TestImpactEndToEnd')) {
    pwsh -NoProfile -File "tools/TestImpact/$script.ps1"
    if ($LASTEXITCODE -ne 0) { throw "$script failed" }
}
pwsh -NoProfile -File .github/scripts/New-CoverageRunSettings.ps1 -SelfTest
if ($LASTEXITCODE -ne 0) { throw 'Coverage settings self-test failed' }
```

These are finite local decision/wiring checks. A full matrix is correct for
unmapped, infrastructure or broadly shared changes. Hosted checks remain a
separate result; do not claim a live run used fewer shards based only on these
fixtures. The model conformance test independently rejects an empty window, so
inventory changes must be checked against its actual reflected candidate set.
