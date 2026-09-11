# PR #2159 review-fix evidence (2026-09-11)

The integration commit `007e621f87405359018a809b988b252e41d0e792` is a
direct descendant of collaborator head `7e4c25ff41fbd90b632556af384a148209f3304d`.
Their selector, artifact-validation, compatibility, and deletion-test fixes were
preserved. This is local executable proof, not a claim that a new hosted run has
already completed.

## Failure first

The stronger workflow contract rejected the collaborator head's timeout budget:
10 minutes for map resolution + 40 minutes waiting for evidence + 5 minutes of
setup/fallback margin exceeds the 50-minute job. Map resolution is now bounded
at 5 minutes. Seven negative controls reject missing or incorrect map-head
arguments, comment-only arguments, renamed or comment-only step identity, and
missing or excessive map timeouts.

The other original review defects already had overlapping fixes in that parent;
they are not represented here as failures still present on the parent.

## Independently repeated results

From the repository root:

```powershell
pwsh -NoProfile -File tools/TestImpact/Select-Shards.ps1 -SelfTest
pwsh -NoProfile -File tools/TestImpact/Resolve-CiValidationReuse.ps1 -SelfTest
pwsh -NoProfile -File tools/TestImpact/Test-ValidationReuseModes.ps1
pwsh -NoProfile -File tools/TestImpact/Test-CiGateModes.ps1
pwsh -NoProfile -File tools/TestImpact/Test-TestImpactEndToEnd.ps1
```

All passed. The final end-to-end command was independently repeated after
integration, and executes the workflow and artifact-emission negative controls.
Its real temporary Git repositories demonstrate both paths:

| Change | Observed decision |
| --- | --- |
| Covered edit, including a PR behind master | 2 of 3 shards: Alpha and Always |
| Documentation-only PR | 0 of 3 shards |
| Landed runtime delta | Rerun Always; import Alpha from validated PR evidence |
| Landed documentation delta | Reuse validation; no reruns |
| Deleted test file | Retain the deleted file's routing evidence; successful process exit |
| Invalid map or CI-selection control change | Require full validation |

Twelve cases execute the production artifact-emission branch. Missing, expired,
wrong-SHA, wrong-slug, or wrong-case imported evidence declines partial reuse;
complete evidence permits it. Rerunning every shard needs no imported artifacts,
and whole-result reuse retains its separate existing contract. Internal policy
and invalid-input fixture modes use enums; string conversion occurs at the
JSON/workflow boundary.

## Empty-import follow-up

Two stricter negative controls reproduced unnecessary import metadata when every
PR shard was scheduled to rerun: one with no candidate artifacts and one with
otherwise valid but unused artifacts. The shared output writer now clears the
import run ID and SHA whenever the import list is empty. The rerun list and
validation/quality decisions stay unchanged; the existing workflow's non-empty
run-ID condition therefore cannot start an empty import's artifact download.
Partial reuse with real imports and whole-result reuse retain their existing
contracts. All five commands above and all 12 emission cases passed after this
follow-up, including the real-Git PR and post-merge paths.

These finite fixtures prove the reviewed wiring and decision rules. They do not
claim an arbitrary repository edit can never require the full matrix, or that
local fixtures substitute for reviewing the resulting GitHub checks.
