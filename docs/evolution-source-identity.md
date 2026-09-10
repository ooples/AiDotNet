# Exact-source identity for program evolution

Program identity must describe the source actually evaluated. Previously `ProgramGenome.Id` and value equality
used a display-normalized string that trimmed trailing whitespace and rewrote line endings. That can merge
different C# raw/verbatim or Python multiline string literals, causing a duplicate rejection or cached score
to stand in for a different program. See the [C# string literal rules](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/language-specification/lexical-structure).

The companion PR now hashes exact source text plus language, with an explicit `program-genome-v2-exact-source`
identity version. Description-only edits remain equal. Ill-formed Unicode is rejected before UTF-8 hashing,
preventing replacement encoding from collapsing distinct malformed inputs. `NormalizedSource` remains a
display/approximate-descriptor helper; it is not a semantic-equivalence proof.

Given two programs with different meaningful string whitespace, when canonicalized, then they have different
identities even if their display-normalized text is equal. Given an exact duplicate with a new description,
when canonicalized, then it retains the same identity and can reuse evaluation evidence.

The proposal operator uses exact source for unchanged checks and the configured source-size bound counts all
characters actually sent to the evaluator. Output schema v2 records the SHA-256 of the complete exact UTF-8 source;
if the saved file is truncated, its truncation flag remains mandatory and that partial file must not be deployed.
The hash is of the complete source, not a claim that a truncated artifact is executable or validated.

## Compatibility and limits

Task, variation and codec compatibility versions are bumped. Old checkpoints must not silently resume under the
new identity semantics. The genome payload shape remains v1 because it already preserves exact source;
deserializing a standalone old payload computes the new identity but does not migrate old archived evaluations.
Start a new run or perform an explicitly validated migration/re-evaluation; do not relabel old scores.

This intentionally evaluates some cosmetic-only edits again. A language-aware parser may later prove safe
equivalence, but language-agnostic whitespace trimming cannot. Source identity is only one part of applicability:
compiler/runtime, dependencies, public and held-out test sets, environment and fidelity must also match before
cross-run reuse or deployment. This patch does not claim a complete dependency fingerprint or sandbox boundary.
