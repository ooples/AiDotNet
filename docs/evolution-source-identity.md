# Exact-source identity for program evolution

Program identity must describe the source actually evaluated. Previously `ProgramGenome.Id` and value equality
used a display-normalized string that trimmed trailing whitespace and rewrote line endings. That can merge
different C# raw/verbatim or Python multiline string literals, causing a duplicate rejection or cached score
to stand in for a different program. See the [C# string literal rules](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/language-specification/lexical-structure).

The companion PR now hashes exact source text plus language, with an explicit `program-genome-v2-exact-source`
identity version. Description-only edits remain equal. Ill-formed Unicode is rejected before UTF-8 hashing,
preventing replacement encoding from collapsing distinct malformed inputs. `NormalizedSource` remains a
display/approximate-descriptor helper; it is not a semantic-equivalence proof.

Malformed model output follows the existing `ParseFailed` retry path, with bounded generic feedback, retained
chat-call/token accounting and the unchanged parent on exhaustion. It is never silently sanitized into a different
program. Variation v5 (`llm-program-variation-v5-protected-source-and-complete-options`) includes that retry fix,
rejects protected-text changes after every edit format, and fingerprints the previously omitted sampling,
prompt-history, descriptor-bin and diff-parser settings. Feature names and marker pairs are separately hashed,
not delimiter-joined strings. These identities describe the configured proposal policy; callers must still pin
the actual model/provider version and environment, not assume a mutable model alias is reproducible.

Given `EnforceEvolveBlocks`, when a full rewrite or diff changes any protected character or marker structure,
then it receives bounded `ParseFailed` feedback and cannot become a child. Multiple block bodies may grow or
shrink; protected mixed CR/LF/CRLF sequences are compared in the original text, not normalized region strings.
This protects edit boundaries, not the safety or semantic correctness of code inside them.

The executable full-rewrite path preserves content whitespace and mixed line endings, and requires a closed
fence. It removes CommonMark fence indentation and the single line terminator separating code from the closing
fence. To retain a source EOF newline, include an extra blank line before that fence. The existing public
`FencedCodeExtractor.Extract` display-oriented behavior is unchanged; evolution uses an internal exact-text path.

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

Program task v5 (`program-evolution-task-v5-cost-semantics-and-marker-identity`) additionally separates marker
identities and incorporates declared evaluation cost units. I/O evaluator v2
(`program-io-evaluator-v2-dispatched-attempt-costs`) counts the current dispatched call when it cancels, and
propagates fatal runner failures instead of assigning ordinary fitness. These also invalidate old task evidence.

Task v6 (`program-evolution-task-v6-measurement-origin`) subsequently preserves original sample provenance during
descriptor merging. Correctness gate v3 preserves fitness provenance while charging fresh checking work separately;
LLM judge v2 refuses unsupported provenance/score blending before a model call. These semantic versions deliberately
invalidate older task/gate/judge checkpoint identities; they do not rewrite old scores or claim statistical freshness.

This intentionally evaluates some cosmetic-only edits again. A language-aware parser may later prove safe
equivalence, but language-agnostic whitespace trimming cannot. Source identity is only one part of applicability:
compiler/runtime, dependencies, public and held-out test sets, environment and fidelity must also match before
cross-run reuse or deployment. This patch does not claim a complete dependency fingerprint or sandbox boundary.

## Local regression evidence

On September 10, 2026, ten new regressions failed against revision `75aa6b1`: nine option changes left the
variation compatibility hash unchanged, and a full rewrite changed protected code. After the v5 fixes,
1,082 authored Evolution/facade tests passed separately on .NET 10 and .NET 8, including 40 new cases.
The local focused harness compiled the existing test sources against the real built library; production and
test-copy DLL hashes matched. Diagnostic analyzers were disabled, so this is not whole-repository or analyzer CI.
The new protected-text helper had 100% line/branch coverage; fenced extraction had 99.1% line/92.72% branch
coverage on .NET 10. Hosted validation and final review remain separate requirements.
