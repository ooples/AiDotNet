# US-08 consumer evidence

Source `8adb2adf1`; companion to [core #51](https://github.com/ooples/AiDotNet.Evolution/pull/51), based on consumer US-06 #2210. Hosted compiler CI pins core `8d28a72d72bf48cb75ceadc6a70a77844e657ac7`. This source-path check does not claim that the older published preview contains the new APIs.

[verification.zip](verification.zip): 697,148 bytes; SHA-256 `33cb54d0d6c17c4db68504ec06205f5016300f4cceeccd4a8bcbf77ced6fbbda`.

- Full compiler/worker dependency build for net10/net8: 0 errors, 5,619 existing generator/library warnings retained.
- Final compiler/portfolio tests: **83 net10 / 83 net8**, no failures/skips.
- Consumer integration: **954 net10 / 954 net8**, no failures/skips. Integration test assemblies reused the already built dependencies (`BuildProjectReferences=false`), rather than compiling the entire consumer again.
- Touched-file whitespace verification passed, with a workspace-loading warning retained.
- Local consumer builds began against the core notification implementation used by study revision `f4d88f2`; the later legacy-default-credit guard is independently covered by core's final 792/719/719 tests. Hosted CI uses the final pinned core revision above for a clean integration build. Do not label mixed local build provenance as a single frozen artifact set.

## Adversarial review

Matching unit strings was insufficient: independent ledgers could multiply the total budget. `IProgramResourceLedgerProvider` now binds every arm and evaluator to the same live ledger; constructor/facade regressions reject mismatches and missing accounting.

The real facade test uses two scripted-provider compiler arms. One emits valid C#, another exhausts a repair. Three model requests, one repair, compilation/parsing/audit costs and terminal outcome attribution are reconciled without network/model spending. Its first attempt correctly failed admission because the fixture had only one compiler's conservative reference-byte allowance. Increasing the fixture allowance for two compiler setups made it pass; production admission was not weakened. Failed TRX and corrected final TRX are both retained.

Additional tests cover five supplied strategy identities, pending-cost checkpoint restore, future deterministic proposals, cost-unit mismatch, duplicate children, unmetered LLM rejection and cumulative usage. Those five identity fixtures test adapter contracts, not the quality of five real LLM algorithms; core's separate study measures actual mutation/crossover/restart/refinement workloads.

`CreditCommitted` is an in-process once-per-commit notification, not transactional exactly-once delivery to an external service. Sink failures are counted, callbacks cannot reenter learning/checkpoint operations, and restore does not replay old notifications. Automatic facade resume with live accounting still refuses uncoordinated ledger/engine state. The core engine's coordinated-boundary checkpoint test remains the replay proof.

No Tensors changes, package publication, reviews on the user's behalf, merges or default promotion. See [usage and boundaries](../../evolution/PROGRAM_PORTFOLIOS.md).
