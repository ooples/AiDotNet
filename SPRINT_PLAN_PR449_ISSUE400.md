# Sprint Plan: PR #449 / Issue #400 (Physics-Informed ML)

## Sources Reviewed
- PR #449: https://github.com/ooples/AiDotNet/pull/449 (body + checklist)
- Issue #400: https://github.com/ooples/AiDotNet/issues/400 (problem statement + success criteria)

## Scope Summary
Issue #400 requests Physics-Informed Neural Networks (PINNs), Neural Operators, and Scientific ML components, plus standard PDE benchmarks (Burgers, Allen-Cahn), operator-learning benchmarks, and comparisons against traditional PDE solvers. PR #449 claims to implement the full PhysicsInformed stack and includes verification checklist items (build/test/coverage/TFM validation).

## Progress
- Sprint 0: completed (baseline alignment + AutoDiff/JIT integration plan).
- Sprint 1: completed (merge derivative helper into existing Autodiff + refactor PhysicsInformed/ScientificML).
- Sprint 2: completed (add PINN/DeepRitz/VPINN unit tests + benchmark tests vs solver baselines).
- Sprint 3: completed (net471 sharded test run + fixes + validation commits).

## Current State (as of this worktree)
PhysicsInformed and ScientificML derivative paths use the Autodiff helper with finite-difference fallback; unit tests and benchmark validations are in place, and net471 shards pass.

### Gaps vs Issue/PR Requirements
No open gaps identified after the PR/issue audit and net471 validation run.

## Definition of Done (DoD)
- All components from Issue #400 are fully implemented and tested.
- Benchmarks for Burgers and Allen-Cahn are available and produce reproducible results.
- Operator learning benchmarks run and show expected accuracy.
- Traditional PDE solver baseline comparisons are included (finite difference or spectral).
- Documentation reflects actual APIs and usage.
- PR #449 verification checklist satisfied (build/tests/coverage/TFM verification, no unresolved Copilot comments).
- Builds pass for `net471` and `net8.0` locally.

## Sprint Plan

### Sprint 0: Baseline Alignment and AutoDiff Integration Plan
**Goal:** Establish a clean baseline aligned with current conventions and decide Autodiff merge strategy.
- Confirm worktree isolation and re-audit PR/issue requirements for net471.
- Decide how to merge PhysicsInformed derivative helper into existing Autodiff (or replace with GradientTape-based implementation).
- Update sprint plan to reflect the revised implementation path.

**Deliverables:**
- Updated sprint plan doc.
- Early correctness fixes committed.

### Sprint 1: Autodiff Merge + ScientificML Alignment
**Goal:** Use existing Autodiff/JIT primitives for derivative computation where possible.
- Merge production-ready derivative logic into `src/Autodiff` as a helper (retain finite-difference fallback).
- Update PhysicsInformed and ScientificML classes to use the merged helper.
- Add targeted unit tests to cover derivative paths and ensure deterministic behavior.

**Deliverables:**
- Functional PINN/DeepRitz/VPINN training with optimizer updates.
- AutoDiff-first residual gradients and finite-difference fallback.

### Sprint 2: Unit + Benchmark Test Coverage
**Goal:** Add tests that satisfy Issue #400 success criteria.
- Unit tests for PINN/DeepRitz/VariationalPINN training (loss/parameter update behavior).
- Benchmark tests that compare model predictions to solver baselines (Burgers/Allen-Cahn + operator benchmarks).

**Deliverables:**
- Backprop-capable DeepONet/FNO/Graph operators.
- Vectorized/batched training loops.

### Sprint 3: Validation + net471 Coverage
**Goal:** Verify with sharded net471 test runs and finalize.
- Run `scripts/run-tests-sharded.ps1 -Framework net471 -Configuration Release -UseRunSettings`.
- Fix any remaining failures.
- Commit after each sprint.

**Deliverables:**
- HNN/LNN/UDE/Symbolic fully functional.
- net471 build verification and review thread resolution.

## Risks and Dependencies
- Higher-order Autodiff support is limited; plan to keep a finite-difference or analytic fallback for second derivatives.
- FFT remains custom for net471; ensure gradients remain stable and avoid unsupported JIT paths.
- Operator learning benchmarks need deterministic seeds for reproducibility.

## Acceptance Checklist
- [x] All Issue #400 items implemented.
- [x] All PR #449 checklist items validated (build, tests, TFM coverage).
- [x] Benchmarks and baselines added and documented with tests.
- [x] PINN/DeepRitz/VPINN unit tests added.
- [x] Autodiff merge completed and validated.
