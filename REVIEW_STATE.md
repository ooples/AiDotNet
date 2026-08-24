
## #2029 CORRECTION (author = me; user corrected 2026-08-24)
SupportsFusedCompiledTraining was MY addition and it is WRONG: a lazy opt-out that
hides real problems. Requirement: ALL models must support fused compiled training.
No per-type overrides. So:
  1. DELETE the SupportsFusedCompiledTraining flag and every override of it.
  2. FIX the underlying reason each opt-out was added.
  3. master's RecurrentGemmaTrainingRegressionTests asserting GetFusedStepCount() > 0
     is CORRECT and must pass -- it was failing only because my flag opted the model out.
