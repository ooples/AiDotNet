# PR #1088 CI Failures — Round 2 (2026-04-08)

## Summary
- 17 failing test jobs
- ~8 jobs timed out at 45 min (separate discussion)
- ~9 jobs have actual test failures

## Category 1: Timeout Attribute on Non-Async Tests (8 tests)
**Error**: `Tests marked with Timeout are only supported for async tests`
**Root Cause**: xUnit 2.x [Timeout] attribute only works on async Task tests, not sync void tests.

### Affected Tests:
1. EfficientNetTests.EfficientNetB0_Forward_ProducesCorrectOutputShape
2. EfficientNetTests.EfficientNetB0_ForwardSmallInput_ProducesCorrectShape
3. MobileNetTests.MobileNetV3Large_Forward_ProducesCorrectOutputShape
4. MobileNetTests.MobileNetV3Small_Forward_ProducesCorrectOutputShape
5. MobileNetTests.MobileNetV2_Forward_ReturnsNonZeroOutput
6. MobileNetTests.MobileNetV2_Forward_ProducesCorrectOutputShape
7. PagedAttentionServerTests.PagedAttentionServer_ForModel_CreatesValidServer
8. PlaygroundExampleCompilationTests.AllPlaygroundExamples_ShouldCompile

**Fix**: Make these test methods async Task, or remove the [Timeout] attribute.

## Category 2: MoE Gradient Flow — Still Zero (2 tests)
**Error**: `No parameters changed after training — gradients may all be zero`
**Root Cause**: Issue #1080 — MoE tape compatibility. Our TensorGather fix was applied but gradients still zero. Deeper investigation needed.

### Affected Tests:
1. MixtureOfExpertsNeuralNetworkTests.GradientFlow_ShouldBeNonZeroAndFinite
2. MixtureOfExpertsNeuralNetworkTests.Training_ShouldChangeParameters

## Category 3: FourierNeuralOperator — Training Still Broken (1 test)
**Error**: `Assert.False() Failure — Expected: False, Actual: True`
**Root Cause**: `before.SequenceEqual(after)` returns true — parameters didn't change after Train().

### Affected Tests:
1. NeuralOperatorTrainingTests.FourierNeuralOperator_TrainUpdatesParameters

## Category 4: SparseLinearLayer SupportsTraining (1 test)
**Error**: `Assert.True() Failure — Expected: True, Actual: False`
**Root Cause**: SparseLinearLayer.SupportsTraining returns false.

### Affected Tests:
1. AdvancedAlgebraLayerTests.SparseLinearLayer_SupportsTraining_IsTrue

## Category 5: MappedRandomForestModel — Missing IParameterizable (3 tests)
**Error**: `Trained model should have learnable parameters` / `should have at least one active feature` / `wrong sign`
**Root Cause**: MappedRandomForestModel doesn't implement IParameterizable or IFeatureAware properly.

### Affected Tests:
1. MappedRandomForestModelTests.Parameters_ShouldBeNonEmpty_AfterTraining
2. MappedRandomForestModelTests.ActiveFeatureIndices_ShouldBeValid
3. MappedRandomForestModelTests.CoefficientSigns_ShouldMatchDataGeneratingProcess

## Category 6: OrthogonalRegression — Negative R² (1 test)
**Error**: `Builder pipeline R² = -1.4823 — should be positive on linear data`
**Root Cause**: OrthogonalRegression not fitting properly through the builder pipeline.

### Affected Tests:
1. OrthogonalRegressionTests.Builder_R2ShouldBePositive

## Category 7: HDBSCAN — Clustering Failures (2 tests)
**Error**: `Builder pipeline ARI = 0.0000` / `Cluster mean 8.46 away from nearest center`
**Root Cause**: HDBSCAN not clustering properly through the builder pipeline.

### Affected Tests:
1. HDBSCANTests.Builder_ClusteringShouldBeatRandom
2. HDBSCANTests.ClusterMeans_ShouldBeNearGenerationCenters

## Category 8: DeepGaussianProcess — Uncertainty Issues (2 tests)
**Error**: `95% CI coverage = 40%` / `More data did not reduce variance`
**Root Cause**: DeepGP uncertainty calibration and variance reduction bugs.

### Affected Tests:
1. DeepGaussianProcessTests.ConfidenceInterval_ShouldCoverTruth_AtMostTestPoints
2. DeepGaussianProcessTests.MoreData_ShouldReducePredictiveVariance

## Category 9: NBEATS — Catastrophic Predictions (7 tests)
**Error**: `R² = -5294` / `R² = -611666` / `Mean residual = 5390`
**Root Cause**: NBEATS model producing catastrophically wrong predictions. Training not working at all.

### Affected Tests:
1. NBEATSModelTests.Builder_R2ShouldBePositive
2. NBEATSModelTests.ResidualMean_ShouldBeNearZero
3. NBEATSModelTests.MoreData_ShouldNotDegrade_R2
4. NBEATSModelTests.TrainingError_ShouldNotExceedTestError
5. NBEATSModelTests.TrendRecovery_LaterTimeShouldHaveHigherPrediction
6. NBEATSModelTests.R2_ShouldBePositive_OnTrendData
7. NBEATSModelTests.TranslationEquivariance_ShiftingTargets_ShiftsPredictions

## Timeout Jobs (separate discussion)
~8 jobs cancelled at 45 min mark — likely test suites taking too long, not actual failures:
- ModelFamily - Diffusion A-I, J-R, S-Z
- ModelFamily - Generated Layers, NeuralNetworks
- Unit - 03 Diffusion/Encoding
- Unit - 08a NN-Classic, 08c NN-VLM, 08e NN-Remaining

## Priority Order
1. **Category 1** (Timeout attr) — 8 tests, trivial fix (make async or remove attr)
2. **Category 4** (SparseLinear) — 1 test, likely simple property fix
3. **Category 5** (MappedRandomForest) — 3 tests, interface implementation
4. **Category 6** (OrthogonalRegression) — 1 test, builder pipeline issue
5. **Category 2** (MoE) — 2 tests, deep tape chain issue
6. **Category 3** (FNO) — 1 test, training not working
7. **Category 7** (HDBSCAN) — 2 tests, clustering algorithm bug
8. **Category 8** (DeepGP) — 2 tests, uncertainty calibration
9. **Category 9** (NBEATS) — 7 tests, training completely broken
