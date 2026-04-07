# PR #1088 Failing Tests Report (2026-04-06)

## Summary
- **5 jobs failed** out of 35 (14 were cancelled/timed out — separate issue)
- **Total failing tests: 24** across 5 categories

---

## Category 1: RAG Embedding Model — ONNX fallback not used (21 tests)

**Root Cause**: `ONNXSentenceTransformer.EmbedCore()` throws `FileNotFoundException` instead of falling back to `GenerateFallbackEmbedding()` when the ONNX model file doesn't exist. The fallback path at line 117 is being bypassed.

**Error**: `System.IO.FileNotFoundException: ONNX model file not found: voyage-model-path.onnx`
**Source**: `ONNXSentenceTransformer.cs:117` — The `EmbedCore` method calls `Session` property which throws instead of using fallback.

**Affected Tests** (all same root cause):
1. `VoyageAIEmbeddingModelTests` — 11 tests
   - `Embed_WithValidText_ReturnsVectorOfCorrectDimension`
   - `Embed_ReturnsNormalizedVector`
   - `Embed_WithLongText_ReturnsEmbedding`
   - `Embed_WithSameTextTwice_ReturnsSameEmbedding`
   - `Embed_WithDifferentTexts_ReturnsDifferentEmbeddings`
   - `Embed_WithFloatType_WorksCorrectly`
   - `Embed_Deterministic_MultipleInstances`
   - `Embed_WithCustomDimension_ReturnsCorrectSize`
   - `EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions`
   - `EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls`
   - `EmbedBatch_AllRowsAreNormalized`

2. `MultiModalEmbeddingModelTests` — 7 tests
   - `Embed_WithValidText_ReturnsVectorOfCorrectDimension`
   - `Embed_WithNormalization_ReturnsNormalizedVector`
   - `Embed_WithoutNormalization_ReturnsUnnormalizedVector`
   - `Embed_WithFloatType_WorksCorrectly`
   - `Embed_Deterministic_MultipleInstances`
   - `Embed_WithCustomDimension_ReturnsCorrectSize`
   - `EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions`

3. `LocalTransformerEmbeddingTests` — 11 tests (same pattern)
   - `Embed_WithValidText_ReturnsVectorOfCorrectDimension`
   - `Embed_WithCustomDimension_ReturnsCorrectSize`
   - `Embed_Deterministic_MultipleInstances`
   - `Embed_WithSpecialCharacters_ReturnsEmbedding`
   - `Embed_WithSameTextTwice_ReturnsSameEmbedding`
   - `Embed_WithFloatType_WorksCorrectly`
   - `Embed_WithDifferentTexts_ReturnsDifferentEmbeddings`
   - `Embed_ReturnsNormalizedVector`
   - `Embed_WithLongText_ReturnsEmbedding`
   - `EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions`
   - `EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls`
   - `EmbedBatch_AllRowsAreNormalized`

**Fix**: The `VoyageAIEmbeddingModel`, `MultiModalEmbeddingModel`, and `LocalTransformerEmbeddingModel` need to override `EmbedCore` to call `EnsureModelLoaded()` / use fallback path, OR the base class `ONNXSentenceTransformer.EmbedCore` needs to correctly use fallback when model file is missing (it already has fallback code at line 117 but a derived class's `EmbedCore` may be calling `base.Embed()` which hits the `Session` property getter that throws).

---

## Category 2: MixtureOfExperts — Training/Gradient flow broken (3 tests)

**Root Cause**: MoE network training produces zero gradients — parameters don't change after training.

**Error Messages**:
- `No parameters changed after training — gradients may all be zero.`
- `Parameters did not change after training. Gradients may be zero or learning rate is 0.`
- `200 iterations loss (0.316869) > 50 iterations loss (0.312309). Optimizer may be diverging.`

**Affected Tests**:
1. `MixtureOfExpertsNeuralNetworkTests.GradientFlow_ShouldBeNonZeroAndFinite`
2. `MixtureOfExpertsNeuralNetworkTests.Training_ShouldChangeParameters`
3. `MixtureOfExpertsNeuralNetworkTests.MoreData_ShouldNotDegrade`

**Fix**: MoE layer `RegisterSubLayer` was added for router and experts but the backward pass may not be propagating gradients through the gating mechanism correctly. Need to verify the MoE backward pass properly distributes gradients to active experts.

---

## Category 3: ModelIndividual — Missing IParameterizable (2 tests)

**Root Cause**: `ModelIndividual<T,TInput,TOutput,TFitness>` does not implement `IParameterizable<T,TInput,TOutput>`. The `InterfaceGuard.Parameterizable()` check throws.

**Error**: `System.InvalidOperationException: ModelIndividual\`4 does not implement IParameterizable<Double, Double[], Double[]>. This operation requires a model with trainable parameters.`
**Source**: `InterfaceGuard.cs:19`

**Affected Tests**:
1. `ModelIndividualTests.DeepCopy_CreatesIndependentCopy`
2. `ModelIndividualTests.Clone_CreatesIndependentCopy`

**Fix**: The test's `MockModel` needs to implement `IParameterizable` (this was identified in the previous round — the summary mentions "MockModel now implements IParameterizable + IFeatureAware" but the fix may not have been committed/pushed to this PR branch).

---

## Category 4: Physics/ScientificML — Backward pass not called before UpdateParameters (4 tests)

**Root Cause**: `DenseLayer.UpdateParameters()` now requires a backward pass to be called first (sets gradients). The PhysicsInformed models call `UpdateParameters` directly without first calling backward on the layers.

**Error**: `System.InvalidOperationException: Backward pass must be called before updating parameters.`
**Source**: `DenseLayer.cs:1137` → `GradientBasedOptimizerBase.UpdateParameters()`

**Affected Tests**:
1. `NeuralOperatorTrainingTests.DeepOperatorNetwork_TrainUpdatesParameters`
2. `NeuralOperatorTrainingTests.FourierNeuralOperator_TrainUpdatesParameters` (also Assert.False failure)
3. `ScientificMLTests.UniversalDifferentialEquation_TrainUpdatesParameters`
4. `ScientificMLTests.HamiltonianNeuralNetwork_TrainUpdatesParameters`
5. `ScientificMLTests.LagrangianNeuralNetwork_TrainUpdatesParameters`

**Fix**: These models need to be migrated to `TrainWithTape()` like the other NN models were in the previous fix round, OR their `Train()` methods need to call `layer.Backward()` before `optimizer.UpdateParameters()`.

---

## Category 5: AiDotNet.Serving — FederatedCoordinator integration tests (3 tests)

**Root Cause**: Federated learning API endpoint returns HTTP 400 instead of expected behavior. Tests depend on the Serving integration test infrastructure (in-memory web host) and the `CreateRun` endpoint fails.

**Error**: `System.InvalidOperationException: CreateRun failed (HTTP 400)` / `System.Net.Http.HttpRequestException: Response status code does not indicate success: 400 (Bad Request)`

**Affected Tests**:
1. `FederatedCoordinatorIntegrationTests.FederatedRun_Lifecycle_FedAvg_AggregatesAndAdvancesRound`
2. `FederatedCoordinatorIntegrationTests.JoinRun_EnterpriseTier_WithNullAttestation_Returns403`
3. `FederatedCoordinatorIntegrationTests.GetParameters_WhenClientNotJoined_Returns403`

**Fix**: The Serving project's FederatedCoordinator endpoint may have a DB/config dependency issue in the test host setup, OR the federated run creation logic has a bug.

---

## Priority Order (by impact and fix difficulty)

1. **Category 3** (ModelIndividual) — 2 tests, simple fix (add IParameterizable to MockModel)
2. **Category 4** (Physics/ScientificML) — 5 tests, moderate fix (migrate to TrainWithTape or add backward calls)
3. **Category 2** (MoE) — 3 tests, moderate fix (debug gradient flow in MoE backward pass)
4. **Category 1** (RAG Embeddings) — 21 tests, single root cause fix (ONNX fallback path)
5. **Category 5** (Serving) — 3 tests, infrastructure issue (may be pre-existing)
