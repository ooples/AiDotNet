---
title: "NeuralNetworkBase<T>"
description: "Base class for all neural network implementations in AiDotNet."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks`

Base class for all neural network implementations in AiDotNet.

## For Beginners

A neural network is a computing system inspired by the human brain. It consists of 
interconnected "layers" of artificial neurons that process information and learn patterns from data.
This class provides the foundation for building different types of neural networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralNetworkBase(ILossFunction<>,Double)` | Creates a layer-only neural network with no semantic architecture. |
| `NeuralNetworkBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Creates a new neural network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AcceleratedMemoHits` | Count of value-memo hits (identical-input short-circuits) on the accelerated path. |
| `AcceleratedVerifyCount` | Count of verify-then-trust verifications performed (one per new shape+version). |
| `AiDotNet#Interfaces#ILayeredModel{T}#Layers` | Gets the ordered list of layers in this model (explicit interface implementation). |
| `AutoCompiledInferenceEngaged` | Whether the compiled-inference path is currently engaged for this model. |
| `CanTrainOnGpu` | Gets whether GPU-resident training can be used right now. |
| `DefaultLossFunction` | Gets the default loss function for this network. |
| `DefaultStreamingThresholdParamsForReport` | Public-readable view of the auto-detect threshold for telemetry callers (e.g. |
| `Engine` | Gets the global execution engine for vector operations. |
| `FastApproxGradClip` | #1662 lever #1 (§5c) — opt-in single-pass approximate gradient clipping for the streaming (optimizer-in-backward) path. |
| `GpuEngine` | Gets the GPU tensor engine when available, or null if not using GPU. |
| `GradientCheckpointingSegmentSize` | Per-instance gradient-checkpointing segment size. |
| `InferenceAccelerationEngaged` | Whether the verify-then-trust compiled-inference path is currently engaged for this model. |
| `IsGradientCheckpointingEnabled` | Gets whether gradient checkpointing is enabled. |
| `IsLayerOnlyModel` | True when the model was constructed via the layer-only ctor (architecture is a stub with no semantic input contract). |
| `IsMemoryManagementEnabled` | Gets whether memory management (gradient checkpointing/pooling) is enabled. |
| `IsMixedPrecisionEnabled` | Gets whether mixed-precision training is enabled. |
| `IsTrainingMode` | Indicates whether the network is currently in training mode. |
| `IsWeightStreamingActive` | True when the model is currently in streaming mode for ANY reason (auto-detect or explicit). |
| `LayerCount` | Gets the number of layers in this neural network. |
| `LayerStructureVersion` | Gets the current layer structure version for cache invalidation. |
| `Layers` | Gets the collection of layers that make up this neural network (read-only access). |
| `LayersReadOnly` | Gets the collection of layers that make up this neural network (internal read-only access). |
| `MaxGradNormT` | Backwards-compatible `T`-typed accessor for code that historically read the protected field directly. |
| `MaxGradNormValue` | Maximum allowed global L2 norm for gradients per training step, exposed as a `double`. |
| `Options` | Configuration options for this neural network model. |
| `ParameterCount` | Gets the total number of parameters in the model. |
| `Random` | Gets the thread-safe random number generator for initialization. |
| `StreamingTraining` | Performs tape-based forward/backward pass and delegates the parameter update to the provided optimizer via `TapeStepContext{`. |
| `StreamingTrainingLearningRate` | Learning rate used by the default streaming 8-bit Adam epilogue. |
| `StreamingTrainingWeightDecay` | Decoupled weight decay used by the default streaming Adam epilogue. |
| `SupportsGpuTraining` | Gets whether all layers in the network support GPU-resident training. |
| `SupportsParameterInitialization` |  |
| `SupportsTraining` | Indicates whether this network supports training (learning from data). |
| `UseCopyOnWriteDeepCopy` | Creates a deep copy of the neural network. |
| `WeightStreamingAutoDetected` | Returns true iff streaming was engaged by auto-detect on THIS instance (vs. |
| `WeightStreamingResidentBytes` | Live read of the streaming pool's resident-bytes counter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Accelerate(Tensor<>,Func<Tensor<>>)` | Opt a model's forward into the #1622 verify-then-trust compiled-inference acceleration with a single wrap. |
| `AcquireTrainSentinel` | Acquires the per-instance training sentinel. |
| `AddBatchNormalizationLayer(Int32,Double,Double)` | Adds a batch normalization layer to the neural network. |
| `AddConvolutionalLayer(Int32,Int32,Int32,ActivationFunction)` | Adds a convolutional layer to the neural network. |
| `AddDropoutLayer(Double)` | Adds a dropout layer to the neural network. |
| `AddLSTMLayer(Int32,Boolean)` | Adds an LSTM layer to the neural network. |
| `AddLayer(LayerType,Int32,ActivationFunction)` | Adds a layer to the neural network. |
| `AddPoolingLayer(Int32[],PoolingType,Int32,Nullable<Int32>)` | Adds a pooling layer to the neural network. |
| `AlignTargetToOutputShape(Tensor<>,Tensor<>)` | Sets which input features should be considered active in the neural network. |
| `AnyLayerHasUnmaterializedParameters` | True if any (possibly nested) layer still holds an unallocated placeholder trainable parameter (a `Length == 0` tensor). |
| `AnyLayerNeedsShapeResolution` | Returns `true` if any layer (including nested sub-layers) reports `IsShapeResolved` = `false`, i.e., hasn't yet seen a forward pass and is still carrying placeholder `Tensor<T>.Empty()` weight refs. |
| `ApplyAutoDetectThresholdOverride(Int64)` | Sets a per-instance threshold for auto-detect. |
| `ApplyGradientClipping(Dictionary<Tensor<>,Tensor<>>,Double,IReadOnlyList<Tensor<>>)` | Best-effort read of the supplied optimizer's current learning rate, used by the network-level extras update path. |
| `ApplyGradients(Vector<>,)` | Applies a flattened gradient vector to update the network's parameters. |
| `AreLayersCompatible(ILayer<>,ILayer<>)` | Checks if two consecutive layers can be connected in a neural network. |
| `BackwardAndStepOnPrecomputedLoss(GradientTape<>,Tensor<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Variant of `Tensor{` that runs backward+optim on a loss tensor whose forward pass was already recorded on a caller-owned `GradientTape`. |
| `BeginGpuExecution(GpuExecutionOptions)` | Begins a GPU execution context for managing GPU-resident tensor lifecycle. |
| `BeginLayerMaterializeScope(Int32)` | Returns an `IDisposable` that pins the given layer's trainable tensors as resident in the streaming pool for the scope's lifetime. |
| `CanUseGpuResidentPath` | Checks if all layers in the network support GPU execution. |
| `ClearLayers` | Clears all layers from the internal layers collection and invalidates the parameter count cache. |
| `ClipGradient(Tensor<>)` | Clips the gradient tensor if its norm exceeds the maximum allowed gradient norm. |
| `ClipGradient(Vector<>)` | Clips the gradient vector if its norm exceeds the maximum allowed gradient norm. |
| `ClipGradients(List<Tensor<>>)` | Applies gradient clipping to prevent exploding gradients. |
| `ClipTensorGradient(Tensor<>,)` | Clips a single gradient tensor if its norm exceeds the specified maximum norm. |
| `Clone` | Creates a clone of the neural network. |
| `CompileForward(Tensor<>)` | Eagerly traces and compiles the forward pass for the given input shape, storing the compiled plan in the per-instance cache. |
| `CompiledVerdictForTesting(Int32[])` | Returns the current verify-then-trust verdict for an input shape at the live structure version: 0 = unknown/not-yet-verified, 1 = trusted (compiled replayed), 2 = rejected (eager). |
| `ComputeGradients(Tensor<>,Tensor<>,ILossFunction<>)` | Computes a flattened gradient vector for all trainable parameters in the network. |
| `ComputeShapeKey(Int32[])` | Computes a deterministic 64-bit shape key (FNV-1a) for the bad-compile cache. |
| `ConfigureFairness(Vector<Int32>,FairnessMetric[])` | Configures fairness evaluation settings. |
| `ConfigureWeightLifetime(GpuOffloadOptions,IGpuOffloadAllocator)` | Opts the model into the AiDotNet.Tensors weight-lifetime machinery — streaming pool, pinned-host, and GPU offload — so models larger than RAM can run without OOMing. |
| `CreateNewInstance` | Creates a new instance of the same type as this neural network. |
| `Deserialize(Byte[])` | Deserializes the neural network from a byte array. |
| `DeserializeInternalUnchecked(Byte[])` | Internal, non-virtual, no-guard deserialization used by trusted framework call sites such as `DeepCopy`. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data that was not covered by the general deserialization process. |
| `DetachFromArena(Tensor<>)` | Copies `output` out of the active `TensorArena` into a GC-owned tensor so it survives the arena's `Dispose` (which recycles Rent-backed buffers). |
| `DisableAutoStreaming` | Opts this model OUT of auto-streaming detection. |
| `DisableMemoryManagement` | Disables memory management and releases associated resources. |
| `DisableMixedPrecision` | Disables mixed-precision training and releases associated resources. |
| `Dispose` | Disposes resources used by the neural network. |
| `Dispose(Boolean)` | Protected Dispose pattern implementation. |
| `DownloadWeightsFromGpu` | Downloads all layer weights from GPU back to CPU. |
| `EmitFusedMissAndFallback(String)` | Attempts the fused-compiled training path — forward + backward + fused optimizer update all in one compiled kernel. |
| `EnableDeterministicMode` | Enables deterministic CPU inference by setting MKL to single-threaded. |
| `EnableInferenceAcceleration(Boolean)` | Opts this model into the #1622 verify-then-trust compiled-inference path regardless of size, the industry-standard explicit "compile this model" gesture (cf. |
| `EnableMemoryManagement(TrainingMemoryConfig)` | Enables memory management with the specified configuration. |
| `EnableMethod(InterpretationMethod[])` | Enables specific interpretation methods. |
| `EnsureArchitectureInitialized` | Ensures the architecture is initialized before training begins. |
| `EnsureBatchForCnnTraining(Tensor<>,Tensor<>)` | Normalizes a (input, target) pair for CNN training loops: when input is a single rank-3 `[C,H,W]` sample, adds a batch dim so downstream layers see `[1,C,H,W]`. |
| `EnsureLayerRandomSeedsWired` | Runs `WireLayerRandomSeeds` exactly once, before the first training forward. |
| `EstimateStructuralParameterCount` | Structural estimate of this model's trainable parameter count, computed from its architecture/options WITHOUT materializing any lazy weight tensors. |
| `ExtractSingleExample(Tensor<>,Int32)` | Extracts a single example from a batch tensor and formats it as a tensor with shape [1, features]. |
| `ExtractSubModel(Int32,Int32)` | Extracts a contiguous sub-model from `startLayer` to `endLayer` (inclusive). |
| `FindNextWeightedLayerAfter(Int32,Int32)` | Returns the index of the `stepsAhead`th weighted layer (one with at least one non-empty trainable tensor) at or after `fromIndex`. |
| `FoldBatchNormForInference` | Freeze-time BatchNorm folding (compiled-inference, Phase 6). |
| `ForceAutoCompiledInferenceForTesting(Boolean)` | Forces the verify-then-trust compiled inference path on/off for this model. |
| `FormatShape(Int32[])` | Renders a shape array as `[d0, d1, ...]` for validator error messages. |
| `ForwardDeferred(Tensor<>)` | Performs a forward pass using deferred execution for optimized GPU performance. |
| `ForwardDeferredAsync(Tensor<>,CancellationToken)` | Performs an asynchronous forward pass using deferred execution for optimized GPU performance. |
| `ForwardGpu(Tensor<>)` | Performs a GPU-resident forward pass, keeping intermediate results on GPU. |
| `ForwardGpu(Tensor<>,IReadOnlyDictionary<String,Tensor<>>)` | Performs a forward pass through the network entirely on GPU. |
| `ForwardWithCheckpointing(Tensor<>)` | Performs forward pass with gradient checkpointing to reduce memory usage. |
| `ForwardWithFeatures(Tensor<>,Int32[])` | Performs a forward pass and returns intermediate layer activations for feature extraction. |
| `ForwardWithGpuContext(Tensor<>)` | Performs a GPU-resident forward pass within a GPU execution context. |
| `ForwardWithMemory(Tensor<>)` | Performs a forward pass through the network while storing intermediate values for backpropagation. |
| `ForwardWithMemory(Tensor<>,IReadOnlyDictionary<String,Tensor<>>)` | Forward pass with named auxiliary inputs that are routed to layers declaring matching ports. |
| `GenerateTextExplanationAsync(Tensor<>,Tensor<>)` | Generates a text explanation for a prediction. |
| `GetActiveFeatureIndices` | Gets the indices of input features that are actively used by the network. |
| `GetAllLayerInfo` | Gets metadata for all layers, including parameter offsets, types, shapes, names, and cost estimates. |
| `GetAnchorExplanationAsync(Tensor<>,)` | Gets anchor explanation for a given input. |
| `GetArchitecture` | Gets the architectural structure of the neural network. |
| `GetCounterfactualAsync(Tensor<>,Tensor<>,Int32)` | Gets counterfactual explanation for a given input and desired output. |
| `GetDeepLIFTAsync(Tensor<>,Tensor<>,Boolean)` | Gets DeepLIFT attributions for a neural network prediction. |
| `GetDynamicShapeInfo` |  |
| `GetExpectedUnbatchedInputRankInternal` | Computes the effective unbatched input rank from the architecture's input dimensions. |
| `GetExtraTrainableLayers` | Override-point for subclasses that own trainable layers outside `Layers` (e.g. |
| `GetExtraTrainableTensors` | Override-point for subclasses that own raw trainable `Tensor` parameters directly on the network (NOT inside any layer) — for example a Vision Transformer's `cls_token` and `positional_embeddings`. |
| `GetFeatureImportance` | Gets the feature importance scores for the model. |
| `GetFeatureInteractionAsync(Int32,Int32)` | Gets feature interaction effects between two features. |
| `GetGlobalFeatureImportanceAsync` | Gets the global feature importance across all predictions. |
| `GetGpuMemoryStats` | Gets GPU memory statistics if running within a GPU execution context. |
| `GetGradCAMAsync(Tensor<>,Int32)` | Gets GradCAM visual explanation for a CNN prediction. |
| `GetGradients` | Gets the gradients from all layers in the neural network. |
| `GetInputShape` | Gets the input shape expected by the neural network. |
| `GetIntegratedGradientsAsync(Tensor<>,Tensor<>,Int32)` | Gets Integrated Gradients attributions for a neural network prediction. |
| `GetLastLoss` | Gets the loss value from the most recent training iteration. |
| `GetLayerActivations(Tensor<>)` | Gets the activations (outputs) from each layer for a given input. |
| `GetLayerInfo(Int32)` | Gets metadata for a specific layer including its parameter offset within the flat parameter vector. |
| `GetLimeExplanationAsync(Tensor<>,Int32)` | Gets LIME explanation for a specific input. |
| `GetLocalFeatureImportanceAsync(Tensor<>)` | Gets the local feature importance for a specific input. |
| `GetMemoryEstimate(Int32,Int32)` | Gets memory usage statistics if memory management is enabled. |
| `GetMixedPrecisionContext` | Gets the mixed-precision training context (if enabled). |
| `GetModelMetadata` | Gets the metadata for this neural network model. |
| `GetModelSpecificInterpretabilityAsync` | Gets model-specific interpretability information. |
| `GetNamedLayerActivations(Tensor<>)` | Gets the intermediate activations from each layer when processing the given input with named keys. |
| `GetOptions` |  |
| `GetOrCreateBaseOptimizer` | Gets or lazily creates the default optimizer for tape-based training. |
| `GetOrCreateParameterBuffer(IReadOnlyList<Tensor<>>)` | Gets or lazily creates the contiguous parameter buffer from the current trainable parameters. |
| `GetOutputShape` |  |
| `GetParameterChunks` |  |
| `GetParameterCount` | Gets the total number of parameters in the model. |
| `GetParameterGradients` | Retrieves the gradients for all trainable parameters in the network. |
| `GetParameters` | Gets all trainable parameters of the network as a single vector. |
| `GetPartialDependenceAsync(Vector<Int32>,Int32)` | Gets partial dependence data for specified features. |
| `GetShapValuesAsync(Tensor<>)` | Gets SHAP values for the given inputs. |
| `HasCrossBatchNormalization` | Whether any layer reachable from this model performs cross-batch normalization (a BatchNorm layer), whose statistics depend on the whole batch — making G8 micro-batching non-equivalent. |
| `InitializeLayers` | Initializes the layers of the neural network based on the architecture. |
| `InsertLayerIntoCollection(Int32,ILayer<>)` | Inserts a layer into the internal layer collection and invalidates the parameter count cache. |
| `InvalidateLayerInfoCache` | Invalidates the cached layer info so that `GetAllLayerInfo` recomputes layer metadata on the next call. |
| `InvalidateParameterCountCache` | Invalidates the parameter count cache. |
| `InvalidateWeightCachesAfterSuccessfulWeightUpdate` | Invalidates EVERY identity-keyed weight cache (GPU weight-buffer uploads AND the CPU engine's derived-weight caches) after a successful in-place weight update. |
| `IsFeatureUsed(Int32)` | Determines if a specific input feature is actively used by the network. |
| `IsGpuTransientFailure(Exception)` | True if the exception chain indicates a transient GPU/CUDA fault (device error, failed host/device copy, or the activation-cache deferred-materializer race) — as opposed to a logical cause (shape drift, hyperparameter change). |
| `IsIdentityActivation(ILayer<>)` | A layer's activation is the identity iff it applies no activation or an `IdentityActivation` (folding across a real nonlinearity would be incorrect). |
| `IsValidInputLayer(ILayer<>)` | Determines if a layer can serve as a valid input layer for the neural network. |
| `IsValidOutputLayer(ILayer<>)` | Determines if a layer can serve as a valid output layer for the neural network. |
| `LayerHasWeights(Int32)` | True iff the layer at `layerIndex` exposes at least one non-empty trainable tensor. |
| `LoadModel(String)` | Loads a neural network model from a file. |
| `LoadState(Stream)` | Loads the model's state from a stream. |
| `MarkTrainMutationStarted` | Marks that an in-place training write (optimizer.Step / streaming Apply / legacy UpdateParameters) is about to begin, so a subsequent OOM is rethrown rather than retried from partially-mutated state. |
| `NormalizeBatchDim(Tensor<>,Tensor<>)` | Universal rank-N → rank-(N+1) batch-dim promotion. |
| `NormalizeInputBatchDim(Tensor<>)` | Read-only counterpart to `Tensor{` for the inference path: only the input is shape-normalized; targets aren't involved in Predict. |
| `OnReleaseCompiledPlans` | Extension hook for `ReleaseCompiledPlans`. |
| `Predict(Tensor<>)` | Inference entry point. |
| `PredictAccelerated(Tensor<>)` | Verify-then-trust compiled inference for foundation-scale models, with a collision-safe value-hash memo. |
| `PredictCompiled(Tensor<>)` | Executes the forward pass using a compiled plan for maximum performance. |
| `PredictCore(Tensor<>)` | Makes a prediction using the neural network. |
| `PredictEager(Tensor<>)` | Eager forward pass through all layers. |
| `PredictInBatches(Tensor<>,Int32)` | Chunked inference path: splits `input` along axis 0 into slices of size `batchSize`, runs `Tensor{` on each slice, and concatenates the per-slice outputs back along axis 0. |
| `PredictWithContext(Tensor<>,InferenceForwardContext)` | Inference forward pass that threads a per-call `InferenceForwardContext` to context-aware layers (paged/cached attention), enabling concurrent multi-sequence decode over a shared KV cache (#99). |
| `PrefetchLayerWeights(Int32)` | Hook into `WeightRegistry.PrefetchAsyncMany` for the given layer's trainable tensors. |
| `PromoteToBatchedTensor(Tensor<>)` | Promotes a rank-3 `[C,H,W]` tensor to rank-4 `[1,C,H,W]`. |
| `RefreshStreamingSchedule` | Feeds the streaming pool this network's per-step weight-access SCHEDULE — forward layer order then backward (training) — so it pages with Belady-optimal eviction (evict the weight whose next use is furthest) instead of LRU. |
| `RefreshWeightRegistry` | Re-walks `Layers` and registers any trainable tensor whose length is now positive (i.e., the layer's lazy weights resolved during a forward pass after `IGpuOffloadAllocator)` was called). |
| `RegisterLayerTrainableTensorsWithWeightRegistry(Int32)` | Registers a single layer's trainable tensors with the weight registry. |
| `RegisterTrainableTensorsWithWeightRegistry` | Walks `Layers` and registers each layer's trainable tensors with the process-wide `WeightRegistry`. |
| `ReleaseCompiledPlans` | Releases every compiled inference plan this network holds, returning their pre-allocated intermediate buffers to the GC. |
| `RemoveLayerFromCollection(ILayer<>)` | Removes a layer from the internal layers collection and invalidates the parameter count cache. |
| `ReplaceParametersFromMap(IEnumerable<ILayer<>>,Dictionary<Tensor<>,Tensor<>>,HashSet<ILayer<>>)` | Recursively replaces trainable parameter tensors with buffer-backed views using a pre-built parameter→view map. |
| `ResetState` | Resets the internal state of the different layers, clearing any remembered information. |
| `ResetWeightStreamingForTests` | Forces the process-wide WeightRegistry into a clean state — drops all registered entries, disposes the streaming pool, clears any offload allocator. |
| `ResolveAvailableMemoryForStreaming` | Best-effort query of the memory ceiling the process can actually use — the GC heap hard limit when one is configured (containers / constrained CI runners), else physical RAM. |
| `ResolveInferenceStoreDtype(Int64,Int64)` | Quant-resident inference store selection (Tier 1 / AiDotNet#1622). |
| `ResolveLazyLayerShapes` | Walks `Layers` in order, propagating concrete input shapes through the chain so every lazy layer has its `InputShape` / `OutputShape` resolved before any Forward / GetParameters / ParameterCount call. |
| `RestoreOriginalParameters` | Restores original tensor references on all layers after a training step. |
| `SafeEstimateStructuralParameterCount` | Calls `EstimateStructuralParameterCount` defensively: the override runs during construction (before subclass fields may be fully set) and must never break auto-detect, so any failure degrades to 0 ("no estimate"). |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` | Saves the model to a file. |
| `SaveOriginalParameters(IEnumerable<ILayer<>>,Dictionary<ILayer<>,IReadOnlyList<Tensor<>>>,HashSet<ILayer<>>)` | Saves original tensor references from all trainable layers before buffer view replacement. |
| `SaveState(Stream)` | Saves the model's current state to a stream. |
| `Serialize` | Serializes the neural network to a byte array. |
| `SerializeInternalUnchecked` | Internal, non-virtual, no-guard serialization used by trusted framework call sites such as `DeepCopy`. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data that is not covered by the general serialization process. |
| `SetAutoCompiledInferenceEnabledForTesting(Boolean)` | Test/diagnostic hook: overrides the process-wide compiled-inference opt-in (`s_autoCompiledInferenceEnabled`) in-process, since the env var is read once at type load. |
| `SetBaseModel(IFullModel<,,>)` | Sets the base model for interpretability analysis. |
| `SetBaseTrainOptimizer(IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Pre-wires the optimizer instance that `GetOrCreateBaseOptimizer` will return on subsequent calls. |
| `SetParameters(Vector<>)` | Sets the parameters of the neural network. |
| `SetTrainingMode(Boolean)` | Sets the neural network to either training or inference mode. |
| `ShouldMicroBatch(Tensor<>)` | G8 engagement decision: split this training step into `MicroBatchChunkSize`-row chunks and accumulate gradients, capping the peak activation memory. |
| `ShouldUseBFloat16Optimizer` | G2 proactive rung: whether to store optimizer moments as BF16 (half the fp32 footprint, minimal convergence impact). |
| `ShouldUseEightBitOptimizer` | G2 engagement decision: whether to use the 8-bit Adam optimizer (quantized moment state). |
| `ShouldUseStreamingTraining` | Autotuner: decides whether this Train step should take the memory-bounded streaming path. |
| `SliceAlongAxis0(Tensor<>,Int32,Int32)` | Axis-0 slice helper for `Int32)`: returns `input[start..end, …]` as a contiguous tensor. |
| `StepSchedulerIfSupported(IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Advances the optimizer's learning-rate scheduler at a training-batch boundary. |
| `TotalElementsMatch(Int32[],Int32[])` | Two shape arrays have the same total element count, with all dims known and positive. |
| `Train(Tensor<>,Tensor<>)` | Trains the neural network on a single input-output pair, OR on a batch when the caller passes a tensor whose leading dimension is the batch axis (shape `[B, …]`). |
| `TrainBatchGpuDeferred(Tensor<>,Tensor<>,IGpuOptimizerConfig,GpuExecutionOptions)` | Performs a complete training step (forward + backward + update) on GPU with deferred execution. |
| `TrainBatchGpuDeferredAsync(Tensor<>,Tensor<>,IGpuOptimizerConfig,GpuExecutionOptions,CancellationToken)` | Performs a complete training step (forward + backward + update) on GPU with deferred execution asynchronously. |
| `TrainBatched(Tensor<>[],Tensor<>[])` | Trains the network on a batch of input/target pairs in a single optimizer step. |
| `TrainWithCustomLoss(Tensor<>,Func<Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Performs tape-based training with a caller-provided loss function. |
| `TrainWithGradientAccumulation(Tensor<>,Tensor<>,Int32)` | Chunked training with TRUE gradient accumulation. |
| `TrainWithTape(Tensor<>,Tensor<>,Double)` | Overload for backward compatibility — accepts a learning rate instead of an optimizer. |
| `TrainWithTapeStreaming(Tensor<>,Tensor<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Boolean)` | Memory-bounded streaming training step: optimizer-in-backward with 8-bit quantized optimizer state and topological-min gradient release. |
| `TryAutoEnableWeightStreaming(Nullable<Boolean>)` | Auto-enables weight streaming if this model's total parameter count crosses the threshold AND the user hasn't already opted in or out explicitly. |
| `TryDeepCopyCopyOnWrite(IFullModel<,Tensor<>,Tensor<>>)` | G6 COW lever (#1624): builds the clone by sharing each layer's weight-tensor STORAGE with the original via the Tensors copy-on-write `CloneShared()` (O(1)-until-write), instead of the eager serialize-roundtrip / flatten copy. |
| `TryFoldBatchNormIntoConv(ConvolutionalLayer<>,BatchNormalizationLayer<>)` | Folds `bn`'s inference affine into `conv`'s kernels/biases in place. |
| `TryFoldBatchNormIntoDense(DenseLayer<>,BatchNormalizationLayer<>)` | Folds `bn`'s inference affine into `dense`'s weights/biases in place. |
| `TryForwardGpuOptimized(Tensor<>,Tensor<>)` | Attempts to perform a GPU-resident forward pass with automatic fallback to CPU. |
| `TryGetArchitectureInputShape` | Returns the input shape that `Layers`[0] actually observes — the starting point for `ResolveLazyLayerShapes`'s chain walk. |
| `TryMapToFusedOptimizerConfig(IGradientBasedOptimizer<,Tensor<>,Tensor<>>,OptimizerType,Single,Single,Single,Single,Single,LrSchedule)` | Inspects a pluggable optimizer and maps it onto the fixed set supported by the Tensors-side fused kernel (`SGD`, `Adam`, `AdamW`). |
| `UpdateParameters(Vector<>)` | Updates the network's parameters with new values. |
| `UpdateParametersGpu(,,)` | Updates all trainable parameters in the network using GPU-computed gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates all trainable parameters in the network using the specified optimizer configuration. |
| `UpdateParametersGpuDeferred(IGpuOptimizerConfig,GpuExecutionOptions)` | Updates all trainable parameters with deferred GPU execution. |
| `UploadWeightsToGpu` | Uploads all layer weights to GPU for GPU-resident training. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates that the provided layers form a valid neural network architecture. |
| `ValidateFairnessAsync(Tensor<>,Int32)` | Validates fairness metrics for the given inputs. |
| `ValidatePartitionPoint(Int32)` | Validates that a partition point between layers is valid for pipeline parallelism. |
| `WireLayerRandomSeeds` | Propagates `RandomSeed` to every layer (and nested sub-layer) so seed-respecting stochastic layers — chiefly `DropoutLayer`, whose mask derives from `RandomSeed` plus a per-forward counter — produce a reproducible stream. |
| `WithParameters(Vector<>)` | Creates a new neural network with the specified parameters. |
| `ZeroGradientsGpu` | Zeros all GPU gradient accumulators in preparation for a new batch. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Architecture` | The architecture definition for this neural network. |
| `DefaultStreamingThresholdParams` | Default parameter-count threshold above which weight streaming is auto-enabled. |
| `LastLoss` | The last calculated loss value during training. |
| `LossFunction` | The loss function used to calculate error during training. |
| `MaxGradNorm` | Backing storage for the gradient-clip max norm. |
| `MicroBatchChunkSize` | G8 micro-batch chunk size: the per-step batch processed at once when accumulating. |
| `NumOps` | Mathematical operations for the numeric type T. |
| `StreamingPrefetchWindow` | Streaming-aware forward path used when `IGpuOffloadAllocator)` has been called (whether explicitly or via the auto-detect threshold). |
| `_baseModel` | Base model instance for interpretability delegation. |
| `_baseTrainOptimizer` | Persistent optimizer for models using the standard TrainStep pattern. |
| `_baseTrainOptimizerExplicitlyConfigured` | True when the base-train optimizer was supplied explicitly via `Tensor{` (e.g. |
| `_cachedLayerInfo` | Cached layer info list, invalidated when layers change. |
| `_cachedParameterCount` | Cached parameter count to avoid repeated Sum() calculations. |
| `_compileHost` | Composable inference-compilation helper — traces the forward pass on first call at each input shape and replays the compiled plan on subsequent calls. |
| `_enabledMethods` | Set of interpretation methods that are enabled for this neural network model. |
| `_explicitlySetActiveFeatures` | Set of feature indices that have been explicitly marked as active. |
| `_fairnessMetrics` | List of fairness metrics to evaluate for this model. |
| `_firstForwardCompleted` | True after this network's first `Tensor{` or `Tensor{` call has completed. |
| `_fusedTrainingCommitted` | Tracks whether the fused compiled training path has EVER successfully run on this model. |
| `_fusedTrainingDisabled` | Sticky disable for the fused training path on this model instance. |
| `_inferenceAccelerationOptIn` | True when this model is foundation-scale and in inference mode, so `Tensor{` routes through the verify-then-trust compiled path + value memo instead of plain eager. |
| `_knownBadCompileShapes` | Tracks input shapes whose compilation has previously failed on this model instance. |
| `_layerInputs` | Stores the input values for each layer during forward pass. |
| `_layerOnlyInitialized` | One-shot flag for the layer-only branch of `EnsureArchitectureInitialized`. |
| `_layerOutputs` | Stores the output values from each layer during forward pass. |
| `_layerRandomSeedsWired` | Runs the forward pass through all layers WITHOUT suppressing tape recording. |
| `_layerShapesResolved` | Tracks whether `ResolveLazyLayerShapes` has already run once on this network instance. |
| `_layerStructureVersion` | Adds a layer to the internal layers collection and invalidates the parameter count cache. |
| `_layers` | The internal collection of layers that make up this neural network. |
| `_loggedFusedFallback` | One-shot guard for the loud fused-fallback warning emitted by `Tensor{`. |
| `_memoryLeversForced` | Set once a training step has OOM'd at full precision/full batch; latches the reactive G2 (8-bit optimizer) and G8 (micro-batch accumulation) memory levers ON for all subsequent steps of this model. |
| `_memoryManager` | Memory manager for gradient checkpointing and activation pooling. |
| `_mixedPrecisionContext` | Mixed-precision training context (null if mixed-precision is disabled). |
| `_parameterBuffer` | Contiguous parameter buffer for zero-copy flat parameter access. |
| `_pendingFusedMissReason` | Reason string set by `String)` when `Boolean)` bails out early, consumed (and cleared) by the post-success diagnostic block in `Tensor{`. |
| `_registrationLifetime` | Per-instance lifetime applied by `RegisterTrainableTensorsWithWeightRegistry` to every trainable tensor it walks. |
| `_sensitiveFeatures` | Indices of features considered sensitive for fairness analysis. |
| `_skipParameterBuffer` | Once foundation-scale models cross the parameter-buffer skip threshold we want each subsequent training step to take the no-buffer path in O(1), not re-scan every parameter tensor with CollectParameters + sum-Length on each call. |
| `_streamingAutoDetectDisabled` | True when the user explicitly opted out of auto-streaming for this instance via `DisableAutoStreaming` (e.g. |
| `_streamingAutoDetectFinalized` | True once auto-detect has FINALIZED on this instance. |
| `_streamingEngagedByAutoDetect` | Set to true ONLY when the auto-detect path itself called `IGpuOffloadAllocator)` (vs the user calling it explicitly via `IGpuOffloadAllocator)` or `ConfigureWeightStreaming(Enabled: true)`). |
| `_streamingThresholdOverride` | Per-instance threshold override applied by `Int64)` when the user passes `WeightStreamingConfig.ThresholdParameters`. |
| `_trainInFlight` | Reentrancy sentinel for `Tensor{`: 0 = idle, 1 = a training step is currently in flight on this instance. |
| `_trainMutationStarted` | True once the current `Tensor{` step has begun an in-place parameter or moment-buffer mutation (optimizer step / streaming apply / legacy update). |
| `_weightLifetimeConfigured` | True once `IGpuOffloadAllocator)` has been called on this network. |
| `s_autoCompiledInferenceEnabled` | Process-wide OPT-IN for the verify-then-trust compiled-inference path (#1622), OFF by default. |
| `s_streamingThresholdParams` | Resolved threshold: env-var override if set + parseable, else the compiled-in default. |

