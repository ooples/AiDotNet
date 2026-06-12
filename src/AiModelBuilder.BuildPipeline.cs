using AiDotNet.Augmentation;
using AiDotNet.AutoML.NAS;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;
using AiDotNet.AutoML.Policies;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet;

/// <summary>
/// Build-pipeline orchestration partial of <see cref="AiModelBuilder{T, TInput, TOutput}"/>:
/// the streaming and standard supervised build/optimize paths. Split out of the main file
/// (audit-2026-05 finding #12) to keep AiModelBuilder.cs reviewable; no behaviour change.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Performs true streaming supervised training without materializing all data in memory.
    /// </summary>
    /// <param name="streamingLoader">The streaming data loader to train from.</param>
    /// <returns>The result of training including the trained model.</returns>
    /// <remarks>
    /// <para>
    /// This method implements true streaming training by iterating through the streaming loader's
    /// batches and training on each batch individually. This allows training on datasets that
    /// are too large to fit in memory.
    /// </para>
    /// <para><b>For Beginners:</b> Unlike the regular training path which loads all data into memory,
    /// this method processes one batch at a time, trains on it, then moves to the next batch.
    /// This is essential for large datasets like ImageNet or large text corpora.
    /// </para>
    /// </remarks>
    private async Task<AiModelResult<T, TInput, TOutput>> BuildStreamingSupervisedAsync(
        IStreamingDataLoader<T, TInput, TOutput> streamingLoader)
    {
        // ConfigureAugmentation isn't wired into the streaming path —
        // BuildSupervisedInternalAsync's one-shot offline augmentation
        // applies to the materialised X tensor, which a streaming loader
        // doesn't produce. Fail loudly when the user configures EITHER
        // a custom augmenter OR a modality settings block (modality
        // factory dispatches the same offline apply path) — silently
        // dropping the augmentation here would reintroduce the stored-
        // but-not-consumed pattern this PR is trying to eliminate
        // (review #1368 C8eil: modality settings were unsupported on
        // streaming path but the gate only checked CustomAugmenter).
        if (_augmentationConfig is { IsEnabled: true } augCfg
            && (augCfg.CustomAugmenter is not null
                || augCfg.ImageSettings is not null
                || augCfg.TabularSettings is not null
                || augCfg.AudioSettings is not null
                || augCfg.TextSettings is not null
                || augCfg.VideoSettings is not null))
        {
            throw new NotSupportedException(
                "ConfigureAugmentation is not yet supported on the streaming data-loader path. " +
                "The augmentation hook is wired into BuildSupervisedInternalAsync's one-shot offline " +
                "augmentation against a materialised X tensor, which a streaming loader does not produce. " +
                "Either switch to an IInputOutputDataLoader (e.g. InMemoryDataLoader) or drop the " +
                "ConfigureAugmentation call (CustomAugmenter or any *Settings block) until streaming " +
                "augmentation is wired through the optimizer's per-batch hooks.");
        }

        // Apply GPU configuration first
        ApplyGpuConfiguration();

        // Apply memory management configuration (gradient checkpointing, etc.)
        ApplyMemoryConfiguration();

        // Ensure we have a model configured
        if (_model is null)
        {
            throw new InvalidOperationException(
                "Streaming training requires a model to be configured. Use ConfigureModel() before calling BuildAsync().");
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // Read epoch count from the optimizer's configured options.
        // Iterations / learning-rate / scheduler / weight-decay all live on
        // the optimizer itself — the facade does NOT shadow them. Previous
        // versions of this method maintained a separate `learningRate`
        // variable that decayed by 0.99 per epoch and was passed directly
        // to ApplyGradients(grad, lr). That created a duplicate decay
        // schedule running alongside whatever the optimizer's own scheduler
        // (Adam's bias correction, AdamW's decoupled weight decay, any
        // attached LearningRateScheduler) was doing — and bypassed the
        // optimizer's Step() entirely on the per-sample legacy path.
        // Now the facade only owns epoch-count and batch iteration; the
        // model's Train(batch) call dispatches through TrainWithTape →
        // Optimizer.Step(TapeStepContext), which is the supported path.
        var optimizerOptions = _optimizer?.GetOptions();
        int epochs = optimizerOptions?.MaxIterations ?? 100;

        // Get loss function
        var lossFunction = _model.DefaultLossFunction;

        // Training metrics
        T totalLoss = numOps.Zero;
        int totalBatches = 0;
        bool pipelineFitted = _preprocessingPipeline?.IsFitted ?? true;

        // Train for the specified number of epochs
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            T epochLoss = numOps.Zero;
            int epochBatches = 0;

            // Iterate through all batches in the streaming loader.
            //
            // BUG FIX (closes #1264 follow-up): the previous implementation
            // unrolled each batch into a per-sample for-loop and called
            // IGradientComputable.ComputeGradients/ApplyGradients on each
            // single sample. That path is the LEGACY pre-tape SGD update —
            // it bypassed the configured optimizer entirely (Adam's m/v
            // moments, AdamW's decoupled weight decay, fused tape kernels)
            // AND threw away the data loader's batching, so callers who
            // configured AdamOptimizer at LR=1e-3 and a batch size of 32
            // got vanilla SGD on a single sample at LR=1e-3 instead.
            //
            // The model's own Train(input, target) entry point dispatches
            // through TrainWithTape → Optimizer.Step(TapeStepContext), which
            // is the correct path. It also handles batched inputs natively
            // via NormalizeBatchDim — passing [B, …] tensors trains on the
            // batch in a single optimizer step. So we now stack each batch's
            // samples along a new leading batch dim and call _model.Train
            // ONCE per batch.
            await foreach (var (inputs, outputs) in streamingLoader.GetBatchesAsync(shuffle: true))
            {
                if (inputs.Length == 0) continue;

                // Fit preprocessing pipeline on the FIRST FULL BATCH if not already
                // fitted. The previous code fitted on `inputs[0]` (a single sample),
                // which makes mean/variance scalers compute mean = that one sample's
                // value and variance = 0 — turning the scaler into a no-op. Fitting
                // on the stacked batch reflects the actual feature distribution
                // across all samples seen in this batch.
                //
                // For TInput == Tensor<T>, stack into a [B, …] tensor before fitting.
                // For other TInput types we currently fall back to single-sample fit
                // since there's no generic batch-stack primitive on those types;
                // tracked as a follow-up.
                if (_preprocessingPipeline is not null && !pipelineFitted)
                {
                    if (inputs[0] is Tensor<T> && inputs.Length > 1
                        && TryStackTensorBatch(inputs.Cast<Tensor<T>>().ToArray(), out var batchedForFit)
                        && batchedForFit is TInput typedBatch)
                    {
                        _preprocessingPipeline.Fit(typedBatch);
                    }
                    else
                    {
                        // Fall back to fitting on a single sample when the
                        // batch isn't stackable (heterogeneous shapes — the
                        // loader chose not to override AggregateSamples to
                        // pad, which is fine but means we can't construct a
                        // [B, …] tensor for the scaler to compute statistics
                        // over). The single-sample fit is a degenerate case
                        // (mean = sample, var = 0) but matches the pre-#1264
                        // behavior and lets training proceed.
                        _preprocessingPipeline.Fit(inputs[0]);
                    }
                    pipelineFitted = true;
                }

                // TARGET scaling (ConfigureTargetScaling) on the streaming path: fit once on the first
                // batch's targets (mirroring the feature pipeline's first-batch fit above), then transform
                // every batch's targets so the model trains in scaled space. Predict inverse-transforms via
                // PreprocessingInfo.InverseTransformPredictions.
                TOutput[] processedOutputs = outputs;
                if (_targetPipeline is not null)
                {
                    if (!_targetPipeline.IsFitted && outputs.Length > 0)
                    {
                        _targetPipeline.Fit(outputs[0] is Tensor<T> && outputs.Length > 1
                            && TryStackTensorBatch(outputs.Cast<Tensor<T>>().ToArray(), out var batchedY)
                            && batchedY is TOutput typedY
                                ? typedY
                                : outputs[0]);
                    }

                    processedOutputs = new TOutput[outputs.Length];
                    for (int i = 0; i < outputs.Length; i++)
                    {
                        processedOutputs[i] = _targetPipeline.Transform(outputs[i]);
                    }
                }

                // Apply preprocessing to each sample BEFORE stacking so the
                // batched tensor reflects the transformed features. Same
                // pipeline-output handling as the previous code.
                TInput[] processedInputs;
                if (_preprocessingPipeline is not null && pipelineFitted)
                {
                    processedInputs = new TInput[inputs.Length];
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        var transformed = _preprocessingPipeline.Transform(inputs[i]);
                        processedInputs[i] = (transformed is TInput typed) ? typed : inputs[i];
                    }
                }
                else
                {
                    processedInputs = inputs;
                }

                // Neural network path: stack the batch into [B, …] tensors
                // and dispatch through _model.Train, which routes via
                // TrainWithTape → Optimizer.Step(TapeStepContext). The
                // optimizer owns all of: LR schedule, momentum / second-
                // moment state, bias correction, weight decay, gradient
                // clipping. The facade contributes batching + epoch loop
                // only.
                // Subclass-friendly check: `is Tensor<T>` admits SparseTensor<T>
                // and any future Tensor<T>-derived type, whereas the previous
                // `typeof(TInput) == typeof(Tensor<T>)` exact-equality check
                // bounced subclassed tensor types into the per-sample slow path
                // for no good reason.
                if (processedInputs.Length > 0 && processedInputs[0] is Tensor<T>
                    && processedOutputs.Length > 0 && processedOutputs[0] is Tensor<T>
                    && _model is INeuralNetwork<T> nn)
                {
                    // BUILDER-OPTIMIZER PLUMBING (closes review-comment #1265.f03A
                    // and #1265.hNaM):
                    // pre-wire the builder's configured optimizer onto the model
                    // so nn.Train resolves to it via GetOrCreateBaseOptimizer.
                    // Without this, nn.Train would lazy-construct a default Adam
                    // and silently drop AiModelBuilder.ConfigureOptimizer settings
                    // (custom AdamW, Lion, attached LR scheduler, hyperparameter
                    // overrides).
                    //
                    // Cast scope: `is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>`
                    // — only succeeds when the builder is parameterized as
                    // <T, Tensor<T>, Tensor<T>>, which is the canonical setup
                    // for NN training. For builders parameterized on other
                    // TInput/TOutput types but somehow still streaming Tensor<T>
                    // samples through to a NN model, the cast falls through and
                    // nn.Train uses the model's own default optimizer. That edge
                    // case is logically inconsistent (the configured optimizer
                    // operates in a different value-space than the model takes
                    // gradients in) and is not supported.
                    //
                    // Done at the top of the batch handler; re-setting on
                    // subsequent batches is a no-op (same instance).
                    if (_optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> gboTrain
                        && nn is NeuralNetworks.NeuralNetworkBase<T> nnBaseForOpt)
                    {
                        nnBaseForOpt.SetBaseTrainOptimizer(gboTrain);
                    }

                    // The loader's StreamingDataLoaderBase.AggregateSamples
                    // override is the documented extension point for padding
                    // / stacking variable-length sequences into a uniform-
                    // shape batch. When the loader uses AggregateSamples to
                    // produce uniform-shape tensors, TryStackTensorBatch
                    // succeeds and we get a single batched optimizer step.
                    // If a loader yields heterogeneous shapes (e.g. an
                    // image dataset without a Resize transform, or a
                    // sequence loader without a padding override), we
                    // gracefully fall back to per-sample nn.Train calls so
                    // training still progresses — matching the behavior of
                    // the per-sample loop that existed pre-#1264. This
                    // preserves correctness for default loaders while
                    // letting padded loaders enjoy the batched fast path.
                    var inputArray = processedInputs.Cast<Tensor<T>>().ToArray();
                    var outputArray = processedOutputs.Cast<Tensor<T>>().ToArray();
                    if (TryStackTensorBatch(inputArray, out var batchedInput) &&
                        TryStackTensorBatch(outputArray, out var batchedTarget))
                    {
                        nn.Train(batchedInput!, batchedTarget!);
                        var lastLoss = nn.GetLastLoss();
                        epochLoss = numOps.Add(epochLoss, lastLoss);
                        epochBatches++;
                    }
                    else
                    {
                        // Heterogeneous shapes — process each sample
                        // independently. Each call still goes through
                        // TrainWithTape → Optimizer.Step so the optimizer
                        // (Adam moments, AdamW weight decay, schedulers)
                        // is correctly engaged, just at a smaller per-
                        // call batch dim.
                        for (int i = 0; i < inputArray.Length; i++)
                        {
                            nn.Train(inputArray[i], outputArray[i]);
                            var sampleLoss = nn.GetLastLoss();
                            epochLoss = numOps.Add(epochLoss, sampleLoss);
                            epochBatches++;
                        }
                    }
                }
                else
                {
                    // Non-neural-network model that exposes IGradientComputable
                    // (e.g. logistic regression / classical online learners).
                    // These models have their own update-step semantics; we
                    // delegate via the standard ComputeGradients / ApplyGradients
                    // pair but DO NOT shadow the optimizer's learning-rate state.
                    // The model's own ApplyGradients picks up its configured
                    // learning rate from its options. Per-sample iteration is
                    // correct here because non-NN models don't have a "batched
                    // forward" concept; each sample is an independent update.
                    for (int i = 0; i < processedInputs.Length; i++)
                    {
                        var gradients = InterfaceGuard.GradientComputable(_model).ComputeGradients(processedInputs[i], processedOutputs[i], lossFunction);
                        // Pass the optimizer's CURRENT learning rate (which
                        // reflects scheduler / decay / step-counter state) instead
                        // of the constant InitialLearningRate. The previous code
                        // shadowed the scheduler — every step used the same
                        // initial LR regardless of how many iterations had run,
                        // so warmup and decay schedules were silently dropped on
                        // the non-NN path.
                        // _optimizer is the broader IOptimizer; only
                        // gradient-based optimizers expose GetCurrentLearningRate.
                        // Cast through the narrower interface so the schedule-
                        // aware LR is used when available; fall back to
                        // InitialLearningRate for non-gradient-based optimizers
                        // (which don't have a schedule concept anyway).
                        double currentLr;
                        if (_optimizer is IGradientBasedOptimizer<T, TInput, TOutput> gbo)
                            currentLr = gbo.GetCurrentLearningRate();
                        else if (_optimizer is not null)
                            currentLr = _optimizer.GetOptions().InitialLearningRate;
                        else
                            currentLr = 0.01;
                        var modelLr = numOps.FromDouble(currentLr);
                        InterfaceGuard.GradientComputable(_model).ApplyGradients(gradients, modelLr);

                        var prediction = _model.Predict(processedInputs[i]);
                        var predictionVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
                        var targetVector = ConversionsHelper.ConvertToVector<T, TOutput>(processedOutputs[i]);
                        var loss = lossFunction.CalculateLoss(predictionVector, targetVector);
                        epochLoss = numOps.Add(epochLoss, loss);
                        epochBatches++;
                    }
                }
            }

            totalLoss = numOps.Add(totalLoss, epochLoss);
            totalBatches += epochBatches;

            // No facade-level learning-rate decay. The optimizer owns its
            // own LR schedule (Adam's bias correction is handled in Step;
            // any attached LearningRateScheduler advances per-step inside
            // Optimizer.Step). The previous code maintained a parallel
            // learningRate variable that decayed 0.99 per epoch and
            // double-applied with whatever the optimizer was doing —
            // that's been removed.

            // Check for early stopping if configured
            if (_optimizer is not null && _optimizer.ShouldEarlyStop())
            {
                break;
            }
        }

        // Calculate average loss
        T avgLoss = totalBatches > 0
            ? numOps.Divide(totalLoss, numOps.FromDouble(totalBatches))
            : numOps.Zero;

        // Build the result
        var optimizationResult = new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = _model,
            BestFitnessScore = avgLoss,
            FitnessHistory = new Vector<T>(new[] { avgLoss }),
            Iterations = totalBatches
        };

        // Create deployment configuration from individual configs
        var deploymentConfig = DeploymentConfiguration.Create(
            _quantizationConfig,
            _cacheConfig,
            _versioningConfig,
            _abTestingConfig,
            _telemetryConfig,
            _exportConfig,
            _gpuAccelerationConfig,
            _compressionConfig,
            _profilingConfig);

        // Fit postprocessing pipeline through the shared helper. Streaming
        // path has no materialised X tensor, so passing trainingInput=null
        // causes the helper to throw the "no training data" diagnostic
        // when the user did configure postprocessing — failing fast with
        // a clear redirect to the supervised batch path (review #1368 C6WJG).
        FitPostprocessingIfNeeded(optimizationResult.BestSolution, default, nameof(BuildStreamingSupervisedAsync));

        // Build result using options pattern like other Build methods
        var options = new AiModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            PreprocessingInfo = (_preprocessingPipeline is not null && pipelineFitted) || _targetPipeline is not null
                ? new PreprocessingInfo<T, TInput, TOutput>
                {
                    Pipeline = _preprocessingPipeline is not null && pipelineFitted ? _preprocessingPipeline : null,
                    TargetPipeline = _targetPipeline,
                }
                : null,
            PostprocessingPipeline = _postprocessingPipeline,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            JitCompilationConfig = _jitCompilationConfig,
            JitCompiledFunction = BuildCompiledPredictFunction(optimizationResult.BestSolution),
            AllowNondeterminism = _allowNondeterminism,
            AugmentationConfig = _augmentationConfig,
            ReasoningConfig = _reasoningConfig,
            DeploymentConfiguration = deploymentConfig,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            InterpretabilityOptions = _interpretabilityOptions,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            LoRAConfiguration = _loraConfiguration,
            MemoryConfig = _memoryConfig,
            // Weight-streaming telemetry — set on every result-build path
            // (supervised batch, AutoML, RL) so the streaming-data-loader
            // path doesn't silently miss the report when the
            // user trained a streaming-eligible model via the streaming
            // loader. Closes review-comment #1271.s-Nc.
            WeightStreamingReport = BuildWeightStreamingReport()
        };

        var nnResult = AttachDiagnostics(new AiModelResult<T, TInput, TOutput>(options));
        ProcessKnowledgeGraphOptions(nnResult);
        AttachSafetyPipeline(nnResult);
        AttachAdversarialRobustness(nnResult);
        return nnResult;
    }

    /// <summary>
    /// Collects all data from a streaming data loader into aggregated features and labels.
    /// </summary>
    /// <param name="streamingLoader">The streaming data loader to collect from.</param>
    /// <returns>A tuple containing the aggregated features and labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reads all batches from a streaming data loader
    /// and combines them into single feature and label collections. This allows streaming
    /// loaders to work with the existing training infrastructure while preserving the
    /// benefit of on-demand data loading from files or other sources.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Attempts to stack an array of single-sample tensors along a new
    /// leading batch dimension. Returns false (with <paramref name="stacked"/>
    /// set to null) when shapes are heterogeneous so callers can fall back
    /// to per-sample handling rather than throwing.
    /// </summary>
    /// <remarks>
    /// Even for samples.Length == 1 this still adds the leading batch dim,
    /// producing a [1, *sampleShape] tensor rather than passing the
    /// unbatched sample through. Subclass training overrides like
    /// Transformer.Train don't call NormalizeBatchDim, so a final batch of
    /// size 1 would otherwise reach the layer pipeline at rank-N while
    /// every other batch of the same epoch reaches it at rank-(N+1) —
    /// a class of bug that bites only when the loader's last batch
    /// happens to be unevenly-sized (review-comment #1265.hc8I).
    /// </remarks>
    /// <summary>Gathers the given rows of a rank-1/rank-2 tensor into a new tensor (grouped training slices).</summary>
    private static Tensor<T> GatherRows(Tensor<T> source, IReadOnlyList<int> rows)
    {
        int cols = source.Rank >= 2 ? source.Shape[1] : 1;
        var data = new T[rows.Count * cols];
        var span = source.Data.Span;
        for (int r = 0; r < rows.Count; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                data[(r * cols) + c] = span[(rows[r] * cols) + c];
            }
        }

        var shape = source.Rank >= 2 ? new[] { rows.Count, cols } : new[] { rows.Count };
        return new Tensor<T>(shape, new AiDotNet.Tensors.LinearAlgebra.Vector<T>(data));
    }

    private static bool TryStackTensorBatch(Tensor<T>[] samples, out Tensor<T>? stacked)
    {
        stacked = null;
        if (samples is null || samples.Length == 0) return false;
        if (samples[0] is null) return false;

        var sampleShape = samples[0]._shape;
        int sampleStride = samples[0].Length;

        // Validate every sample matches the first sample's shape AND length
        // before allocating output storage. A heterogeneous batch is a
        // legitimate runtime case (the loader's AggregateSamples override
        // is the documented place to pad / resize), so signal it via the
        // bool return rather than throwing — the caller can then route
        // through the per-sample fallback path.
        for (int b = 1; b < samples.Length; b++)
        {
            if (samples[b] is null) return false;
            var bShape = samples[b]._shape;
            if (bShape.Length != sampleShape.Length) return false;
            for (int d = 0; d < sampleShape.Length; d++)
            {
                if (bShape[d] != sampleShape[d]) return false;
            }
            if (samples[b].Length != sampleStride) return false;
        }

        var batchedShape = new int[sampleShape.Length + 1];
        batchedShape[0] = samples.Length;
        for (int d = 0; d < sampleShape.Length; d++) batchedShape[d + 1] = sampleShape[d];

        var result = new Tensor<T>(batchedShape);
        for (int b = 0; b < samples.Length; b++)
        {
            int offset = b * sampleStride;
            for (int j = 0; j < sampleStride; j++) result[offset + j] = samples[b][j];
        }
        stacked = result;
        return true;
    }

    private async Task<(TInput Features, TOutput Labels)> CollectStreamingDataAsync(
        IStreamingDataLoader<T, TInput, TOutput> streamingLoader)
    {
        var allInputs = new List<TInput>();
        var allOutputs = new List<TOutput>();

        // Collect all batches asynchronously
        await foreach (var (inputs, outputs) in streamingLoader.GetBatchesAsync(shuffle: false))
        {
            foreach (var input in inputs)
            {
                allInputs.Add(input);
            }
            foreach (var output in outputs)
            {
                allOutputs.Add(output);
            }
        }

        if (allInputs.Count == 0)
        {
            throw new InvalidOperationException("Streaming data loader returned no data.");
        }

        // Aggregate the collected samples into single feature/label structures
        var aggregatedFeatures = AggregateStreamingInputs(allInputs);
        var aggregatedLabels = AggregateStreamingOutputs(allOutputs);

        return (aggregatedFeatures, aggregatedLabels);
    }

    /// <summary>
    /// Aggregates a list of input samples into a single TInput structure.
    /// </summary>
    private TInput AggregateStreamingInputs(List<TInput> inputs) =>
        DataAggregationHelper.Aggregate<T, TInput>(inputs, "input");

    /// <summary>
    /// Aggregates a list of output samples into a single TOutput structure.
    /// </summary>
    private TOutput AggregateStreamingOutputs(List<TOutput> outputs) =>
        DataAggregationHelper.Aggregate<T, TOutput>(outputs, "output");

    /// <summary>
    /// Internal method that performs supervised training with the provided input features and output values.
    /// This contains all the core supervised learning logic.
    /// </summary>
    /// <param name="x">Matrix of input features.</param>
    /// <param name="y">Vector of output values.</param>
    /// <returns>A task that represents the asynchronous operation, containing the trained model.</returns>
    private async Task<AiModelResult<T, TInput, TOutput>> BuildSupervisedInternalAsync(
        TInput x, TOutput y, CancellationToken cancellationToken)
    {
        // SUPERVISED TRAINING PATH

        // Create profiler session if profiling is enabled
        var profilerSession = CreateProfilerSession();
        using var _ = profilerSession?.Scope("BuildSupervisedInternalAsync");

        // Apply GPU configuration first (before any operations that might use GPU)
        ApplyGpuConfiguration();

        // ============================================================================
        // Training Infrastructure Initialization
        // ============================================================================

        // Variables to track training infrastructure state
        string? experimentRunId = null;
        string? experimentId = null;
        IExperimentRun<T>? experimentRun = null;
        string? monitorSessionId = null;
        string? checkpointPath = null;
        string? registeredModelName = null;
        int? modelVersion = null;
        string? dataVersionHash = null;
        var trainingStartTime = DateTime.UtcNow;

        // If a federated-aware data loader is configured, prefer its natural client partitions (e.g., LEAF users).
        // This keeps federated simulations faithful to benchmark/client boundaries and avoids leaking partitioning
        // concerns into the public facade API.
        List<(int ClientId, int StartRow, int SampleCount)>? federatedClientRanges = null;
        if (_federatedLearningOptions != null &&
            _dataLoader is IFederatedClientDataLoader<T, TInput, TOutput> federatedClientDataLoader)
        {
            var aggregated = BuildAggregatedDatasetFromClientData(federatedClientDataLoader.ClientData);
            x = aggregated.X;
            y = aggregated.Y;
            federatedClientRanges = aggregated.ClientRanges;
        }

        // Convert and validate inputs
        int xSamples = ConversionsHelper.GetSampleCount<T, TInput>(x);
        int ySamples = ConversionsHelper.GetSampleCount<T, TOutput>(y);

        if (xSamples != ySamples)
            throw new ArgumentException("Number of rows in features must match length of actual values", nameof(x));

        // Convert inputs to Matrix/Vector for internal processing
        var convertedX = ConversionsHelper.ConvertToMatrix<T, TInput>(x);
        var convertedY = ConversionsHelper.ConvertToVector<T, TOutput>(y);

        // AUTOML SEARCH (if configured and no model explicitly set)
        // AutoML finds the best model type and hyperparameters automatically
        AutoMLRunSummary? autoMLSummary = null;
        if (_autoMLModel != null && _model == null)
        {
            Console.WriteLine("AutoML configured - starting model search...");
            var searchStartedUtc = DateTimeOffset.UtcNow;

            // Step 1: Split data FIRST to prevent data leakage
            TInput autoMLPreparedX = x;
            TOutput autoMLPreparedY = y;

            bool shuffleBeforeSplit = !(_autoMLOptions?.TaskFamilyOverride == AutoMLTaskFamily.TimeSeriesForecasting
                || _autoMLOptions?.TaskFamilyOverride == AutoMLTaskFamily.TimeSeriesAnomalyDetection);

            var splitResult = DataSplitter.Split<T, TInput, TOutput>(
                autoMLPreparedX, autoMLPreparedY, trainRatio: 0.7, validationRatio: 0.15, shuffle: shuffleBeforeSplit);
            var autoMLXTrain = splitResult.XTrain;
            var autoMLYTrain = splitResult.yTrain;
            var autoMLXVal = splitResult.XVal;
            var autoMLYVal = splitResult.yVal;

            // Step 2: Apply data preparation (SMOTE, outlier removal) to training data ONLY after split.
            // Applying before split would leak test/validation information via synthetic samples.
            if (_dataPreparationPipeline != null && _dataPreparationPipeline.Count > 0)
            {
                if (autoMLXTrain is Matrix<T> xMatrix && autoMLYTrain is Vector<T> yVector)
                {
                    var (prepX, prepY) = _dataPreparationPipeline.FitResample(xMatrix, yVector);
                    autoMLXTrain = (TInput)(object)prepX;
                    autoMLYTrain = (TOutput)(object)prepY;
                }
                else if (autoMLXTrain is Tensor<T> xTensor && autoMLYTrain is Tensor<T> yTensor)
                {
                    var (prepX, prepY) = _dataPreparationPipeline.FitResampleTensor(xTensor, yTensor);
                    autoMLXTrain = (TInput)(object)prepX;
                    autoMLYTrain = (TOutput)(object)prepY;
                }
            }

            // Step 3: Apply preprocessing pipeline (scaling, encoding, etc.) to training only
            if (_preprocessingPipeline != null)
            {
                autoMLXTrain = _preprocessingPipeline.FitTransform(autoMLXTrain);
                autoMLXVal = _preprocessingPipeline.Transform(autoMLXVal);
            }

            if (_autoMLOptions?.TaskFamilyOverride is AutoMLTaskFamily taskFamilyOverride)
            {
                int featureCount = InputHelper<T, TInput>.GetInputSize(autoMLXTrain);
                var candidates = AutoMLDefaultCandidateModelsPolicy.GetDefaultCandidates(taskFamilyOverride, featureCount, _autoMLOptions.Budget.Preset);
                if (candidates.Count > 0)
                {
                    _autoMLModel.SetCandidateModels(candidates.ToList());
                }

                if (!_autoMLOptions.OptimizationMetricOverride.HasValue)
                {
                    var (metric, maximize) = AutoMLDefaultMetricPolicy.GetDefault(taskFamilyOverride);
                    _autoMLModel.SetOptimizationMetric(metric, maximize);
                }
            }

            // Wire the per-candidate hook so the user's WeightStreamingConfig
            // (force-on, force-off, or threshold override) applies to every
            // candidate the search instantiates — not just the post-search
            // winner. Without this, large neural candidates run with the
            // env-var/default threshold during the search itself, so a 562B
            // model can OOM during candidate evaluation even though the
            // caller configured force-on streaming. Subscribe BEFORE
            // SearchAsync, unsubscribe AFTER (search-scoped) so the hook
            // doesn't leak into a follow-up SearchAsync on the same
            // _autoMLModel instance with a different builder's config.
            void OnAutoMLCandidate(IFullModel<T, TInput, TOutput> candidate)
            {
                if (candidate is NeuralNetworks.NeuralNetworkBase<T> nnCandidate)
                {
                    ApplyWeightStreamingConfigTo(nnCandidate);
                }
            }
            _autoMLModel.OnCandidateCreated += OnAutoMLCandidate;

            IFullModel<T, TInput, TOutput> bestModel;
            try
            {
                // Run AutoML search to find the best model
                bestModel = await _autoMLModel.SearchAsync(
                    autoMLXTrain,
                    autoMLYTrain,
                    autoMLXVal,
                    autoMLYVal,
                    _autoMLModel.TimeLimit,
                    CancellationToken.None);
            }
            finally
            {
                _autoMLModel.OnCandidateCreated -= OnAutoMLCandidate;
            }

            _model = bestModel;
            // Re-apply checkpointing config now that AutoML has selected the
            // model — the BuildAsync entry-point applied it once against the
            // pre-AutoML _model (often null), so the AutoML-chosen model would
            // otherwise miss the user's UseGradientCheckpointing=true setting.
            ApplyGradientCheckpointingFromMemoryConfig();
            // Same for weight-streaming config: BuildAsync's earlier call ran
            // against the pre-AutoML _model. After AutoML picks bestModel,
            // the user's WeightStreamingConfig (Enabled override or per-instance
            // ThresholdParameters) needs to flow through to the new model
            // instance or auto-detect on the first forward will use the
            // env-var / default threshold instead of the configured one.
            // Closes review-comment #1271.s-NU.
            ApplyWeightStreamingConfig();

            var searchEndedUtc = DateTimeOffset.UtcNow;
            autoMLSummary = CreateAutoMLRunSummary(searchStartedUtc, searchEndedUtc);

            Console.WriteLine("AutoML search complete.");
            Console.WriteLine($"Best score: {_autoMLModel.BestScore}");
            Console.WriteLine($"Trials completed: {_autoMLModel.GetTrialHistory().Count}");
        }

        // Validate model is set (either by user, agent, or AutoML)
        if (_model == null)
            throw new InvalidOperationException("Model implementation must be specified. Use ConfigureModel() to set a model, ConfigureAutoML() for automatic model selection, or enable agent assistance.");

        // ============================================================================
        // SELF-SUPERVISED LEARNING PRETRAINING (#1361 #4) — runs BEFORE main training.
        // ConfigureSelfSupervisedLearning(configure, pretrainAction) is the wire-up
        // entry point — the SSL subsystem requires an encoder-shaped INeuralNetwork
        // that can't be transparently extracted from arbitrary IFullModel<T, TInput,
        // TOutput>. The user-supplied action is responsible for running the SSL
        // method (SimCLR / MoCo / BYOL / DINO / MAE / Barlow Twins) over its
        // pretraining batches and returning the model that should feed into main
        // supervised training (typically the same model with its encoder updated).
        // The single-argument overload (Action<SSLConfig>) stores configuration
        // without running any pretraining stage — that path is config-only.
        // ============================================================================
        if (_sslPretrainAction is not null)
        {
            if (_sslConfig is null)
                throw new InvalidOperationException(
                    "_sslPretrainAction was set without _sslConfig — internal builder invariant violated.");
            _model = await _sslPretrainAction(_model, _sslConfig, CancellationToken.None).ConfigureAwait(false);
            if (_model is null)
                throw new InvalidOperationException(
                    "ConfigureSelfSupervisedLearning's pretrainAction returned null. " +
                    "The hook must return a non-null IFullModel<T, TInput, TOutput> for main training to proceed.");
        }

        // Wire instance-level preprocessing/postprocessing onto DocumentNeuralNetworkBase models.
        // This replaces the former static PreprocessingRegistry/PostprocessingRegistry approach,
        // which caused race conditions when multiple models were built concurrently.
        ConfigureDocumentTransformers(_model);

        // Use defaults for the optimizer if not set. When the caller
        // ALSO configured regularization but did not pick an optimizer,
        // promote to AdamOptimizer (a GradientBasedOptimizerBase) so the
        // SetRegularization wiring below succeeds instead of throwing
        // — the regularization request is the strong signal that the user
        // expects a gradient-based optimizer (review #1368 C88Kn:
        // NormalOptimizer default + non-default regularization gave a
        // surprising Build-time throw with no clear fix because the user
        // never explicitly chose NormalOptimizer).
        IOptimizer<T, TInput, TOutput> optimizer;
        if (_optimizer is not null)
        {
            optimizer = _optimizer;
        }
        else if (_regularization is not null)
        {
            optimizer = new Optimizers.AdamOptimizer<T, TInput, TOutput>(_model);
        }
        else
        {
            optimizer = new NormalOptimizer<T, TInput, TOutput>(_model);
        }

        // Wire ConfigureRegularization through to the optimizer. Without
        // this, the user's regularization was stored on the builder
        // (_regularization) but the gradient-application loop inside
        // GradientBasedOptimizerBase read its own default L2 instead —
        // a stored-but-not-consumed bug discovered by AiDotNet#1345
        // Bucket7 ConfigureRegularization test. The setter on
        // GradientBasedOptimizerBase swaps the protected field at runtime
        // so optimizers constructed before ConfigureRegularization was
        // called still pick up the user's choice.
        if (_regularization is not null)
        {
            if (optimizer is Optimizers.GradientBasedOptimizerBase<T, TInput, TOutput> gradOptForReg)
            {
                gradOptForReg.SetRegularization(_regularization);
            }
            else
            {
                // Non-gradient optimizers (NormalOptimizer, evolutionary,
                // any custom IOptimizer outside the GradientBasedOptimizerBase
                // family) don't have a Regularization slot. Fail fast here
                // so the misconfiguration surfaces at Build time rather than
                // as silently-dropped regularization at training time
                // (review #1368).
                throw new InvalidOperationException(
                    "ConfigureRegularization is only supported on gradient-based optimizers " +
                    $"(AdamOptimizer / SGDOptimizer / AdamWOptimizer / etc.); the active optimizer " +
                    $"is {optimizer.GetType().Name} which has no Regularization slot. " +
                    "Either switch to a GradientBasedOptimizerBase subclass or remove the " +
                    "ConfigureRegularization call.");
            }
        }

        // LORA ADAPTATION (if configured)
        // Apply LoRA adapters to neural network layers for parameter-efficient fine-tuning
        if (_loraConfiguration != null && _model is NeuralNetworks.NeuralNetworkBase<T> neuralNetForLoRA)
        {
            System.Diagnostics.Trace.TraceInformation("Applying LoRA adapters to neural network layers...");

            // AiDotNet#1370 shape oracle: pre-loop asks every layer to declare its
            // shape from constructor args alone (TryDeclareShape). Layers like
            // MultiHeadAttentionLayer (knows embeddingDim from ctor) and any
            // layer constructed with explicit shape (e.g. LayerNormalizationLayer
            // with the featureSize ctor) return true without needing input.
            // Lazy convs / inferred-shape layers still return false and trigger
            // the existing warmup forward as a fallback.
            // PR #1388 review C7iL5: TryDeclareShape() is a public virtual
            // extension point — a custom layer override can throw arbitrary
            // exceptions. Treat non-fatal failures as "shape not declared"
            // (falls back to the warmup forward below), but let cancellation
            // and OOM propagate so the host can still abort. Trace the
            // failure with the layer type + full exception so the operator
            // can diagnose silently-skipped declarations.
            static bool TryDeclareShapeSafely(NeuralNetworks.Layers.LayerBase<T> layer)
            {
                try
                {
                    return layer.TryDeclareShape();
                }
                catch (Exception ex) when (
                    ex is not OperationCanceledException
                    && ex is not OutOfMemoryException
                    && ex is not StackOverflowException)
                {
                    System.Diagnostics.Trace.TraceWarning(
                        $"TryDeclareShape failed for {layer.GetType().FullName} — " +
                        $"treating as 'needs warmup': {ex}");
                    return false;
                }
            }

            // PR #1388 review C8mvN: only let LoRA-targeted layers drive the
            // warmup-skip decision. A non-target lazy layer (e.g. a lazy
            // ActivationLayer or DropoutLayer) won't be wrapped by ApplyLoRA
            // anyway — counting it as "needs warmup" forces the warmup
            // forward needlessly on mixed networks. Use the configuration's
            // own non-mutating eligibility predicate when available; for a
            // custom ILoRAConfiguration implementation that doesn't expose
            // one, fall back to "every LayerBase counts" (conservative —
            // may force an unnecessary warmup, but never skips one
            // incorrectly).
            var loraTargetProbe = _loraConfiguration as LoRA.DefaultLoRAConfiguration<T>;

            int declaredCount = 0;
            int needsWarmupCount = 0;
            for (int i = 0; i < neuralNetForLoRA.Layers.Count; i++)
            {
                var layer = neuralNetForLoRA.Layers[i];
                if (layer is not NeuralNetworks.Layers.LayerBase<T> declarable)
                {
                    // Non-LayerBase<T> layers (rare, e.g. wrapper adapters from a
                    // prior pass) bypass the oracle entirely — the ApplyLoRA call
                    // handles its own shape probing.
                    continue;
                }
                if (loraTargetProbe is not null && !loraTargetProbe.IsLoRATarget(declarable))
                {
                    // Not a LoRA target — its shape doesn't gate the warmup-skip
                    // decision. Skip without bumping either counter.
                    continue;
                }
                if (TryDeclareShapeSafely(declarable))
                    declaredCount++;
                else
                    needsWarmupCount++;
            }

            // If every shape-aware layer declared successfully, skip the warmup
            // forward entirely — this is the win that beats PyTorch / HuggingFace
            // PEFT's construction-time shape requirement: we get the zero-warmup
            // behavior when shapes are known, AND still support lazy layers via
            // the warmup fallback below when needed.
            bool skipWarmup = needsWarmupCount == 0;
            if (skipWarmup)
            {
                System.Diagnostics.Trace.TraceInformation(
                    $"LoRA warmup forward SKIPPED — all {declaredCount} shape-aware layer(s) " +
                    "declared shape from constructor args (AiDotNet#1370 shape oracle).");
            }
            else
            {
                System.Diagnostics.Trace.TraceInformation(
                    $"LoRA warmup forward required — {needsWarmupCount} layer(s) still need a forward " +
                    $"pass to resolve shape ({declaredCount} declared from ctor).");

                // Warmup forward to materialise lazy-init layers that didn't
                // self-declare. LoRAAdapterBase.CreateLoRALayer needs the
                // layer's input/output dimensions at adapter-construction
                // time; lazy layers that fall through TryDeclareShape report
                // (0, …) until first Forward materialises the shape.
                // Without the warmup, LoRALayer's ctor would throw
                // ArgumentOutOfRangeException("Output size must be positive").
                // Best-effort: if the warmup throws (e.g. the user wired a
                // forward path that requires training mode), the ApplyLoRA-side
                // IsShapeResolved guard silently skips still-unresolved layers
                // so the wrap loop succeeds on the materialised ones.
                // Discovered by AiDotNet#1345 Bucket10 ConfigureLoRA test.
                try
                {
                    bool prevTrainingMode = neuralNetForLoRA.IsTrainingMode;
                    neuralNetForLoRA.SetTrainingMode(false);
                    try
                    {
                        // One sample is enough to resolve lazy-layer shapes;
                        // a full-dataset forward would do O(N) work and
                        // allocate a full pass of activation tensors just to
                        // shape-resolve. Carve off a 1-row probe.
                        var warmupProbe = TrySliceFirstSampleForLoRAWarmup(x);
                        var warmupResult = _model.Predict(warmupProbe);
                        System.GC.KeepAlive(warmupResult);
                    }
                    finally
                    {
                        neuralNetForLoRA.SetTrainingMode(prevTrainingMode);
                    }
                }
                catch (OperationCanceledException)
                {
                    // Cancellation propagates — caller wants out, not a swallowed warmup.
                    throw;
                }
                catch (OutOfMemoryException)
                {
                    // Critical: don't mask. The host may need to abort.
                    // StackOverflowException is intentionally NOT listed —
                    // modern .NET terminates the process on SOE rather than
                    // letting it propagate, so a catch clause for it is
                    // unreachable (review #1368 C7mpq).
                    throw;
                }
                catch (Exception ex)
                {
                    // Best-effort warmup: documented forward-mode requirements
                    // (e.g. layers that need IsTrainingMode=true) can throw here.
                    // The ApplyLoRA-side IsShapeResolved guard silently skips
                    // still-unresolved layers so the wrap loop succeeds on
                    // materialized ones (review #1368 C6WOG: narrowed to let
                    // OperationCanceledException + OutOfMemoryException +
                    // StackOverflowException propagate; everything else is
                    // genuine warmup variance and stays as a Trace warning).
                    // Include ex.ToString() so the trace carries the full
                    // stack trace + inner exceptions, not just the top-frame
                    // message. Trace.TraceWarning is the only signal an
                    // operator has when the warmup fails silently (this PR's
                    // review C88M6: ex.Message dropped the origin frame and
                    // any chained inner exception, leaving a downstream
                    // skipped-lazy-layer mystery if the warmup actually
                    // failed inside an unrelated subsystem).
                    System.Diagnostics.Trace.TraceWarning(
                        $"LoRA warmup forward failed (proceeding — layers that materialised get wrapped; " +
                        $"lazy ones skipped via IsShapeResolved guard): {ex}");
                }
            }

            int adaptedCount = 0;
            int skippedLazyCount = 0;
            for (int i = 0; i < neuralNetForLoRA.Layers.Count; i++)
            {
                var originalLayer = neuralNetForLoRA.Layers[i];

                // AiDotNet#1370: gate on TryDeclareShape() rather than IsShapeResolved.
                // Layers like MHA that allocate weights from ctor-known dims return true
                // from TryDeclareShape even when InputShape still has a -1 seq placeholder
                // — LoRA wraps weight matrices, the seq placeholder doesn't matter.
                //
                // PR #1388 follow-up review C9PtZ: only probe TryDeclareShape on
                // layers that ApplyLoRA would actually wrap. A non-target lazy
                // layer (e.g. a lazy ActivationLayer or DropoutLayer) would get
                // its TryDeclareShape called, potentially allocating weights or
                // emitting a Trace warning, only for ApplyLoRA below to return
                // it unchanged. Gate on the same IsLoRATarget predicate the
                // pre-scan loop uses so the side effects of TryDeclareShape
                // only run for actual adaptation candidates.
                if (originalLayer is NeuralNetworks.Layers.LayerBase<T> lazyCheck
                    && (loraTargetProbe is null || loraTargetProbe.IsLoRATarget(lazyCheck))
                    && !TryDeclareShapeSafely(lazyCheck))
                {
                    skippedLazyCount++;
                    continue;
                }

                var adaptedLayer = _loraConfiguration.ApplyLoRA(originalLayer);

                // If the layer was adapted (wrapped with LoRA), update the list
                if (!ReferenceEquals(originalLayer, adaptedLayer))
                {
                    neuralNetForLoRA.Layers[i] = adaptedLayer;
                    adaptedCount++;
                }
            }

            if (skippedLazyCount > 0)
            {
                System.Diagnostics.Trace.TraceInformation($"LoRA skipped {skippedLazyCount} layer(s) whose shape was not resolved post-warmup.");
            }
            System.Diagnostics.Trace.TraceInformation($"LoRA applied to {adaptedCount} layers (rank={_loraConfiguration.Rank}, alpha={_loraConfiguration.Alpha})");
        }


        // Wrap model and optimizer for distributed training if configured
        IFullModel<T, TInput, TOutput> model = _model;
        IOptimizer<T, TInput, TOutput> finalOptimizer = optimizer;

        // Enable mixed-precision training BEFORE distributed training wrapping (if configured)
        // This ensures mixed-precision is applied to the base model/optimizer before any wrapping
        if (_mixedPrecisionConfig != null)
        {
            // Verify T is float
            if (typeof(T) != typeof(float))
            {
                throw new InvalidOperationException(
                    $"Mixed-precision training requires T = float, got T = {typeof(T).Name}. " +
                    $"Use AiModelBuilder<float, ...> to enable mixed-precision training.");
            }

            // Enable on neural network model if applicable
            if (_model is NeuralNetworkBase<T> neuralNet)
            {
                neuralNet.EnableMixedPrecision(_mixedPrecisionConfig);
            }

            // Enable on gradient-based optimizer if applicable
            if (optimizer is GradientBasedOptimizerBase<T, TInput, TOutput> gradOptimizer)
            {
                gradOptimizer.EnableMixedPrecision(_mixedPrecisionConfig);
            }
        }

        // Enable distributed training if backend or configuration was explicitly provided
        if (_distributedBackend != null || _distributedConfiguration != null)
        {
            // Use provided backend or default to InMemory for single-process
            var backend = _distributedBackend ?? new DistributedTraining.InMemoryCommunicationBackend<T>(rank: 0, worldSize: 1);

            // Use provided configuration or create default from backend
            var shardingConfig = _distributedConfiguration ?? new DistributedTraining.ShardingConfiguration<T>(backend);

            // Check if model/optimizer are already sharded to avoid double-wrapping
            bool isModelAlreadySharded = _model is DistributedTraining.IShardedModel<T, TInput, TOutput>;
            bool isOptimizerAlreadySharded = optimizer is DistributedTraining.IShardedOptimizer<T, TInput, TOutput>;

            // Only wrap if not already sharded
            if (isModelAlreadySharded || isOptimizerAlreadySharded)
            {
                // Model or optimizer already sharded - skip wrapping to avoid double-wrapping
                model = _model;
                finalOptimizer = optimizer;
            }
            else
            {
                // Switch on strategy to create appropriate model/optimizer pair
                (model, finalOptimizer) = _distributedStrategy switch
                {
                    DistributedStrategy.DDP => CreateDistributedPair(
                        new DistributedTraining.DDPModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.DDPOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.FSDP => CreateDistributedPair(
                        new DistributedTraining.FSDPModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.FSDPOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.ZeRO1 => CreateDistributedPair(
                        new DistributedTraining.ZeRO1Model<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.ZeRO1Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.ZeRO2 => CreateDistributedPair(
                        new DistributedTraining.ZeRO2Model<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.ZeRO2Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.ZeRO3 => CreateDistributedPair(
                        new DistributedTraining.ZeRO3Model<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.ZeRO3Optimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.PipelineParallel => CreateDistributedPair(
                        new DistributedTraining.PipelineParallelModel<T, TInput, TOutput>(
                            _model, shardingConfig,
                            microBatchCount: _pipelineMicroBatchCount,
                            partitionStrategy: _pipelinePartitionStrategy,
                            schedule: _pipelineSchedule,
                            checkpointConfig: _pipelineCheckpointConfig),
                        new DistributedTraining.PipelineParallelOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.TensorParallel => CreateDistributedPair(
                        new DistributedTraining.TensorParallelModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.TensorParallelOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    DistributedStrategy.Hybrid => CreateDistributedPair(
                        new DistributedTraining.HybridShardedModel<T, TInput, TOutput>(_model, shardingConfig),
                        new DistributedTraining.HybridShardedOptimizer<T, TInput, TOutput>(optimizer, shardingConfig)),
                    _ => throw new InvalidOperationException($"Unsupported distributed strategy: {_distributedStrategy}")
                };
            }
        }

        bool usePartitionedFederatedData = _federatedLearningOptions != null && federatedClientRanges != null;
        int expectedFederatedSampleCount = 0;
        if (usePartitionedFederatedData)
        {
            foreach (var range in federatedClientRanges!)
            {
                expectedFederatedSampleCount += range.SampleCount;
            }
        }

        // Preprocess the data in two stages:
        // 1. Data preparation (row-changing operations like outlier removal, augmentation)
        // 2. Preprocessing pipeline (transforms like scaling, encoding)
        TInput preparedX = x;
        TOutput preparedY = y;
        TInput preprocessedX;
        TOutput preprocessedY;
        PreprocessingInfo<T, TInput, TOutput>? preprocessingInfo = null;

        // Data preparation (outlier removal, augmentation via SMOTE, etc.) is applied AFTER
        // splitting to prevent data leakage. Applying FitResample before split would allow
        // synthetic samples derived from test/validation data to leak into the training set.
        // See the split paths below where FitResample is applied to training data only.

        // Step 1: Split and preprocess
        // CRITICAL: To prevent data leakage, the preprocessing pipeline must be fitted ONLY on
        // training data. Fitting on the full dataset (before splitting) leaks test/validation
        // statistics (mean, std dev, etc.) into the training pipeline, artificially inflating
        // metrics and producing overly-optimistic results.
        //
        // For federated learning with partitioned client data, ALL data is training data
        // (no split), so FitTransform on everything is correct.

        TInput XTrain;
        TOutput yTrain;
        // These variables are assigned in all code paths before use (train/val/test split or
        // federated path). The pragma suppresses nullable warnings for the initial default
        // declarations since TInput/TOutput are reference types (Matrix<T>/Vector<T>).
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type
        TInput XVal = default;
        TOutput yVal = default;
        TInput XTest = default;
        TOutput yTest = default;
#pragma warning restore CS8600

        if (usePartitionedFederatedData)
        {
            // Federated path: all data is training data — FitResample on everything is correct.
            if (_dataPreparationPipeline != null && _dataPreparationPipeline.Count > 0)
            {
                if (preparedX is Matrix<T> fedMatrix && preparedY is Vector<T> fedVector)
                {
                    var (prepX, prepY) = _dataPreparationPipeline.FitResample(fedMatrix, fedVector);
                    preparedX = (TInput)(object)prepX;
                    preparedY = (TOutput)(object)prepY;
                }
                else if (preparedX is Tensor<T> fedTensor && preparedY is Tensor<T> fedYTensor)
                {
                    var (prepX, prepY) = _dataPreparationPipeline.FitResampleTensor(fedTensor, fedYTensor);
                    preparedX = (TInput)(object)prepX;
                    preparedY = (TOutput)(object)prepY;
                }
            }

            // Federated path: all data is training data — FitTransform on everything is correct.
            if (_preprocessingPipeline is not null)
            {
                preprocessedX = _preprocessingPipeline.FitTransform(preparedX);
                preprocessedY = preparedY;

                preprocessingInfo = new PreprocessingInfo<T, TInput, TOutput>(
                    _preprocessingPipeline,
                    targetPipeline: _targetPipeline
                );
            }
            else
            {
                preprocessedX = preparedX;
                preprocessedY = preparedY;
            }

            var preprocessedMatrix = ConversionsHelper.ConvertToMatrix<T, TInput>(preprocessedX);
            var preprocessedVector = ConversionsHelper.ConvertToVector<T, TOutput>(preprocessedY);

            if (preprocessedMatrix.Rows != preprocessedVector.Length)
            {
                throw new InvalidOperationException(
                    "Federated learning with partitioned client data requires preprocessing to preserve X/y row alignment. " +
                    $"Got X rows={preprocessedMatrix.Rows} and y length={preprocessedVector.Length}.");
            }

            if (preprocessedMatrix.Rows != expectedFederatedSampleCount)
            {
                throw new InvalidOperationException(
                    "Federated learning with partitioned client data requires preprocessing to preserve the total number of samples and row ordering. " +
                    $"Expected {expectedFederatedSampleCount} samples from the data loader partitions but preprocessing produced {preprocessedMatrix.Rows}. " +
                    "If you are using outlier removal, filtering, or other preprocessing that drops/reorders rows, disable it for partitioned federated learning or " +
                    "apply preprocessing at the per-client level before aggregating client datasets.");
            }

            // For natural per-client datasets (e.g., LEAF), avoid re-splitting at the sample level so that we can
            // preserve the client boundaries through preprocessing.
            XTrain = preprocessedX;
            yTrain = preprocessedY;
        }
        else
        {
            // Standard supervised learning path: split FIRST, then fit preprocessing on training only.
            // This prevents data leakage from test/validation sets into the preprocessing pipeline.
            // Disable shuffling for time-series tasks to preserve chronological ordering.
            // Disable shuffling for time-series models to preserve chronological ordering.
            // Random shuffling destroys the sequential dependencies that TS models rely on.
            bool isTimeSeriesModel = _model is TimeSeries.TimeSeriesModelBase<T>
                || _autoMLOptions?.TaskFamilyOverride == AutoMLTaskFamily.TimeSeriesForecasting
                || _autoMLOptions?.TaskFamilyOverride == AutoMLTaskFamily.TimeSeriesAnomalyDetection;
            bool shuffleBeforeSplit = !isTimeSeriesModel;
            (XTrain, yTrain, XVal, yVal, XTest, yTest) = DataSplitter.Split<T, TInput, TOutput>(
                preparedX, preparedY, trainRatio: 0.7, validationRatio: 0.15, shuffle: shuffleBeforeSplit);

            // Apply data preparation (SMOTE, outlier removal, etc.) to training data ONLY after split.
            // Applying before split would leak test/validation information via synthetic samples.
            if (_dataPreparationPipeline != null && _dataPreparationPipeline.Count > 0)
            {
                if (XTrain is Matrix<T> trainMatrix && yTrain is Vector<T> trainVector)
                {
                    var (prepX, prepY) = _dataPreparationPipeline.FitResample(trainMatrix, trainVector);
                    XTrain = (TInput)(object)prepX;
                    yTrain = (TOutput)(object)prepY;
                }
                else if (XTrain is Tensor<T> trainTensor && yTrain is Tensor<T> trainYTensor)
                {
                    var (prepX, prepY) = _dataPreparationPipeline.FitResampleTensor(trainTensor, trainYTensor);
                    XTrain = (TInput)(object)prepX;
                    yTrain = (TOutput)(object)prepY;
                }
            }

            // TARGET scaling (ConfigureTargetScaling): fit on TRAINING targets only, transform val/test
            // into the same scaled space so in-build metrics compare like-with-like. Predict inverse-
            // transforms outputs back to original units via PreprocessingInfo.InverseTransformPredictions.
            if (_targetPipeline is not null)
            {
                yTrain = _targetPipeline.FitTransform(yTrain);
#pragma warning disable CS8604 // yVal / yTest are assigned by DataSplitter.Split above
                yVal = _targetPipeline.Transform(yVal);
                yTest = _targetPipeline.Transform(yTest);
#pragma warning restore CS8604
            }

            if (_preprocessingPipeline is not null)
            {
                // FitTransform on training data only — learns statistics from training set
                XTrain = _preprocessingPipeline.FitTransform(XTrain);

                // Transform (NOT FitTransform) validation and test data using training-fitted pipeline
#pragma warning disable CS8604 // XVal and XTest are assigned by DataSplitter.Split above
                XVal = _preprocessingPipeline.Transform(XVal);
                XTest = _preprocessingPipeline.Transform(XTest);
#pragma warning restore CS8604

                preprocessingInfo = new PreprocessingInfo<T, TInput, TOutput>(
                    _preprocessingPipeline,
                    targetPipeline: _targetPipeline
                );

                preprocessedX = XTrain; // For downstream references
                preprocessedY = yTrain;
            }
            else
            {
                // No preprocessing pipeline configured - pass through, but keep any training-only data preparation
                preprocessedX = XTrain;
                preprocessedY = yTrain;
                if (_targetPipeline is not null)
                {
                    // Target-only scaling still needs a carrier so Predict can inverse-transform.
                    preprocessingInfo = new PreprocessingInfo<T, TInput, TOutput> { TargetPipeline = _targetPipeline };
                }
            }
        }

        // Resolve the effective augmenter: explicit CustomAugmenter wins,
        // otherwise fall back to the modality factory which translates the
        // ImageSettings / TabularSettings / AudioSettings / TextSettings /
        // VideoSettings blocks into a pipeline of built-in augmenters
        // (review #1368 C6WKu). The factory returns null when no modality
        // settings are populated, leaving augmentation disabled — that
        // preserves the prior no-op behavior for callers who only set
        // IsEnabled without populating a settings block.
        object? effectiveAugmenter = null;
        if (_augmentationConfig is { IsEnabled: true } augCfg)
        {
            effectiveAugmenter = augCfg.CustomAugmenter ?? ResolveModalityAugmenter(augCfg);
        }
        // Apply ConfigureAugmentation if an augmenter (explicit or
        // modality-derived) resolved. Applied once before the optimizer
        // runs (offline data augmentation); per-batch / per-epoch (online)
        // augmentation would require deeper hooks into the optimizer's
        // batch loop. Discovered by AiDotNet#1345 Bucket8 ConfigureAugmentation
        // test.
        if (effectiveAugmenter is not null && _augmentationConfig is not null)
        {
            object customAug = effectiveAugmenter;
            // Split the cast check + the TInput cast into two separate
            // branches so the diagnostic can distinguish "augmenter type
            // mismatch" from "preprocessed input is not the expected
            // TInput at this point" (review #1368 C4TP1: a correctly-
            // typed augmenter with a TInput-changing preprocessor was
            // misleadingly blamed on the augmenter under the prior
            // combined-conditional).
            if (customAug is not AiDotNet.Augmentation.IAugmentation<T, TInput> typedAug)
            {
                throw new InvalidOperationException(
                    $"ConfigureAugmentation: CustomAugmenter of type {customAug.GetType().Name} is not " +
                    $"IAugmentation<{typeof(T).Name}, {typeof(TInput).Name}>. " +
                    "The augmenter's TData generic argument must match the AiModelBuilder's TInput. " +
                    "Use AugmentationConfig.SetCustomAugmenter<TNum, TData>(...) or the strongly-typed " +
                    "AugmentationConfig<T, TInput>.Augmenter property for a compile-time-checked " +
                    "setter that catches this at the call site.");
            }
            // `preprocessedX` is statically `TInput` (declared at L2799), so the
            // earlier `preprocessedX is not TInput` pattern-match was effectively
            // an always-false branch for non-null values — the reviewer (review
            // #1368 C6WKa) correctly flagged it as dead. Keep an explicit null
            // guard since TInput may be a reference type and a buggy upstream
            // preprocessing pipeline could legitimately produce null (caught
            // here at Build time instead of NRE-ing inside the augmenter).
            if (preprocessedX is null)
            {
                throw new InvalidOperationException(
                    "ConfigureAugmentation: the preprocessing pipeline output is null. " +
                    "ConfigurePreprocessing transformers must not return null for non-empty input. " +
                    "Check the configured pipeline's FitTransform implementation.");
            }

            var augContext = new AiDotNet.Augmentation.AugmentationContext<T>(
                isTraining: true,
                seed: _augmentationConfig.Seed);
            // typedAug is IAugmentation<T, TInput>, so Apply returns
            // TInput directly — no runtime cast needed (review #1368 C88R8:
            // prior `augmented is TInput` was trivially true for reference
            // types and a compile-time-known true for value types).
            TInput augmented = typedAug.Apply(preprocessedX, augContext);
            // Update the train-side X with the augmented data so the
            // optimizer sees the transformed inputs.
            preprocessedX = augmented;
            XTrain = augmented;
            // Emit the two ConfigureAugmentation constraint warnings
            // ONCE per process (not per Build) so multi-Build / CI
            // pipelines that exercise ConfigureAugmentation many times
            // don't get the same two lines in every trace (review
            // #1368 C4TPM). The flags are static so they're shared
            // across all AiModelBuilder<T, TInput, TOutput> instances —
            // the contract is process-wide informational.
            if (System.Threading.Interlocked.Exchange(ref AugmentationWarningLatch.OfflineEmitted, 1) == 0)
            {
                System.Diagnostics.Trace.TraceInformation(
                    "ConfigureAugmentation: applied a single offline pass to the training set before the optimizer runs. " +
                    "Per-epoch / per-batch stochastic augmentation is not yet wired into the optimizer batch loop; " +
                    "non-deterministic augmenters (random crop, noise, masking) will produce one fixed augmented " +
                    "copy reused every epoch. (This message logs once per process.)");
            }
            if (System.Threading.Interlocked.Exchange(ref AugmentationWarningLatch.XOnlyEmitted, 1) == 0)
            {
                System.Diagnostics.Trace.TraceInformation(
                    "ConfigureAugmentation: only transforms training X, not labels y. The configured augmenter " +
                    "must be 1:1 row-preserving on inputs (no row reorder / drop / N->M expansion). " +
                    "Non-1:1 augmenters will silently desynchronise X and y. (This message logs once per process.)");
            }
        }

        // Cross-validation can be performed using the new evaluation framework via AiModelResult.
        // Users can call CrossValidationEngine<T> directly for cross-validation needs.
        CrossValidationResult<T, TInput, TOutput>? cvResults = null;

        // ============================================================================
        // Causal Discovery (if configured)
        // ============================================================================
        CausalDiscovery.CausalDiscoveryResult<T>? causalDiscoveryResult = null;
        if (_causalDiscoveryOptions != null)
        {
            try
            {
                var cdStopwatch = System.Diagnostics.Stopwatch.StartNew();

                // Select algorithm: user-specified or auto-select based on data characteristics
                var algorithmType = _causalDiscoveryOptions.Algorithm
                    ?? AutoSelectCausalAlgorithm(convertedX.Columns);

                var algorithm = CausalDiscovery.CausalDiscoveryAlgorithmFactory<T>.Create(
                    algorithmType, _causalDiscoveryOptions);

                // Use feature names from options, or generate defaults
                var featureNames = _causalDiscoveryOptions.FeatureNames;
                if ((featureNames == null || featureNames.Length != convertedX.Columns) && convertedX.Columns > 0)
                {
                    featureNames = new string[convertedX.Columns];
                    for (int fi = 0; fi < convertedX.Columns; fi++)
                        featureNames[fi] = $"X{fi}";
                }

                // Run causal discovery with target variable for supervised context
                var causalGraph = algorithm.DiscoverStructure(convertedX, convertedY, featureNames);

                cdStopwatch.Stop();

                var category = CausalDiscovery.CausalDiscoveryAlgorithmFactory<T>.GetCategory(algorithmType);

                causalDiscoveryResult = new CausalDiscovery.CausalDiscoveryResult<T>(
                    graph: causalGraph,
                    algorithmUsed: algorithmType,
                    category: category,
                    elapsedTime: cdStopwatch.Elapsed);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"Causal discovery failed: {ex.Message}");
            }
        }

        // ============================================================================
        // Start Training Infrastructure (before optimization)
        // ============================================================================

        // Track data version for reproducibility
        if (_dataVersionControl is not null)
        {
            // Compute a hash of the training data for lineage tracking
            // This enables reproducibility by recording exactly which data was used
            var dataVersionNumOps = MathHelper.GetNumericOperations<T>();
            dataVersionHash = ComputeDataVersionHash(convertedX, convertedY, dataVersionNumOps);

            // Note: For in-memory data, we cannot use CreateDatasetVersion (requires dataPath).
            // Instead, we track the data characteristics via the experiment run parameters below.
            // The data_version_hash is logged to experiment run parameters (line ~1122) and linked
            // via _dataVersionControl.LinkDatasetToRun (line ~1131) for full traceability.
            // Key metadata tracked: rows, columns, target_length, feature_count, training/validation/test samples.
        }

        // Start experiment tracking run
        if (_experimentTracker is not null)
        {
            experimentId = _experimentTracker.CreateExperiment(
                name: "supervised-training",
                description: $"Supervised learning with {model.GetType().Name}",
                tags: new Dictionary<string, string>
                {
                    ["model_type"] = model.GetType().Name,
                    ["optimizer_type"] = finalOptimizer.GetType().Name,
                    ["framework"] = "AiDotNet"
                });

            experimentRun = _experimentTracker.StartRun(
                experimentId: experimentId,
                runName: $"run-{trainingStartTime:yyyyMMdd-HHmmss}",
                tags: new Dictionary<string, string>
                {
                    ["start_time"] = trainingStartTime.ToString("O")
                });

            experimentRunId = experimentRun.RunId;

            // Log hyperparameters from optimizer options
            var optimizerOptions = finalOptimizer.GetOptions();
            experimentRun.LogParameters(new Dictionary<string, object>
            {
                ["model_type"] = model.GetType().FullName ?? model.GetType().Name,
                ["optimizer_type"] = finalOptimizer.GetType().FullName ?? finalOptimizer.GetType().Name,
                ["max_iterations"] = optimizerOptions.MaxIterations,
                ["use_early_stopping"] = optimizerOptions.UseEarlyStopping,
                ["early_stopping_patience"] = optimizerOptions.EarlyStoppingPatience,
                ["training_samples"] = XTrain is Matrix<T> trainMatrix ? trainMatrix.Rows : 0,
                ["validation_samples"] = XVal is Matrix<T> valMatrix ? valMatrix.Rows : 0,
                ["test_samples"] = XTest is Matrix<T> testMatrix ? testMatrix.Rows : 0,
                ["feature_count"] = convertedX.Columns,
                ["target_length"] = convertedY.Length,
                ["data_version_hash"] = dataVersionHash ?? "not_tracked"
            });

            // Link data version to experiment run for full traceability
            if (_dataVersionControl is not null && dataVersionHash is not null)
            {
                try
                {
                    var datasetName = $"training-data-{model.GetType().Name}";
                    _dataVersionControl.LinkDatasetToRun(
                        datasetName: datasetName,
                        versionHash: dataVersionHash,
                        runId: experimentRunId,
                        modelId: null); // Model ID will be set after registry
                }
                catch (Exception)
                {
                    // Data version control linkage is optional - don't fail training
                }
            }
        }

        // Start training monitor session
        if (_trainingMonitor is not null)
        {
            monitorSessionId = _trainingMonitor.StartSession(
                sessionName: $"training-{model.GetType().Name}",
                metadata: new Dictionary<string, object>
                {
                    ["model_type"] = model.GetType().Name,
                    ["optimizer_type"] = finalOptimizer.GetType().Name,
                    ["experiment_run_id"] = experimentRunId ?? string.Empty,
                    ["start_time"] = trainingStartTime
                });
        }

        // ============================================================================
        // Hyperparameter Optimization (if configured)
        // ============================================================================

        HyperparameterOptimizationResult<T>? hyperparameterOptimizationResult = null;
        int? bestHyperparameterTrialId = null;
        Dictionary<string, object>? bestHyperparameters = null;

        if (_hyperparameterOptimizer is not null && _hyperparameterSearchSpace is not null)
        {
            var numOps = MathHelper.GetNumericOperations<T>();

            try
            {
                // Log hyperparameter optimization start
                if (_trainingMonitor is not null && monitorSessionId is not null)
                {
                    _trainingMonitor.LogMessage(monitorSessionId, LogLevel.Info, "Starting hyperparameter optimization...");
                }

                // Create objective function that trains the model and returns validation loss
                T ObjectiveFunction(Dictionary<string, object> trialHyperparameters)
                {
                    // Apply trial hyperparameters first so Reset() initializes adaptive state from them
                    var optimizerOptions = finalOptimizer.GetOptions();
                    ApplyTrialHyperparameters(optimizerOptions, trialHyperparameters);

                    // Reset optimizer state to reinitialize adaptive parameters from new hyperparameters
                    finalOptimizer.Reset();

                    // Ensure the optimizer has a model set (required if optimizer was constructed without one)
                    finalOptimizer.SetModel(model);

                    // Log hyperparameters for this trial to experiment tracker
                    if (experimentRun is not null)
                    {
                        foreach (var kvp in trialHyperparameters)
                        {
                            experimentRun.LogParameter($"hpo_trial_{kvp.Key}", kvp.Value);
                        }
                    }

                    // Train with current hyperparameters
#pragma warning disable CS8604 // XVal/yVal/XTest/yTest assigned by DataSplitter.Split
                    var trialResult = finalOptimizer.Optimize(
                        OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
                            XTrain, yTrain, XVal, yVal, XTest, yTest));
#pragma warning restore CS8604

                    // Return validation MSE as objective (minimizing)
                    if (trialResult.ValidationResult.ErrorStats is not null)
                    {
                        return trialResult.ValidationResult.ErrorStats.MSE;
                    }

                    // Fallback to training loss if validation unavailable
                    if (trialResult.TrainingResult.ErrorStats is not null)
                    {
                        return trialResult.TrainingResult.ErrorStats.MSE;
                    }

                    // Return maximum value for failed trials to penalize them heavily
                    // when minimizing (zero would incorrectly indicate perfect performance)
                    return numOps.MaxValue;
                }

                // Run hyperparameter optimization
                hyperparameterOptimizationResult = _hyperparameterOptimizer.Optimize(
                    ObjectiveFunction,
                    _hyperparameterSearchSpace,
                    _hyperparameterTrials);

                // Extract best trial information
                if (hyperparameterOptimizationResult.BestTrial is not null)
                {
                    bestHyperparameterTrialId = hyperparameterOptimizationResult.BestTrial.TrialNumber;
                    bestHyperparameters = hyperparameterOptimizationResult.BestParameters;

                    // Log best hyperparameters to experiment tracker
                    if (experimentRun is not null && bestHyperparameters is not null)
                    {
                        foreach (var kvp in bestHyperparameters)
                        {
                            experimentRun.LogParameter($"best_{kvp.Key}", kvp.Value);
                        }

                        var bestValue = hyperparameterOptimizationResult.BestTrial.ObjectiveValue ?? numOps.Zero;
                        experimentRun.LogMetric("best_trial_objective", bestValue);
                    }

                    if (_trainingMonitor is not null && monitorSessionId is not null)
                    {
                        _trainingMonitor.LogMessage(monitorSessionId, LogLevel.Info,
                            $"HPO complete: best trial={bestHyperparameterTrialId}, completed={hyperparameterOptimizationResult.CompletedTrials}");
                    }
                }

                // Reset optimizer for final training
                finalOptimizer.Reset();
            }
            catch (Exception ex)
            {
                // Hyperparameter optimization is optional - log warning and continue
                if (_trainingMonitor is not null && monitorSessionId is not null)
                {
                    _trainingMonitor.LogMessage(monitorSessionId, LogLevel.Warning, $"HPO failed: {ex.Message}");
                }
            }
        }

        // Uncertainty quantification: create deep ensemble template before optimization
        var deepEnsembleTemplate = _uncertaintyQuantificationOptions is { Enabled: true, Method: UncertaintyQuantificationMethod.DeepEnsemble }
            ? _model.DeepCopy()
            : null;

        OptimizationResult<T, TInput, TOutput> optimizationResult;
        FederatedLearningMetadata? federatedLearningMetadata = null;
#pragma warning disable CS8604 // XVal/yVal/XTest/yTest assigned by DataSplitter.Split
        var optimizationInputData = OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest);
#pragma warning restore CS8604

        // FEDERATED LEARNING PATH (facade-first: orchestration stays internal)
        if (_federatedLearningOptions != null)
        {
            if (_knowledgeDistillationOptions != null)
            {
                throw new InvalidOperationException("Federated learning cannot be combined with knowledge distillation in the same Build() call.");
            }

            if (_distributedBackend != null || _distributedConfiguration != null)
            {
                throw new InvalidOperationException("Federated learning is not currently compatible with distributed training configuration. Use either federated learning or distributed training per build.");
            }

            var flOptions = _federatedLearningOptions;

            Dictionary<int, FederatedClientDataset<TInput, TOutput>> clientPartitions;
            int effectiveClientCount;

            if (usePartitionedFederatedData)
            {
                clientPartitions = CreateFederatedClientPartitionsFromClientRanges(
                    XTrain,
                    yTrain,
                    federatedClientRanges!);

                effectiveClientCount = clientPartitions.Count;

                if (effectiveClientCount <= 0)
                {
                    throw new InvalidOperationException("Federated client data resolved from the data loader is empty.");
                }

                if (flOptions.NumberOfClients != effectiveClientCount)
                {
                    Console.WriteLine(
                        $"[AiDotNet] Warning: FederatedLearningOptions.NumberOfClients={flOptions.NumberOfClients} does not match the data loader client count {effectiveClientCount}. Using the data loader client count.");
                }
            }
            else
            {
                clientPartitions = CreateFederatedClientPartitions(XTrain, yTrain, flOptions.NumberOfClients, flOptions.RandomSeed);
                effectiveClientCount = clientPartitions.Count;
            }

            var trainer = new AiDotNet.FederatedLearning.Trainers.InMemoryFederatedTrainer<T, TInput, TOutput>(
                optimizerPrototype: finalOptimizer,
                learningRateOverride: flOptions.LearningRate,
                randomSeed: flOptions.RandomSeed,
                convergenceThreshold: flOptions.ConvergenceThreshold,
                minRoundsBeforeConvergence: flOptions.MinRoundsBeforeConvergence,
                federatedLearningOptions: flOptions,
                clientSelectionStrategy: _federatedClientSelectionStrategy,
                serverOptimizer: _federatedServerOptimizer,
                heterogeneityCorrection: _federatedHeterogeneityCorrection,
                homomorphicEncryptionProvider: _federatedHomomorphicEncryptionProvider);

            var aggregationStrategy = _federatedAggregationStrategy ?? CreateDefaultFederatedAggregationStrategy(flOptions);
            trainer.SetAggregationStrategy(aggregationStrategy);
            trainer.Initialize(model, effectiveClientCount);

            federatedLearningMetadata = trainer.Train(
                clientData: clientPartitions,
                rounds: flOptions.MaxRounds,
                clientSelectionFraction: flOptions.ClientSelectionFraction,
                localEpochs: flOptions.LocalEpochs);

            optimizationResult = new OptimizationResult<T, TInput, TOutput>
            {
                BestSolution = trainer.GetGlobalModel(),
                Iterations = federatedLearningMetadata.RoundsCompleted
            };
        }
        else if (UseDirectTrainingPath(model))
        {
            // Branch rationale documented on UseDirectTrainingPath
            // (non-parametric models, ClusteringBase, NN + LoRA).
            if (_knowledgeDistillationOptions is not null)
            {
                throw new NotSupportedException(
                    "Knowledge distillation is not supported for non-parametric models. " +
                    "Remove the ConfigureKnowledgeDistillation() call.");
            }

            // DIRECT TRAINING PATH for non-parametric models (TS, density-based clustering, etc.)
            // and partitioning clustering models (KMeans, BIRCH, GMM, etc.) — clustering's
            // ClusteringBase reports SupportsParameterInitialization=true (it has cluster
            // centers / labels / membership matrices that ARE parameters in the
            // IParameterizable sense), but its train path is the K-means EM loop, not the
            // outer-optimizer's clone-evaluate-select. Routing it through the regular
            // optimizer path made BuildAsync run hundreds of iterations of unrelated random
            // search, never actually call ClusteringBase.Train, take 150+ seconds on 90
            // samples, then return a fresh-init untrained model whose Predict() throws
            // "Model must be trained" — exactly the timeout pattern the 25 clustering
            // Builder_* tests in #1224 Cluster B were hitting. Probe _model (unwrapped)
            // not the local model variable, mirroring the useFullData check below — the
            // local may be a distributed-training wrapper.
            // These models use their own internal optimizers and don't benefit from the outer
            // optimizer's clone-evaluate-select loop. Train directly on the full training data.
            // For clustering/density models, train on the full dataset (not the
            // train/test split) since cluster structure depends on having all data points.
            // Note: preprocessedX/preprocessedY may only contain the training split in the
            // standard (non-federated) path. For clustering we need the complete dataset.
            //
            // Probe the *unwrapped* model: by this point the local `model` variable may be a
            // DDP/FSDP/ZeRO* distributed-training wrapper (see lines ~1921-1953), and a wrapper
            // is never a ClusteringBase<T> so the check would silently flip to false and route
            // clustering training back through the train/test split — which is exactly the bug
            // this clustering-data path was added to prevent.
            bool useFullData = _model is Clustering.Base.ClusteringBase<T>;
            // Clustering models need ALL data points for correct density estimation.
            // Use preparedX/preparedY (the full dataset before train/test split) when
            // the preprocessing pipeline is not configured. When it IS configured,
            // apply the (already-fitted) pipeline to the full data so models operate
            // in the same coordinate space as predictions.
            TInput fullX;
            TOutput fullY;
            if (useFullData)
            {
                if (_preprocessingPipeline is not null && _preprocessingPipeline.IsFitted)
                {
                    fullX = _preprocessingPipeline.Transform(preparedX);
                    fullY = preparedY;
                }
                else
                {
                    fullX = preparedX;
                    fullY = preparedY;
                }
            }
            else
            {
                fullX = XTrain;
                fullY = yTrain;
            }
            var directX = fullX;
            var directY = fullY;

            // Override the earlier split-based sample counts logged at lines ~2222-2224 so
            // experiment run metadata reflects the dataset actually fed to Train(). Without
            // this, clustering runs would be reported with a train/val/test split that was
            // never used (we just trained on the full dataset).
            if (useFullData && experimentRun is not null)
            {
                int effectiveSamples = InputHelper<T, TInput>.GetInputSize(directX);
                experimentRun.LogParameters(new Dictionary<string, object>
                {
                    ["training_samples"] = effectiveSamples,
                    ["validation_samples"] = 0,
                    ["test_samples"] = 0,
                    ["full_data_used"] = true,
                });
            }

            model.Train(directX, directY);

            // Compute evaluation metrics
            int inputSize = InputHelper<T, TInput>.GetInputSize(directX);
            TOutput trainPredOutput = model.Predict(directX);
            var trainPredVec = ConversionsHelper.ConvertToVector<T, TOutput>(trainPredOutput);
            var trainActual = ConversionsHelper.ConvertToVector<T, TOutput>(directY);

            var trainErrorStats = new ErrorStats<T>(new ErrorStatsInputs<T>
            {
                Actual = trainActual,
                Predicted = trainPredVec,
                FeatureCount = inputSize
            });
            var trainPredStats = new PredictionStats<T>(new PredictionStatsInputs<T>
            {
                Actual = trainActual,
                Predicted = trainPredVec,
                NumberOfParameters = inputSize
            });

            // trainPredOutput is already TOutput from model.Predict

            optimizationResult = new OptimizationResult<T, TInput, TOutput>
            {
                BestSolution = model,
                Iterations = 1,
                SelectedFeatureIndices = Enumerable.Range(0, inputSize).ToList(),
                TrainingResult = new OptimizationResult<T, TInput, TOutput>.DatasetResult
                {
                    X = directX, Y = directY, Predictions = trainPredOutput,
                    ErrorStats = trainErrorStats,
                    PredictionStats = trainPredStats
                }
            };
        }
        else
        {
            // REGULAR TRAINING PATH
            if (_knowledgeDistillationOptions is not null)
            {
                // Knowledge-distillation tape integration is still pending
                // (the teacher-aware loss combiner needs a tape-aware
                // wrapper around the standard loss to keep gradients
                // flowing through both terms). Until that lands, surface
                // a runtime warning so users discover the gap early
                // rather than after Build returns silently. Restore the
                // hard contract: a user who called ConfigureKnowledgeDistillation
                // expects KD to actually run; downgrading the original
                // NotSupportedException to a Trace warning silently broke
                // that contract (review #1368). The configured options
                // still round-trip onto AiModelResult.KnowledgeDistillationOptions
                // for consumers who want to drive distillation manually,
                // but they must opt out of automatic Build-time KD by
                // omitting ConfigureKnowledgeDistillation OR by switching
                // to a model path that supports it.
                throw new NotSupportedException(
                    "ConfigureKnowledgeDistillation is not yet integrated with the tape-based training flow " +
                    "for this model path. Either omit ConfigureKnowledgeDistillation, switch to a model " +
                    "type that supports it (parametric-model branch), or drive distillation manually " +
                    "post-build via AiModelResult.KnowledgeDistillationOptions. Track upstream integration " +
                    "in the AiDotNet repo issues.");
            }

            // Ensure the optimizer has the model configured before optimization
            // This is required for InitializeRandomSolution to access model.ParameterCount
            finalOptimizer.SetModel(model);

            if (_trainingGroups is not null
                && model is NeuralNetworks.NeuralNetworkBase<T> groupedNet
                && XTrain is Tensor<T> groupedX && yTrain is Tensor<T> groupedY)
            {
                // GROUPED training (ConfigureTrainingGroups): one fit per QUERY GROUP per epoch — the
                // shape pairwise/listwise ranking losses require (pooled fits give the loss conflicting
                // targets across groups and the net collapses to a constant). Epoch budget comes from the
                // configured optimizer's MaxIterations, matching the facade's standard epoch source.
                int groupedEpochs = finalOptimizer.GetOptions()?.MaxIterations ?? 100;
                foreach (var group in _trainingGroups)
                {
                    foreach (var idx in group)
                    {
                        if (idx < 0 || idx >= groupedX.Shape[0])
                        {
                            throw new ArgumentOutOfRangeException(nameof(_trainingGroups),
                                $"Training-group row index {idx} is outside the training set (0..{groupedX.Shape[0] - 1}). " +
                                "Group indices refer to TRAINING rows AFTER the facade's internal split/shuffle — " +
                                "for date-grouped data configure a time-series model or pre-split data so ordering is preserved.");
                        }
                    }
                }

                for (int epoch = 0; epoch < groupedEpochs; epoch++)
                {
                    foreach (var group in _trainingGroups)
                    {
                        if (group.Count == 0)
                        {
                            continue;
                        }

                        var groupX = GatherRows(groupedX, group);
                        var groupY = GatherRows(groupedY, group);
                        if (groupY.Rank == 1)
                        {
                            // Batch training produces [B, outputSize] predictions — promote rank-1
                            // targets to a column [B, 1], exactly as TryStackTensorBatch does for
                            // the streaming path's per-sample [1] targets.
                            groupY = groupY.Reshape(new[] { groupY.Shape[0], 1 });
                        }

                        groupedNet.Train(groupX, groupY);
                    }
                }

                optimizationResult = new OptimizationResult<T, TInput, TOutput>
                {
                    BestSolution = model,
                };
            }
            else
            {
                // Optimize the final model on the full training set
                optimizationResult = finalOptimizer.Optimize(optimizationInputData);
            }
        }

        // ============================================================================
        // FINE-TUNING (#1357 / #1361) — applies preference learning, RLHF, SFT, etc.
        // to the optimizer-trained model BEFORE metric finalization so that any
        // checkpoint/result returned reflects the post-fine-tune weights.
        // ============================================================================
        if (_fineTuningConfiguration?.Enabled == true)
        {
            var ftImpl = _fineTuningConfiguration.Implementation
                ?? throw new InvalidOperationException(
                    "ConfigureFineTuning was enabled but no Implementation was provided. " +
                    "Set FineTuningConfiguration.Implementation to a concrete IFineTuning<T, TInput, TOutput> instance " +
                    "(e.g. new SupervisedFineTuning<...>(options), new DirectPreferenceOptimization<...>(options)).");
            if (_fineTuningConfiguration.TrainingData is null)
                throw new InvalidOperationException(
                    "ConfigureFineTuning was enabled but no TrainingData was supplied. " +
                    "Set FineTuningConfiguration.TrainingData to a FineTuningData<T, TInput, TOutput> appropriate for the chosen method " +
                    "(SFT needs Inputs+Outputs; DPO/SimPO need preference pairs; RLHF/GRPO need rewards).");
            if (optimizationResult.BestSolution is null)
                throw new InvalidOperationException(
                    "ConfigureFineTuning was enabled but the optimizer did not produce a BestSolution to fine-tune. " +
                    "This usually means main training failed silently — check earlier logs.");

            // Honor the BuildAsync caller's cancellation token across the
            // fine-tune await — without this the awaited operation cannot
            // be cancelled once the optimizer's main-training pass returns
            // (review #1361 fix #5).
            var fineTunedModel = await ftImpl.FineTuneAsync(
                optimizationResult.BestSolution,
                _fineTuningConfiguration.TrainingData,
                cancellationToken).ConfigureAwait(false);

            // Rebind so downstream metric/checkpoint code sees the post-FT
            // model. The optimizer-result metrics are still pre-FT (computed
            // earlier in this method) — a full rebind that re-evaluates loss
            // on the fine-tuned model is tracked as a heavy-lift follow-up
            // (review #1361 fix #4 partial).
            optimizationResult.BestSolution = fineTunedModel;
            _model = fineTunedModel;
        }

        // ============================================================================
        // STAGED TRAINING PIPELINE (#1361 #2) — executes the user-defined sequence of
        // training stages after main training (and any one-shot fine-tuning). Each
        // stage takes the previous stage's output model + its own training data and
        // produces the next stage's input model. Stages with Enabled=false or whose
        // RunCondition returns false are skipped. The final stage's output replaces
        // optimizationResult.BestSolution so downstream consumers see the post-
        // pipeline weights.
        // ============================================================================
        if (_trainingPipelineConfiguration?.Stages is { Count: > 0 } stages)
        {
            if (optimizationResult.BestSolution is null)
                throw new InvalidOperationException(
                    "ConfigureTrainingPipeline was provided but main training did not produce a BestSolution. " +
                    "Check earlier logs for an upstream training failure.");

            var currentModel = optimizationResult.BestSolution;
            TrainingStageResult<T, TInput, TOutput>? previousStageResult = null;

            for (int stageIndex = 0; stageIndex < stages.Count; stageIndex++)
            {
                var stage = stages[stageIndex];
                if (!stage.Enabled) continue;
                if (stage.RunCondition is { } cond && !cond(previousStageResult)) continue;

                if (stage.CustomTrainingFunction is null)
                    throw new InvalidOperationException(
                        $"TrainingPipeline stage '{stage.Name}' (index {stageIndex}) has Enabled=true but no " +
                        $"CustomTrainingFunction. The current wire-up requires each enabled stage to provide " +
                        $"its own training delegate; the StageType={stage.StageType} / FineTuningMethod=" +
                        $"{stage.FineTuningMethod} auto-dispatch path is not yet implemented. " +
                        $"Set stage.CustomTrainingFunction to an async (model, data, ct) => trainedModel delegate, " +
                        $"or remove this stage from the pipeline.");
                if (stage.TrainingData is null && !stage.IsEvaluationOnly)
                    throw new InvalidOperationException(
                        $"TrainingPipeline stage '{stage.Name}' (index {stageIndex}) has no TrainingData and is " +
                        $"not marked IsEvaluationOnly. Each training stage needs a FineTuningData<T, TInput, " +
                        $"TOutput> appropriate for its FineTuningMethod.");

                var stageStart = DateTime.UtcNow;
                var stageResult = new TrainingStageResult<T, TInput, TOutput>
                {
                    StageName = stage.Name,
                    StageIndex = stageIndex,
                };
                try
                {
                    if (!stage.IsEvaluationOnly)
                    {
                        // Pass through the BuildAsync caller's cancellation token
                        // so user-supplied stage delegates can honor cancellation
                        // (review #1361 fix #5).
                        currentModel = await stage.CustomTrainingFunction(
                            currentModel,
                            stage.TrainingData!,
                            cancellationToken).ConfigureAwait(false);
                        if (currentModel is null)
                            throw new InvalidOperationException(
                                $"TrainingPipeline stage '{stage.Name}' returned null from " +
                                $"CustomTrainingFunction. Stages must return a non-null model.");
                    }
                    stageResult.Model = currentModel;
                    stageResult.Success = true;
                }
                catch (Exception ex)
                {
                    stageResult.Success = false;
                    stageResult.ErrorMessage = ex.Message;
                    throw;
                }
                finally
                {
                    stageResult.Duration = DateTime.UtcNow - stageStart;
                }
                previousStageResult = stageResult;
            }

            // Rebind _model so downstream weight-streaming reports, checkpoint
            // writes, and any other consumer that reads _model directly sees
            // the post-pipeline model (review #1361 fix #4 partial — the
            // optimizer metrics on optimizationResult are still pre-pipeline;
            // re-evaluating them on the pipeline-output model is the heavy-
            // lift follow-up).
            optimizationResult.BestSolution = currentModel;
            _model = currentModel;
        }

        // ============================================================================
        // CURRICULUM LEARNING (#1361 #3) — runs a curriculum-scheduled refinement pass
        // over a user-supplied Dataset after main training (and any fine-tuning /
        // pipeline stages). The CurriculumLearner ranks samples by difficulty using
        // either the user's CustomDifficultyEstimator or a LossBasedDifficultyEstimator
        // tied to the trained model's internal loss, then trains in phases from easy
        // to hard. The post-curriculum model replaces optimizationResult.BestSolution.
        //
        // Dataset auto-extraction from the configured DataLoader is out-of-scope for
        // this wire-up — different loaders have different per-sample contracts. When
        // the caller does not supply CurriculumLearningOptions.Dataset, the curriculum
        // pass is skipped (configuration-only mode).
        // ============================================================================
        if (_curriculumLearningOptions is not null && _curriculumLearningOptions.Dataset is not null)
        {
            if (optimizationResult.BestSolution is null)
                throw new InvalidOperationException(
                    "ConfigureCurriculumLearning was provided with a Dataset but main training did not " +
                    "produce a BestSolution to curriculum-train. Check earlier logs for an upstream " +
                    "training failure.");

            var curriculumConfig = new AiDotNet.CurriculumLearning.CurriculumLearnerConfig<T>
            {
                TotalEpochs = _curriculumLearningOptions.TotalEpochs ?? 100,
                NumPhases = _curriculumLearningOptions.NumPhases ?? 5,
                InitialDataFraction = MathHelper.GetNumericOperations<T>().FromDouble(
                    _curriculumLearningOptions.InitialDataFraction ?? 0.2),
                FinalDataFraction = MathHelper.GetNumericOperations<T>().FromDouble(
                    _curriculumLearningOptions.FinalDataFraction ?? 1.0),
                ScheduleType = _curriculumLearningOptions.ScheduleType,
                RecalculateDifficulties = _curriculumLearningOptions.RecalculateDifficulties ?? false,
                DifficultyRecalculationFrequency = _curriculumLearningOptions.DifficultyRecalculationFrequency ?? 10,
                NormalizeDifficulties = _curriculumLearningOptions.NormalizeDifficulties ?? true,
                EarlyStoppingPatience = _curriculumLearningOptions.EarlyStopping?.Patience ?? 10,
                EarlyStoppingMinDelta = MathHelper.GetNumericOperations<T>().FromDouble(
                    _curriculumLearningOptions.EarlyStopping?.MinDelta ?? 0.001),
                UseEarlyStopping = _curriculumLearningOptions.EarlyStopping?.Enabled ?? true,
                BatchSize = _curriculumLearningOptions.BatchSize ?? 32,
                LearningRate = MathHelper.GetNumericOperations<T>().FromDouble(0.001),
                ShuffleWithinPhase = _curriculumLearningOptions.ShuffleWithinPhase ?? true,
                UseDifficultyWeighting = _curriculumLearningOptions.UseDifficultyWeighting ?? false,
                RandomSeed = _curriculumLearningOptions.RandomSeed,
                Verbosity = _curriculumLearningOptions.Verbosity,
            };

            var difficultyEstimator = _curriculumLearningOptions.CustomDifficultyEstimator
                ?? new AiDotNet.CurriculumLearning.DifficultyEstimators
                       .LossBasedDifficultyEstimator<T, TInput, TOutput>(
                           lossFunction: null,
                           normalize: curriculumConfig.NormalizeDifficulties);

            var curriculumLearner = new AiDotNet.CurriculumLearning.CurriculumLearner<T, TInput, TOutput>(
                baseModel: optimizationResult.BestSolution,
                config: curriculumConfig,
                difficultyEstimator: difficultyEstimator,
                scheduler: _curriculumLearningOptions.CustomScheduler);

            curriculumLearner.Train(_curriculumLearningOptions.Dataset);

            // The CurriculumLearner mutates BaseModel in place; reassign to make the
            // post-curriculum weights explicit for downstream consumers.
            // Rebind _model so downstream consumers see the post-curriculum
            // weights (review #1361 fix #4 partial).
            optimizationResult.BestSolution = curriculumLearner.BaseModel;
            _model = curriculumLearner.BaseModel;
        }

        var trainingEndTime = DateTime.UtcNow;
        var trainingDuration = trainingEndTime - trainingStartTime;

        // ============================================================================
        // Finalize Training Infrastructure (after optimization)
        // ============================================================================

        // Collect final metrics from optimization result
        var finalMetrics = new Dictionary<string, T>();
        if (optimizationResult.TrainingResult.ErrorStats is not null)
        {
            finalMetrics["training_rmse"] = optimizationResult.TrainingResult.ErrorStats.RMSE;
            finalMetrics["training_mae"] = optimizationResult.TrainingResult.ErrorStats.MAE;
            finalMetrics["training_mse"] = optimizationResult.TrainingResult.ErrorStats.MSE;
        }
        if (optimizationResult.ValidationResult.ErrorStats is not null)
        {
            finalMetrics["validation_rmse"] = optimizationResult.ValidationResult.ErrorStats.RMSE;
            finalMetrics["validation_mae"] = optimizationResult.ValidationResult.ErrorStats.MAE;
            finalMetrics["validation_mse"] = optimizationResult.ValidationResult.ErrorStats.MSE;
        }
        if (optimizationResult.TestResult.ErrorStats is not null)
        {
            finalMetrics["test_rmse"] = optimizationResult.TestResult.ErrorStats.RMSE;
            finalMetrics["test_mae"] = optimizationResult.TestResult.ErrorStats.MAE;
            finalMetrics["test_mse"] = optimizationResult.TestResult.ErrorStats.MSE;
        }

        // Log final metrics to experiment run
        if (experimentRun is not null)
        {
            experimentRun.LogMetrics(finalMetrics, step: 1);
            experimentRun.LogParameter("training_duration_seconds", trainingDuration.TotalSeconds);
            experimentRun.Complete();
        }

        // Log final metrics to training monitor
        if (_trainingMonitor is not null && monitorSessionId is not null)
        {
            _trainingMonitor.LogMetrics(monitorSessionId, finalMetrics, step: 1);
            _trainingMonitor.EndSession(monitorSessionId);
        }

        // Save checkpoint if checkpoint manager is configured
        if (_checkpointManager is not null && optimizationResult.BestSolution is not null)
        {
            var checkpointMetrics = new Dictionary<string, T>(finalMetrics);
            var checkpointMetadata = new Dictionary<string, object>
            {
                ["experiment_run_id"] = experimentRunId ?? string.Empty,
                ["training_duration_seconds"] = trainingDuration.TotalSeconds,
                ["model_type"] = optimizationResult.BestSolution.GetType().Name
            };

            checkpointPath = _checkpointManager.SaveCheckpoint(
                model: optimizationResult.BestSolution,
                optimizer: finalOptimizer,
                epoch: 1,
                step: 1,
                metrics: checkpointMetrics,
                metadata: checkpointMetadata);
        }

        // Register model in model registry if configured
        if (_modelRegistry is not null && optimizationResult.BestSolution is not null)
        {
            registeredModelName = $"{model.GetType().Name}-{trainingStartTime:yyyyMMdd-HHmmss}";

            var modelMetadata = new ModelMetadata<T>
            {
                Name = registeredModelName,
                Version = "1.0",
                TrainingDate = trainingStartTime,
                FeatureCount = convertedX.Columns,
                Complexity = optimizationResult.BestSolution.GetType().GetProperties().Length,
                Description = $"Model trained via AiModelBuilder on {trainingStartTime:yyyy-MM-dd HH:mm:ss} UTC",
                AdditionalInfo = new Dictionary<string, object>
                {
                    ["experiment_run_id"] = experimentRunId ?? string.Empty,
                    ["training_duration_seconds"] = trainingDuration.TotalSeconds,
                    ["optimizer_type"] = finalOptimizer.GetType().Name
                }
            };

            // Add final metrics to AdditionalInfo (only non-null values)
            foreach (var metric in finalMetrics)
            {
                if (metric.Value is not null)
                {
                    modelMetadata.AdditionalInfo[$"metric_{metric.Key}"] = metric.Value;
                }
            }

            // Register the model name first — CreateModelVersion requires a
            // pre-existing registration. Without this call, the upstream
            // CreateModelVersion throws ArgumentException ("Model not found
            // in registry"). Discovered by AiDotNet#1345's integration-test
            // framework (Bucket3_QualityOfLifeTests
            // ConfigureModelRegistry_AndBuildAsync_TracksTrainedModel).
            _modelRegistry.RegisterModel(
                name: registeredModelName,
                model: optimizationResult.BestSolution,
                metadata: modelMetadata,
                tags: new Dictionary<string, string>
                {
                    ["auto-registered"] = "true",
                    ["source"] = "build-async",
                    ["registered-at"] = trainingStartTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
                });

            modelVersion = _modelRegistry.CreateModelVersion(
                modelName: registeredModelName,
                model: optimizationResult.BestSolution,
                metadata: modelMetadata,
                description: $"Auto-registered from training run {experimentRunId ?? "unknown"}");
        }

        // Apply uncertainty quantification if configured
        ApplyUncertaintyQuantificationIfConfigured(optimizationResult.BestSolution, _uncertaintyQuantificationOptions);

        // Create deployment configuration from individual configs
        var deploymentConfig = DeploymentConfiguration.Create(
            _quantizationConfig,
            _cacheConfig,
            _versioningConfig,
            _abTestingConfig,
            _telemetryConfig,
            _exportConfig,
            _gpuAccelerationConfig,
            _compressionConfig,
            _profilingConfig);

        // Build hyperparameters dictionary from optimizer options for result tracking
        var hyperparameters = new Dictionary<string, object>();
        try
        {
            var opts = finalOptimizer.GetOptions();
            hyperparameters["max_iterations"] = opts.MaxIterations;
            hyperparameters["use_early_stopping"] = opts.UseEarlyStopping;
            hyperparameters["early_stopping_patience"] = opts.EarlyStoppingPatience;
            hyperparameters["model_type"] = model.GetType().Name;
            hyperparameters["optimizer_type"] = finalOptimizer.GetType().Name;
        }
        catch (Exception)
        {
            // Ignore errors collecting hyperparameters - they are optional
        }

        // Build training metrics history from optimization result
        var trainingMetricsHistory = new Dictionary<string, List<double>>();
        if (optimizationResult.FitnessHistory is not null && optimizationResult.FitnessHistory.Length > 0)
        {
            var fitnessHistoryAsDouble = new List<double>();
            for (int i = 0; i < optimizationResult.FitnessHistory.Length; i++)
            {
                // Use Convert.ToDouble for generic type conversion (standard pattern in this codebase)
                fitnessHistoryAsDouble.Add(Convert.ToDouble(optimizationResult.FitnessHistory[i]));
            }
            trainingMetricsHistory["fitness"] = fitnessHistoryAsDouble;
        }

        // QUANTIZATION (if configured)
        // Apply post-training quantization to reduce model size and improve inference speed
        QuantizationInfo? quantizationInfo = null;
        if (_quantizationConfig != null && _quantizationConfig.Mode != QuantizationMode.None && optimizationResult.BestSolution != null)
        {
            try
            {
                // Use preprocessed training data as calibration data for consistent quantization
                // This ensures calibration sees the same data distribution as during training
                int calibrationSampleCount = _quantizationConfig?.CalibrationSamples ?? 100;
                IEnumerable<TInput>? calibrationData = null;
                if (XTrain is TInput[] xTrainArray)
                {
                    calibrationData = xTrainArray.Take(Math.Min(calibrationSampleCount, xTrainArray.Length));
                }
                else if (XTrain is IEnumerable<TInput> xTrainEnumerable)
                {
                    calibrationData = xTrainEnumerable.Take(calibrationSampleCount);
                }
                else if (XTrain != null)
                {
                    // Wrap single item as calibration data if not enumerable
                    calibrationData = new[] { XTrain };
                }

                var (quantizedModel, quantizationInfoResult) = ApplyQuantizationIfConfigured(
                    optimizationResult.BestSolution,
                    _quantizationConfig,
                    calibrationData);

                if (quantizedModel != null && quantizationInfoResult != null)
                {
                    // CRITICAL: Use the quantized model instead of the original
                    optimizationResult.BestSolution = quantizedModel;
                    quantizationInfo = quantizationInfoResult;
                    var strategy = _quantizationConfig?.Strategy ?? QuantizationStrategy.Dynamic;
                    Console.WriteLine($"Quantization applied: {strategy} strategy, " +
                        $"{quantizationInfo.BitWidth}-bit, compression ratio: {quantizationInfo.CompressionRatio:F2}x");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Quantization failed: {ex.Message}. Model will use original precision.");
            }
        }

        // Fit the postprocessing pipeline on the model's training-set
        // predictions BEFORE attaching it to the result. Routed through
        // the shared FitPostprocessingIfNeeded helper so every Build*
        // path applies identical fit/fail logic (review #1368 C6WJG).
        FitPostprocessingIfNeeded(optimizationResult.BestSolution, XTrain, nameof(BuildSupervisedInternalAsync));

        // Return AiModelResult with CV results, agent data, JIT compilation, reasoning config, and training infrastructure
        var options = new AiModelResultOptions<T, TInput, TOutput>
        {
            OptimizationResult = optimizationResult,
            PreprocessingInfo = preprocessingInfo,
            PostprocessingPipeline = _postprocessingPipeline,
            KnowledgeDistillationOptions = _knowledgeDistillationOptions,
            AutoMLSummary = autoMLSummary,
            BiasDetector = _biasDetector,
            FairnessEvaluator = _fairnessEvaluator,
            InterpretabilityOptions = _interpretabilityOptions,
            RagRetriever = _ragRetriever,
            RagReranker = _ragReranker,
            RagGenerator = _ragGenerator,
            QueryProcessors = _queryProcessors,
            LoRAConfiguration = _loraConfiguration,
            CrossValidationResult = cvResults,
            DeploymentConfiguration = deploymentConfig,
            QuantizationInfo = quantizationInfo,
            InferenceOptimizationConfig = _inferenceOptimizationConfig,
            JitCompilationConfig = _jitCompilationConfig,
            JitCompiledFunction = BuildCompiledPredictFunction(optimizationResult.BestSolution),
            AllowNondeterminism = _allowNondeterminism,
            AugmentationConfig = _augmentationConfig,
            ReasoningConfig = _reasoningConfig,
            KnowledgeGraph = _knowledgeGraph,
            GraphStore = _graphStore,
            HybridGraphRetriever = _hybridGraphRetriever,
            Tokenizer = _tokenizer,
            TokenizationConfig = _tokenizationConfig,
            ProgramSynthesisModel = _programSynthesisModel,
            ProgramSynthesisServingClient = _programSynthesisServingClient,
            ProgramSynthesisServingClientOptions = _programSynthesisServingClientOptions,
            PromptTemplate = null,
            PromptOptimizer = null,
            FewShotExampleSelector = null,
            PromptAnalyzer = null,
            PromptCompressor = null,
            // Diagnostics Properties
            ProfilingReport = profilerSession?.GetReport(),
            WeightStreamingReport = BuildWeightStreamingReport(),

            // Training Infrastructure Properties
            MemoryConfig = _memoryConfig,
            ExperimentRunId = experimentRunId,
            ExperimentId = experimentId,
            ModelVersion = modelVersion,
            RegisteredModelName = registeredModelName,
            CheckpointPath = checkpointPath,
            DataVersionHash = dataVersionHash,
            Hyperparameters = hyperparameters.Count > 0 ? hyperparameters : null,
            TrainingMetricsHistory = trainingMetricsHistory.Count > 0 ? trainingMetricsHistory : null,

            // Hyperparameter Optimization Properties
            HyperparameterOptimizationResult = hyperparameterOptimizationResult,
            HyperparameterTrialId = bestHyperparameterTrialId
        };

        var finalResult = AttachDiagnostics(new AiModelResult<T, TInput, TOutput>(options));

        finalResult.SetUncertaintyQuantificationOptions(_uncertaintyQuantificationOptions);
        TryComputeAndAttachDeepEnsembleModels(finalResult, deepEnsembleTemplate, optimizationInputData, optimizer, _uncertaintyQuantificationOptions);
        TryComputeAndAttachUncertaintyCalibrationArtifacts(finalResult);

        if (causalDiscoveryResult != null)
        {
            finalResult.SetCausalDiscoveryResult(causalDiscoveryResult);
        }

        if (federatedLearningMetadata != null)
        {
            finalResult.GetModelMetadata().SetProperty(FederatedLearningMetadata.MetadataKey, federatedLearningMetadata);
        }

        ProcessKnowledgeGraphOptions(finalResult);

        // Build and attach the composable safety pipeline if configured
        AttachSafetyPipeline(finalResult);
        AttachAdversarialRobustness(finalResult);

        return finalResult;
    }
}
