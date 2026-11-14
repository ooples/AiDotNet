using System.Globalization;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.CheckpointManagement;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <summary>
/// Step-level streaming training support (<see cref="ConfigureStreamingTraining"/>): distributed coordinator lifecycle,
/// lazy-layer shape synchronization, gradient accumulation, coordinated Rank-0 auto-checkpointing, cross-world-size
/// checkpoint resume, and synchronized held-out validation.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    internal const string StreamingEpochKey = "streaming.epoch";
    internal const string StreamingBatchInEpochKey = "streaming.batch_in_epoch";
    internal const string StreamingGlobalStepKey = "streaming.global_step";
    internal const string StreamingSeedKey = "streaming.seed";
    internal const string StreamingBatchSizeKey = "streaming.batch_size";
    internal const string StreamingSamplesInEpochKey = "streaming.samples_in_epoch";
    internal const string StreamingGlobalBatchSizeKey = "streaming.global_batch_size";
    internal const string StreamingWorldSizeKey = "streaming.world_size";
    internal const string StreamingMpScaleKey = "mp.scale";
    internal const string StreamingMpGrowthTrackerKey = "mp.growth_tracker";

    private static readonly FieldInfo? LossScalerGrowthTrackerField =
        typeof(MixedPrecision.LossScaler<float>).GetField(
            "_consecutiveSuccessfulUpdates",
            BindingFlags.Instance | BindingFlags.NonPublic);

    private DistributedTraining.ICommunicationBackend<T>? ResolveStreamingDistributedBackend()
    {
        return _distributedBackend ?? _distributedConfiguration?.CommunicationBackend;
    }

    private (bool Engaged, string Status) ConfigureStreamingMixedPrecision()
    {
        if (_mixedPrecisionConfig is null)
        {
            return (false, "not requested");
        }

        if (typeof(T) != typeof(float))
        {
            throw new InvalidOperationException(
                $"Mixed-precision training requires T = float, got T = {typeof(T).Name}. " +
                "Use AiModelBuilder<float, ...> to enable mixed-precision training.");
        }

        if (_model is NeuralNetworks.NeuralNetworkBase<T> mpNeuralNet)
        {
            if (mpNeuralNet.IsMixedPrecisionEnabled)
            {
                mpNeuralNet.DisableMixedPrecision();
            }

            mpNeuralNet.EnableMixedPrecision(_mixedPrecisionConfig);
            return (true, $"engaged: {_mixedPrecisionConfig.PrecisionType} on streaming path");
        }

        return (false, $"ignored: {_model?.GetType().Name ?? "null"} does not support mixed precision on the streaming path");
    }

    private async Task InitializeStreamingLazyAndDistributedStateAsync(
        IStreamingDataLoader<T, TInput, TOutput> rawLoader,
        DistributedTraining.ICommunicationBackend<T>? distBackend,
        DistributedTraining.DistributedTapeCoordinator<T>? distCoordinator,
        CancellationToken cancellationToken)
    {
        if (_model is not NeuralNetworks.NeuralNetworkBase<T> neuralNet)
        {
            return;
        }

        if (_optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> gbo)
        {
            neuralNet.SetBaseTrainOptimizer(gbo);
        }

        if (distBackend is null || distBackend.WorldSize <= 1)
        {
            bool canWarmupBeforePipelineFit = distBackend is not null
                || _streamingTrainingOptions?.ResumeFromLatestCheckpoint == true
                || _preprocessingPipeline is not { IsFitted: false };
            if (canWarmupBeforePipelineFit && neuralNet.HasLazyUninitializedLayers())
            {
                if (!rawLoader.IsLoaded)
                {
                    await rawLoader.LoadAsync(cancellationToken);
                }

                await foreach (var (peekInputs, _) in rawLoader.GetBatchesAsync(shuffle: false, cancellationToken: cancellationToken))
                {
                    if (peekInputs.Length == 0)
                    {
                        continue;
                    }

                    var candidateInput = _preprocessingPipeline is { IsFitted: true } fittedPipe
                        && fittedPipe.Transform(peekInputs[0]) is TInput transformedInput
                            ? transformedInput
                            : peekInputs[0];
                    if (candidateInput is Tensor<T> sampleTensor)
                    {
                        WarmupLazyModelWithSampleShape(neuralNet, sampleTensor._shape);
                    }

                    break;
                }

                neuralNet.ActiveMixedPrecisionContext?.ClearFp32SnapshotPool();
            }

            distCoordinator?.EnsureInitialParametersSynchronized(
                neuralNet.CollectTrainableParameters(),
                neuralNet.InvalidateLayerWeightCaches);
            return;
        }

        int localLazy = neuralNet.HasLazyUninitializedLayers() ? 1 : 0;
        var numOps = MathHelper.GetNumericOperations<T>();
        var lazyFlag = new Vector<T>(1);
        lazyFlag.AsWritableSpan()[0] = numOps.FromDouble(localLazy);
        distBackend.AllReduce(lazyFlag, DistributedTraining.ReductionOperation.Max);
        bool anyRankNeedsLazyInit = numOps.ToDouble(lazyFlag.AsSpan()[0]) > 0.0;

        if (anyRankNeedsLazyInit)
        {
            int[]? rank0SampleShape = null;
            if (distBackend.Rank == 0)
            {
                if (!rawLoader.IsLoaded)
                {
                    await rawLoader.LoadAsync(cancellationToken);
                }

                await foreach (var (peekInputs, _) in rawLoader.GetBatchesAsync(shuffle: false, cancellationToken: cancellationToken))
                {
                    if (peekInputs.Length == 0)
                    {
                        continue;
                    }

                    if (peekInputs[0] is Tensor<T> sampleTensor)
                    {
                        rank0SampleShape = sampleTensor._shape;
                    }

                    break;
                }
            }

            int[]? sharedShape = BroadcastIntShape(distBackend, rank0SampleShape, root: 0);
            if (sharedShape is { Length: > 0 } && localLazy == 1)
            {
                WarmupLazyModelWithSampleShape(neuralNet, sharedShape);
            }

            neuralNet.ActiveMixedPrecisionContext?.ClearFp32SnapshotPool();
            distCoordinator?.ResetParameterSync();
        }

        distCoordinator?.EnsureInitialParametersSynchronized(
            neuralNet.CollectTrainableParameters(),
            neuralNet.InvalidateLayerWeightCaches);
        neuralNet.ActiveMixedPrecisionContext?.ClearFp32SnapshotPool();
    }

    private void WarmupLazyModelWithSampleShape(NeuralNetworks.NeuralNetworkBase<T> neuralNet, int[] sampleShape)
    {
        if (_model is null || !typeof(TInput).IsAssignableFrom(typeof(Tensor<T>)))
        {
            return;
        }

        var batchShape = new int[sampleShape.Length + 1];
        batchShape[0] = 1;
        Array.Copy(sampleShape, 0, batchShape, 1, sampleShape.Length);
        var dummyInput = new Tensor<T>(batchShape);

        bool wasTraining = neuralNet.IsTrainingMode;
        if (wasTraining)
        {
            neuralNet.SetTrainingMode(false);
        }

        try
        {
            _model.Predict((TInput)(object)dummyInput);
        }
        catch
        {
            // Best-effort lazy shape resolution; any genuine shape mismatch will surface on the first real training batch.
        }
        finally
        {
            if (wasTraining)
            {
                neuralNet.SetTrainingMode(true);
            }
        }
    }

    private static Dictionary<string, T> StreamingCheckpointMetrics(T trainLoss, bool hasValidationLoss, T? validationLoss)
    {
        var metrics = new Dictionary<string, T> { ["loss"] = trainLoss };
        if (hasValidationLoss && validationLoss is not null)
        {
            metrics["validation_loss"] = validationLoss;
        }

        return metrics;
    }

    private Dictionary<string, object> StreamingCheckpointMetadata(
        int epoch,
        int batchInEpoch,
        long samplesInEpoch,
        long globalStep,
        StreamingTrainingOptions<T, TInput, TOutput>? options,
        IStreamingDataLoader<T, TInput, TOutput> rawLoader,
        Data.Loaders.DistributedStreamingDataLoader<T, TInput, TOutput>? distLoader,
        DistributedTraining.ICommunicationBackend<T>? distBackend,
        int gradientAccumulationSteps)
    {
        int worldSize = distBackend is not null ? Math.Max(1, distBackend.WorldSize) : 1;
        int baseGlobalBatch = distLoader is not null ? distLoader.GlobalBatchSize : rawLoader.BatchSize;
        int effectiveGlobalBatch = checked(baseGlobalBatch * Math.Max(1, gradientAccumulationSteps));

        var metadata = new Dictionary<string, object>
        {
            [StreamingEpochKey] = epoch,
            [StreamingBatchInEpochKey] = batchInEpoch,
            [StreamingSamplesInEpochKey] = samplesInEpoch,
            [StreamingGlobalStepKey] = globalStep,
            [StreamingBatchSizeKey] = rawLoader.BatchSize,
            [StreamingGlobalBatchSizeKey] = effectiveGlobalBatch,
            [StreamingWorldSizeKey] = worldSize,
        };

        if (options?.Seed is int seed)
        {
            metadata[StreamingSeedKey] = seed;
        }

        if (TryReadLossScalerState(_model, out double mpScale, out int mpGrowthTracker))
        {
            metadata[StreamingMpScaleKey] = mpScale;
            metadata[StreamingMpGrowthTrackerKey] = mpGrowthTracker;
        }

        return metadata;
    }

    private static bool TryReadLossScalerState(
        IFullModel<T, TInput, TOutput>? model,
        out double scale,
        out int growthTracker)
    {
        if (model is NeuralNetworks.NeuralNetworkBase<T> { ActiveMixedPrecisionContext.LossScaler: { } scaler })
        {
            scale = scaler.Scale;
            growthTracker = LossScalerGrowthTrackerField?.GetValue(scaler) is int gt ? gt : 0;
            return true;
        }

        scale = 1.0;
        growthTracker = 0;
        return false;
    }

    private static void RestoreLossScalerState(
        IFullModel<T, TInput, TOutput>? model,
        bool hasSavedScale,
        double savedScale,
        int savedGrowthTracker)
    {
        if (model is not NeuralNetworks.NeuralNetworkBase<T> nn || nn.ActiveMixedPrecisionContext is not { } mpCtx)
        {
            return;
        }

        mpCtx.ClearFp32SnapshotPool();
        if (hasSavedScale && mpCtx.LossScaler is { } scaler && savedScale > 0.0)
        {
            scaler.Reset(savedScale);
            LossScalerGrowthTrackerField?.SetValue(scaler, savedGrowthTracker);
        }
    }

    private (double? MetricValue, bool MinimizeMetric) ResolveAutoCheckpointMetric(Dictionary<string, T> metrics)
    {
        if (_checkpointManager is CheckpointManagerBase<T, TInput, TOutput> managerBase)
        {
            var (metricName, minimizeMetric, saveOnImprovement) = managerBase.GetAutoSaveMetricConfig();
            if (saveOnImprovement && metricName is { Length: > 0 })
            {
                if (metrics.TryGetValue(metricName, out var metricVal))
                {
                    return (ConvertToDoubleScalar(metricVal), minimizeMetric);
                }

                if (string.Equals(metricName, "val_loss", StringComparison.OrdinalIgnoreCase)
                    && metrics.TryGetValue("validation_loss", out var valLoss))
                {
                    return (ConvertToDoubleScalar(valLoss), minimizeMetric);
                }
            }

            return (null, minimizeMetric);
        }

        return (null, true);
    }

    private bool TryCoordinatedStreamingCheckpoint(
        int epoch,
        int batchInEpoch,
        long samplesInEpoch,
        long globalStep,
        T trainLoss,
        bool hasValidationLoss,
        T? validationLoss,
        StreamingTrainingOptions<T, TInput, TOutput>? streamingOptions,
        IStreamingDataLoader<T, TInput, TOutput> rawLoader,
        Data.Loaders.DistributedStreamingDataLoader<T, TInput, TOutput>? distLoader,
        DistributedTraining.ICommunicationBackend<T>? distBackend,
        DistributedTraining.DistributedTapeCoordinator<T>? distCoordinator,
        int gradientAccumulationSteps,
        bool forceSave)
    {
        if (_checkpointManager is null || _optimizer is null || _model is null)
        {
            return false;
        }

        int stepInt = checked((int)globalStep);
        var metrics = StreamingCheckpointMetrics(trainLoss, hasValidationLoss, validationLoss);
        var (metricValue, minimizeMetric) = ResolveAutoCheckpointMetric(metrics);
        var metadata = StreamingCheckpointMetadata(
            epoch,
            batchInEpoch,
            samplesInEpoch,
            globalStep,
            streamingOptions,
            rawLoader,
            distLoader,
            distBackend,
            gradientAccumulationSteps);

        if (distBackend is null || (distBackend.WorldSize <= 1 && distCoordinator is null))
        {
            if (forceSave)
            {
                _checkpointManager.SaveCheckpoint(_model, _optimizer, epoch, stepInt, metrics, metadata);
                _checkpointManager.UpdateAutoSaveState(stepInt, metricValue, minimizeMetric);
                return true;
            }

            var checkpointId = _checkpointManager.TryAutoSaveCheckpoint(
                _model,
                _optimizer,
                epoch,
                stepInt,
                metrics,
                metricValue: metricValue,
                shouldMinimize: minimizeMetric,
                metadata: metadata);
            return checkpointId is not null;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var saveDecision = new Vector<T>(1);
        if (distBackend.Rank == 0)
        {
            bool shouldSaveOnRoot = forceSave
                || _checkpointManager.ShouldAutoSaveCheckpoint(stepInt, metricValue, minimizeMetric);
            saveDecision.AsWritableSpan()[0] = numOps.FromDouble(shouldSaveOnRoot ? 1.0 : 0.0);
        }

        saveDecision = distBackend.Broadcast(saveDecision, root: 0);
        bool shouldSave = numOps.ToDouble(saveDecision.AsSpan()[0]) > 0.0;
        if (!shouldSave)
        {
            return false;
        }

        if (distCoordinator is not null
            && _model is NeuralNetworks.NeuralNetworkBase<T> nn
            && _optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> gbo)
        {
            var trainableParams = nn.CollectTrainableParameters();
            byte[] consolidatedOptBytes = distCoordinator.ConsolidateOptimizerStateBytes(gbo, trainableParams);
            bool isShardedStrategy = distCoordinator.Strategy is DistributedStrategy.ZeRO1
                or DistributedStrategy.ZeRO2
                or DistributedStrategy.ZeRO3
                or DistributedStrategy.FSDP;

            if (!isShardedStrategy)
            {
                Exception? rank0Err = null;
                if (distBackend.Rank == 0)
                {
                    try
                    {
                        _checkpointManager.SaveCheckpoint(_model, _optimizer, epoch, stepInt, metrics, metadata);
                        _checkpointManager.UpdateAutoSaveState(stepInt, metricValue, minimizeMetric);
                    }
                    catch (Exception ex)
                    {
                        rank0Err = ex;
                    }
                }

                PropagateRank0CheckpointError(distBackend, rank0Err);
            }
            else
            {
                ILearningRateScheduler? liveScheduler =
                    (_optimizer as GradientBasedOptimizerBase<T, TInput, TOutput>)?.LearningRateScheduler;
                Exception? rank0Err = null;
                if (distBackend.Rank == 0)
                {
                    try
                    {
                        _optimizer.Deserialize(consolidatedOptBytes);
                        _checkpointManager.SaveCheckpoint(_model, _optimizer, epoch, stepInt, metrics, metadata);
                        _checkpointManager.UpdateAutoSaveState(stepInt, metricValue, minimizeMetric);
                    }
                    catch (Exception ex)
                    {
                        rank0Err = ex;
                    }
                }

                PropagateRank0CheckpointError(distBackend, rank0Err);

                distCoordinator.RestoreOptimizerStateFromConsolidatedBytes(
                    gbo,
                    trainableParams,
                    distBackend.Rank == 0 ? consolidatedOptBytes : null);

                if (liveScheduler is not null && _optimizer is GradientBasedOptimizerBase<T, TInput, TOutput> gboSched)
                {
                    gboSched.SetLearningRateScheduler(liveScheduler);
                }
            }
        }
        else
        {
            Exception? rank0Err = null;
            if (distBackend.Rank == 0)
            {
                try
                {
                    _checkpointManager.SaveCheckpoint(_model, _optimizer, epoch, stepInt, metrics, metadata);
                    _checkpointManager.UpdateAutoSaveState(stepInt, metricValue, minimizeMetric);
                }
                catch (Exception ex)
                {
                    rank0Err = ex;
                }
            }

            PropagateRank0CheckpointError(distBackend, rank0Err);
        }

        distBackend.Barrier();
        return true;
    }

    private static void PropagateRank0CheckpointError(
        DistributedTraining.ICommunicationBackend<T> distBackend,
        Exception? rank0Error)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var okVec = new Vector<T>(1);
        if (distBackend.Rank == 0)
        {
            okVec.AsWritableSpan()[0] = numOps.FromDouble(rank0Error is null ? 1.0 : 0.0);
        }

        okVec = distBackend.Broadcast(okVec, root: 0);
        if (numOps.ToDouble(okVec.AsSpan()[0]) <= 0.0)
        {
            if (rank0Error is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(rank0Error).Throw();
            }

            throw new IOException("Rank 0 failed while writing a distributed checkpoint.");
        }
    }

    /// <summary>
    /// Restores model state, optimizer state (including consolidated cross-world-size moment resharding and the
    /// learning-rate schedule), dynamic loss-scaler state, and the exact global sample cursor from the checkpoint
    /// manager's latest checkpoint. Returns <c>(0, 0, 0, 0)</c> when there is no checkpoint yet.
    /// </summary>
    private (long GlobalStep, int Epoch, int BatchInEpoch, long SamplesInEpoch) ResumeStreamingFromLatestCheckpoint(
        StreamingTrainingOptions<T, TInput, TOutput> options,
        IStreamingDataLoader<T, TInput, TOutput> rawLoader,
        DistributedTraining.ICommunicationBackend<T>? distBackend,
        DistributedTraining.DistributedTapeCoordinator<T>? distCoordinator)
    {
        if (_checkpointManager is null)
        {
            throw new InvalidOperationException(
                "StreamingTrainingOptions.ResumeFromLatestCheckpoint requires ConfigureCheckpointManager.");
        }

        if (_optimizer is null || _model is null)
        {
            throw new InvalidOperationException(
                "StreamingTrainingOptions.ResumeFromLatestCheckpoint requires ConfigureModel and ConfigureOptimizer.");
        }

        if (distBackend is null || distBackend.WorldSize <= 1)
        {
            return ResumeStreamingSingleRank(options, rawLoader, distCoordinator);
        }

        return ResumeStreamingDistributed(options, distBackend, distCoordinator);
    }

    private (long GlobalStep, int Epoch, int BatchInEpoch, long SamplesInEpoch) ResumeStreamingSingleRank(
        StreamingTrainingOptions<T, TInput, TOutput> options,
        IStreamingDataLoader<T, TInput, TOutput> rawLoader,
        DistributedTraining.DistributedTapeCoordinator<T>? distCoordinator)
    {
        var latest = _checkpointManager!.LoadLatestCheckpoint();
        if (latest is null)
        {
            return (0L, 0, 0, 0L);
        }

        var metadata = latest.Metadata;
        long globalStep = ReadStreamingCursor(metadata, StreamingGlobalStepKey, latest.CheckpointId);
        int epoch = checked((int)ReadStreamingCursor(metadata, StreamingEpochKey, latest.CheckpointId));
        int batchInEpoch = checked((int)ReadStreamingCursor(metadata, StreamingBatchInEpochKey, latest.CheckpointId));
        int savedBatchSize = checked((int)ReadStreamingCursor(metadata, StreamingBatchSizeKey, latest.CheckpointId));

        ValidateCheckpointSeed(metadata, options.Seed, latest.CheckpointId);

        long samplesInEpoch;
        if (metadata.TryGetValue(StreamingSamplesInEpochKey, out var samplesObj) && samplesObj is not null)
        {
            samplesInEpoch = Convert.ToInt64(samplesObj, CultureInfo.InvariantCulture);
        }
        else
        {
            if (savedBatchSize != rawLoader.BatchSize)
            {
                throw new InvalidOperationException(
                    $"Cannot resume from checkpoint '{latest.CheckpointId}': it was written with batch size " +
                    $"{savedBatchSize} but the loader now uses {rawLoader.BatchSize}.");
            }

            samplesInEpoch = (long)batchInEpoch * savedBatchSize;
        }

        if (_checkpointManager is not CheckpointManager<T, TInput, TOutput> typedManager)
        {
            throw new NotSupportedException(
                "Resuming requires CheckpointManager<T, TInput, TOutput>, which stores the model's own state " +
                "(ICheckpointableModel) in a sidecar that can be restored into the configured model.");
        }

        if (!typedManager.RestoreModelState(latest.CheckpointId, _model!))
        {
            throw new InvalidOperationException(
                $"Checkpoint '{latest.CheckpointId}' has no saved model state to resume from.");
        }

        if (_model is NeuralNetworks.NeuralNetworkBase<T> restoredNn)
        {
            restoredNn.InvalidateLayerWeightCaches();
            distCoordinator?.ResetParameterSync();
            distCoordinator?.EnsureInitialParametersSynchronized(
                restoredNn.CollectTrainableParameters(),
                restoredNn.InvalidateLayerWeightCaches);
        }

        ILearningRateScheduler? configuredScheduler =
            (_optimizer as GradientBasedOptimizerBase<T, TInput, TOutput>)?.LearningRateScheduler;

        if (distCoordinator is not null
            && _model is NeuralNetworks.NeuralNetworkBase<T> nnForOpt
            && _optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> gbo
            && latest.OptimizerData.Length > 0)
        {
            distCoordinator.RestoreOptimizerStateFromConsolidatedBytes(
                gbo,
                nnForOpt.CollectTrainableParameters(),
                latest.OptimizerData);
        }
        else
        {
            latest.RestoreOptimizer(_optimizer!);
        }

        ApplyConfiguredSchedulerAfterRestore(options, configuredScheduler, globalStep);

        bool hasMpScale = TryReadMpMetadata(metadata, out double mpScale, out int mpGrowthTracker);
        RestoreLossScalerState(_model, hasMpScale, mpScale, mpGrowthTracker);

        _checkpointManager.UpdateAutoSaveState(checked((int)globalStep));
        return (globalStep, epoch, batchInEpoch, samplesInEpoch);
    }

    private (long GlobalStep, int Epoch, int BatchInEpoch, long SamplesInEpoch) ResumeStreamingDistributed(
        StreamingTrainingOptions<T, TInput, TOutput> options,
        DistributedTraining.ICommunicationBackend<T> distBackend,
        DistributedTraining.DistributedTapeCoordinator<T>? distCoordinator)
    {
        int status = 0;
        long globalStep = 0L;
        int epoch = 0;
        int batchInEpoch = 0;
        long samplesInEpoch = 0L;
        bool hasMpScale = false;
        double mpScale = 1.0;
        int mpGrowthTracker = 0;
        byte[]? rank0StateBytes = null;
        byte[]? rank0OptimizerBytes = null;
        Exception? rank0Error = null;

        if (distBackend.Rank == 0)
        {
            try
            {
                var latest = _checkpointManager!.LoadLatestCheckpoint();
                if (latest is not null)
                {
                    var metadata = latest.Metadata;
                    globalStep = ReadStreamingCursor(metadata, StreamingGlobalStepKey, latest.CheckpointId);
                    epoch = checked((int)ReadStreamingCursor(metadata, StreamingEpochKey, latest.CheckpointId));
                    batchInEpoch = checked((int)ReadStreamingCursor(metadata, StreamingBatchInEpochKey, latest.CheckpointId));
                    int savedBatchSize = checked((int)ReadStreamingCursor(metadata, StreamingBatchSizeKey, latest.CheckpointId));

                    ValidateCheckpointSeed(metadata, options.Seed, latest.CheckpointId);

                    if (metadata.TryGetValue(StreamingSamplesInEpochKey, out var samplesObj) && samplesObj is not null)
                    {
                        samplesInEpoch = Convert.ToInt64(samplesObj, CultureInfo.InvariantCulture);
                    }
                    else
                    {
                        samplesInEpoch = (long)batchInEpoch * savedBatchSize;
                    }

                    if (_checkpointManager is not CheckpointManager<T, TInput, TOutput> typedManager)
                    {
                        throw new NotSupportedException(
                            "Resuming requires CheckpointManager<T, TInput, TOutput>, which stores the model's own state " +
                            "(ICheckpointableModel) in a sidecar that can be restored into the configured model.");
                    }

                    if (!typedManager.RestoreModelState(latest.CheckpointId, _model!))
                    {
                        throw new InvalidOperationException(
                            $"Checkpoint '{latest.CheckpointId}' has no saved model state to resume from.");
                    }

                    string stateSidecarPath = latest.FilePath is { Length: > 0 } existingPath
                        ? existingPath + ".state"
                        : Path.Combine(_checkpointManager.GetCheckpointDirectory(), $"checkpoint_{latest.CheckpointId}.json.state");
                    if (File.Exists(stateSidecarPath))
                    {
                        rank0StateBytes = File.ReadAllBytes(stateSidecarPath);
                    }

                    rank0OptimizerBytes = latest.OptimizerData;
                    hasMpScale = TryReadMpMetadata(metadata, out mpScale, out mpGrowthTracker);
                    status = 1;
                }
            }
            catch (Exception ex)
            {
                status = -1;
                rank0Error = ex;
            }
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var header = new Vector<T>(14);
        if (distBackend.Rank == 0)
        {
            Span<T> w = header.AsWritableSpan();
            w[0] = numOps.FromDouble(status);
            WriteLongAsU16Chunks(numOps, w, 1, globalStep);
            WriteIntAsU16Chunks(numOps, w, 5, epoch);
            WriteIntAsU16Chunks(numOps, w, 7, batchInEpoch);
            WriteLongAsU16Chunks(numOps, w, 9, samplesInEpoch);
            w[13] = numOps.FromDouble(hasMpScale ? 1.0 : 0.0);
        }

        header = distBackend.Broadcast(header, root: 0);
        {
            ReadOnlySpan<T> r = header.AsSpan();
            status = (int)Math.Round(numOps.ToDouble(r[0]));
            globalStep = ReadLongFromU16Chunks(numOps, r, 1);
            epoch = ReadIntFromU16Chunks(numOps, r, 5);
            batchInEpoch = ReadIntFromU16Chunks(numOps, r, 7);
            samplesInEpoch = ReadLongFromU16Chunks(numOps, r, 9);
            hasMpScale = numOps.ToDouble(r[13]) > 0.0;
        }

        if (status < 0)
        {
            if (rank0Error is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(rank0Error).Throw();
            }

            throw new InvalidOperationException("Distributed checkpoint resume failed validation on Rank 0.");
        }

        if (status == 0)
        {
            return (0L, 0, 0, 0L);
        }

        if (hasMpScale)
        {
            var mpVec = new Vector<T>(3);
            if (distBackend.Rank == 0)
            {
                Span<T> mpW = mpVec.AsWritableSpan();
                mpW[0] = numOps.FromDouble(mpScale);
                WriteIntAsU16Chunks(numOps, mpW, 1, mpGrowthTracker);
            }

            mpVec = distBackend.Broadcast(mpVec, root: 0);
            ReadOnlySpan<T> mpR = mpVec.AsSpan();
            mpScale = numOps.ToDouble(mpR[0]);
            mpGrowthTracker = ReadIntFromU16Chunks(numOps, mpR, 1);
        }

        byte[] sharedStateBytes = BroadcastByteArray(distBackend, distBackend.Rank == 0 ? rank0StateBytes : null, root: 0);
        if (sharedStateBytes.Length > 0 && _model is ICheckpointableModel checkpointableModel)
        {
            if (distBackend.Rank != 0)
            {
                using var ms = new MemoryStream(sharedStateBytes, writable: false);
                checkpointableModel.LoadState(ms);
            }
        }
        else
        {
            int paramLen = InterfaceGuard.Parameterizable(_model!).GetParameters().Length;
            Vector<T> fullParams = distBackend.Rank == 0
                ? InterfaceGuard.Parameterizable(_model!).GetParameters()
                : new Vector<T>(paramLen);
            fullParams = distBackend.Broadcast(fullParams, root: 0);
            if (distBackend.Rank != 0)
            {
                InterfaceGuard.Parameterizable(_model!).SetParameters(fullParams);
            }
        }

        if (_model is NeuralNetworks.NeuralNetworkBase<T> restoredNn)
        {
            restoredNn.InvalidateLayerWeightCaches();
            distCoordinator?.ResetParameterSync();
            distCoordinator?.EnsureInitialParametersSynchronized(
                restoredNn.CollectTrainableParameters(),
                restoredNn.InvalidateLayerWeightCaches);
        }

        ILearningRateScheduler? configuredScheduler =
            (_optimizer as GradientBasedOptimizerBase<T, TInput, TOutput>)?.LearningRateScheduler;

        if (distCoordinator is not null
            && _model is NeuralNetworks.NeuralNetworkBase<T> nnForOpt
            && _optimizer is IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> gbo)
        {
            distCoordinator.RestoreOptimizerStateFromConsolidatedBytes(
                gbo,
                nnForOpt.CollectTrainableParameters(),
                distBackend.Rank == 0 ? rank0OptimizerBytes : null);
        }
        else
        {
            byte[] optBytes = BroadcastByteArray(distBackend, distBackend.Rank == 0 ? rank0OptimizerBytes : null, root: 0);
            if (optBytes.Length > 0)
            {
                _optimizer!.Deserialize(optBytes);
            }
        }

        ApplyConfiguredSchedulerAfterRestore(options, configuredScheduler, globalStep);
        RestoreLossScalerState(_model, hasMpScale, mpScale, mpGrowthTracker);

        if (distBackend.Rank == 0)
        {
            _checkpointManager!.UpdateAutoSaveState(checked((int)globalStep));
        }

        distBackend.Barrier();
        return (globalStep, epoch, batchInEpoch, samplesInEpoch);
    }

    private void ApplyConfiguredSchedulerAfterRestore(
        StreamingTrainingOptions<T, TInput, TOutput> options,
        ILearningRateScheduler? configuredScheduler,
        long globalStep)
    {
        if (options.UseConfiguredSchedulerOnResume
            && configuredScheduler is not null
            && _optimizer is GradientBasedOptimizerBase<T, TInput, TOutput> gradientOptimizer)
        {
            int restoredSchedulerStep = gradientOptimizer.LearningRateScheduler is LearningRateSchedulerBase restored
                ? restored.CurrentStep
                : checked((int)globalStep);
            FastForwardScheduler(configuredScheduler, restoredSchedulerStep);
            gradientOptimizer.SetLearningRateScheduler(configuredScheduler);
        }
    }

    private static void ValidateCheckpointSeed(Dictionary<string, object> metadata, int? expectedSeed, string checkpointId)
    {
        if (!metadata.TryGetValue(StreamingSeedKey, out var savedSeedObj)
            || savedSeedObj is null
            || expectedSeed is null
            || Convert.ToInt32(savedSeedObj, CultureInfo.InvariantCulture) != expectedSeed.Value)
        {
            if (savedSeedObj is null && expectedSeed is null)
            {
                return;
            }

            throw new InvalidOperationException(
                $"Cannot resume from checkpoint '{checkpointId}': it was written with shuffle seed " +
                $"'{(savedSeedObj ?? "none")}' but this run uses '{(expectedSeed?.ToString(CultureInfo.InvariantCulture) ?? "none")}'. " +
                "A different seed changes the data order, so the skipped batches would not be the batches already trained on.");
        }
    }

    private static bool TryReadMpMetadata(Dictionary<string, object> metadata, out double scale, out int growthTracker)
    {
        scale = 1.0;
        growthTracker = 0;
        if (!metadata.TryGetValue(StreamingMpScaleKey, out var scaleObj) || scaleObj is null)
        {
            return false;
        }

        scale = Convert.ToDouble(scaleObj, CultureInfo.InvariantCulture);
        if (metadata.TryGetValue(StreamingMpGrowthTrackerKey, out var gtObj) && gtObj is not null)
        {
            growthTracker = Convert.ToInt32(gtObj, CultureInfo.InvariantCulture);
        }

        return true;
    }

    private static void WriteLongAsU16Chunks(INumericOperations<T> numOps, Span<T> dst, int offset, long value)
    {
        ulong u = unchecked((ulong)value);
        dst[offset] = numOps.FromDouble((ushort)(u & 0xFFFFUL));
        dst[offset + 1] = numOps.FromDouble((ushort)((u >> 16) & 0xFFFFUL));
        dst[offset + 2] = numOps.FromDouble((ushort)((u >> 32) & 0xFFFFUL));
        dst[offset + 3] = numOps.FromDouble((ushort)((u >> 48) & 0xFFFFUL));
    }

    private static long ReadLongFromU16Chunks(INumericOperations<T> numOps, ReadOnlySpan<T> src, int offset)
    {
        ulong c0 = (ulong)(ushort)Math.Round(numOps.ToDouble(src[offset]));
        ulong c1 = (ulong)(ushort)Math.Round(numOps.ToDouble(src[offset + 1]));
        ulong c2 = (ulong)(ushort)Math.Round(numOps.ToDouble(src[offset + 2]));
        ulong c3 = (ulong)(ushort)Math.Round(numOps.ToDouble(src[offset + 3]));
        return unchecked((long)(c0 | (c1 << 16) | (c2 << 32) | (c3 << 48)));
    }

    private static void WriteIntAsU16Chunks(INumericOperations<T> numOps, Span<T> dst, int offset, int value)
    {
        uint u = unchecked((uint)value);
        dst[offset] = numOps.FromDouble((ushort)(u & 0xFFFFU));
        dst[offset + 1] = numOps.FromDouble((ushort)((u >> 16) & 0xFFFFU));
    }

    private static int ReadIntFromU16Chunks(INumericOperations<T> numOps, ReadOnlySpan<T> src, int offset)
    {
        uint c0 = (uint)(ushort)Math.Round(numOps.ToDouble(src[offset]));
        uint c1 = (uint)(ushort)Math.Round(numOps.ToDouble(src[offset + 1]));
        return unchecked((int)(c0 | (c1 << 16)));
    }

    [AiDotNet.Attributes.ScalarKernel("One-time lazy layer input shape broadcast")]
    private static int[]? BroadcastIntShape(
        DistributedTraining.ICommunicationBackend<T> backend,
        int[]? rootShape,
        int root)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var lenVec = new Vector<T>(1);
        if (backend.Rank == root)
        {
            lenVec.AsWritableSpan()[0] = numOps.FromDouble(rootShape?.Length ?? 0);
        }

        lenVec = backend.Broadcast(lenVec, root);
        int ndim = (int)Math.Round(numOps.ToDouble(lenVec.AsSpan()[0]));
        if (ndim <= 0)
        {
            return null;
        }

        var shapeVec = new Vector<T>(ndim);
        if (backend.Rank == root && rootShape is not null)
        {
            Span<T> w = shapeVec.AsWritableSpan();
            for (int i = 0; i < ndim; i++)
            {
                w[i] = numOps.FromDouble(rootShape[i]);
            }
        }

        shapeVec = backend.Broadcast(shapeVec, root);
        var result = new int[ndim];
        ReadOnlySpan<T> r = shapeVec.AsSpan();
        for (int i = 0; i < ndim; i++)
        {
            result[i] = (int)Math.Round(numOps.ToDouble(r[i]));
        }

        return result;
    }

    [AiDotNet.Attributes.ScalarKernel("Checkpoint sidecar byte-array broadcast")]
    private static byte[] BroadcastByteArray(
        DistributedTraining.ICommunicationBackend<T> backend,
        byte[]? rootBytes,
        int root)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var lenHeader = new Vector<T>(2);
        if (backend.Rank == root)
        {
            WriteIntAsU16Chunks(numOps, lenHeader.AsWritableSpan(), 0, rootBytes?.Length ?? 0);
        }

        lenHeader = backend.Broadcast(lenHeader, root);
        int byteLen = ReadIntFromU16Chunks(numOps, lenHeader.AsSpan(), 0);
        if (byteLen <= 0)
        {
            return Array.Empty<byte>();
        }

        const int maxBytesPerChunk = 1024 * 1024;
        var output = backend.Rank == root && rootBytes is not null && rootBytes.Length == byteLen
            ? rootBytes
            : new byte[byteLen];

        int offset = 0;
        while (offset < byteLen)
        {
            int chunkLen = Math.Min(maxBytesPerChunk, byteLen - offset);
            var chunkVec = new Vector<T>(chunkLen);
            if (backend.Rank == root && rootBytes is not null)
            {
                Span<T> w = chunkVec.AsWritableSpan();
                for (int i = 0; i < chunkLen; i++)
                {
                    w[i] = numOps.FromDouble(rootBytes[offset + i]);
                }
            }

            chunkVec = backend.Broadcast(chunkVec, root);
            if (backend.Rank != root)
            {
                ReadOnlySpan<T> r = chunkVec.AsSpan();
                for (int i = 0; i < chunkLen; i++)
                {
                    output[offset + i] = (byte)Math.Round(numOps.ToDouble(r[i]));
                }
            }

            offset += chunkLen;
        }

        return output;
    }

    private static long ReadStreamingCursor(Dictionary<string, object> metadata, string key, string checkpointId)
    {
        if (!metadata.TryGetValue(key, out var value) || value is null)
        {
            throw new InvalidOperationException(
                $"Cannot resume from checkpoint '{checkpointId}': it has no '{key}' entry, so it was not written by " +
                "the streaming training loop.");
        }

        return Convert.ToInt64(value, CultureInfo.InvariantCulture);
    }

    private static void FastForwardScheduler(ILearningRateScheduler scheduler, int step)
    {
        if (scheduler is LearningRateSchedulerBase schedulerBase)
        {
            schedulerBase.LoadState(new Dictionary<string, object>
            {
                ["current_step"] = step,
                ["current_lr"] = schedulerBase.GetLearningRateAtStep(step),
            });
            return;
        }

        scheduler.Reset();
        for (int i = 0; i < step; i++)
        {
            scheduler.Step();
        }
    }

    private static double ConvertToDoubleScalar(T value)
    {
        return MathHelper.GetNumericOperations<T>().ToDouble(value);
    }

    private static T AddLossScalars(T a, T b)
    {
        return MathHelper.GetNumericOperations<T>().Add(a, b);
    }

    private static T ComputeRunningMeanLoss(T sum, int count)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return count > 0 ? numOps.Divide(sum, numOps.FromDouble(count)) : numOps.Zero;
    }

    private void AppendStreamingMicroBatches(
        TInput[] processedInputs,
        TOutput[] processedOutputs,
        List<(Tensor<T> Input, Tensor<T> Target)> microBatchBuffer)
    {
        if (processedInputs.Length == 0 || processedOutputs.Length == 0)
        {
            return;
        }

        var inputArray = processedInputs.Cast<Tensor<T>>().ToArray();
        var outputArray = processedOutputs.Cast<Tensor<T>>().ToArray();
        if (TryStackTensorBatch(inputArray, out var batchedInput)
            && TryStackTensorBatch(outputArray, out var batchedTarget)
            && batchedInput is not null
            && batchedTarget is not null)
        {
            microBatchBuffer.Add((batchedInput, batchedTarget));
            return;
        }

        for (int i = 0; i < inputArray.Length; i++)
        {
            microBatchBuffer.Add((inputArray[i], outputArray[i]));
        }
    }

    private static void ExecuteAccumulatedStreamingStep(
        NeuralNetworks.NeuralNetworkBase<T> neuralNet,
        List<(Tensor<T> Input, Tensor<T> Target)> microBatchBuffer,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? gboTrain)
    {
        neuralNet.TrainWithTapeAccumulated(microBatchBuffer, optimizer: gboTrain);
        microBatchBuffer.Clear();
    }

    private async Task<(T EpochLoss, int EpochBatches, long GlobalStep, T? LastValidationLoss, bool HasValidationLoss, long LastCheckpointStep, bool StepBudgetReached)>
        FlushRemainingAccumulatedMicroBatchesAsync(
            NeuralNetworks.NeuralNetworkBase<T> neuralNet,
            List<(Tensor<T> Input, Tensor<T> Target)> microBatchBuffer,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? gboTrain,
            int epoch,
            int batchInEpoch,
            long samplesInEpoch,
            T epochLoss,
            int epochBatches,
            long globalStep,
            T? lastValidationLoss,
            bool hasValidationLoss,
            long lastCheckpointStep,
            StreamingTrainingOptions<T, TInput, TOutput>? streamingOptions,
            string? streamingMonitorSessionId,
            IStreamingDataLoader<T, TInput, TOutput> rawLoader,
            Data.Loaders.DistributedStreamingDataLoader<T, TInput, TOutput>? distLoader,
            DistributedTraining.ICommunicationBackend<T>? distBackend,
            DistributedTraining.DistributedTapeCoordinator<T>? distCoordinator,
            int gradientAccumulationSteps)
    {
        ExecuteAccumulatedStreamingStep(neuralNet, microBatchBuffer, gboTrain);
        var lastLoss = neuralNet.GetLastLoss();
        epochLoss = AddLossScalars(epochLoss, lastLoss);
        epochBatches++;
        globalStep++;
        T runningEpochLoss = ComputeRunningMeanLoss(epochLoss, epochBatches);

        if (streamingOptions?.ValidationLoader is not null
            && streamingOptions.ValidateEveryNSteps > 0
            && globalStep % streamingOptions.ValidateEveryNSteps == 0)
        {
            lastValidationLoss = await EvaluateStreamingValidationAsync(
                streamingOptions, globalStep, streamingMonitorSessionId);
            hasValidationLoss = true;
        }

        if (TryCoordinatedStreamingCheckpoint(
            epoch,
            batchInEpoch,
            samplesInEpoch,
            globalStep,
            runningEpochLoss,
            hasValidationLoss,
            lastValidationLoss,
            streamingOptions,
            rawLoader,
            distLoader,
            distBackend,
            distCoordinator,
            gradientAccumulationSteps,
            forceSave: false))
        {
            lastCheckpointStep = globalStep;
        }

        bool stepBudgetReached = streamingOptions?.MaxSteps is long budget && globalStep >= budget;
        return (epochLoss, epochBatches, globalStep, lastValidationLoss, hasValidationLoss, lastCheckpointStep, stepBudgetReached);
    }

    private IFullModel<T, TInput, TOutput> WrapStreamingModelForDistributed(
        IFullModel<T, TInput, TOutput> trainedModel,
        DistributedTraining.ICommunicationBackend<T>? distBackend)
    {
        if (distBackend is null && _distributedConfiguration is null)
        {
            return trainedModel;
        }

        if (trainedModel is DistributedTraining.IShardedModel<T, TInput, TOutput> alreadySharded)
        {
            if (distBackend is { IsInitialized: true })
            {
                alreadySharded.GatherFullParameters();
            }

            return trainedModel;
        }

        var backend = distBackend ?? new DistributedTraining.InMemoryCommunicationBackend<T>(rank: 0, worldSize: 1);
        if (!backend.IsInitialized)
        {
            backend.Initialize();
        }

        var shardingConfig = _distributedConfiguration ?? new DistributedTraining.ShardingConfiguration<T>(backend);

        IFullModel<T, TInput, TOutput> wrapped = _distributedStrategy switch
        {
            DistributedStrategy.DDP or DistributedStrategy.None =>
                new DistributedTraining.DDPModel<T, TInput, TOutput>(trainedModel, shardingConfig),
            DistributedStrategy.FSDP =>
                new DistributedTraining.FSDPModel<T, TInput, TOutput>(trainedModel, shardingConfig),
            DistributedStrategy.ZeRO1 =>
                new DistributedTraining.ZeRO1Model<T, TInput, TOutput>(trainedModel, shardingConfig),
            DistributedStrategy.ZeRO2 =>
                new DistributedTraining.ZeRO2Model<T, TInput, TOutput>(trainedModel, shardingConfig),
            DistributedStrategy.ZeRO3 =>
                new DistributedTraining.ZeRO3Model<T, TInput, TOutput>(trainedModel, shardingConfig),
            DistributedStrategy.PipelineParallel =>
                new DistributedTraining.PipelineParallelModel<T, TInput, TOutput>(
                    trainedModel,
                    shardingConfig,
                    microBatchCount: _pipelineMicroBatchCount,
                    partitionStrategy: _pipelinePartitionStrategy,
                    schedule: _pipelineSchedule,
                    checkpointConfig: _pipelineCheckpointConfig),
            DistributedStrategy.TensorParallel =>
                new DistributedTraining.TensorParallelModel<T, TInput, TOutput>(trainedModel, shardingConfig),
            DistributedStrategy.Hybrid =>
                new DistributedTraining.HybridShardedModel<T, TInput, TOutput>(trainedModel, shardingConfig),
            _ => throw new InvalidOperationException($"Unsupported distributed strategy: {_distributedStrategy}")
        };

        if (wrapped is DistributedTraining.IShardedModel<T, TInput, TOutput> shardedModel)
        {
            // Populate LocalShard and CachedFullParameters while all ranks are still in lockstep so
            // post-BuildAsync single-rank Predict calls on FSDP/ZeRO3 never block waiting for peers.
            shardedModel.GatherFullParameters();
        }

        return wrapped;
    }

    private static T ComputeGlobalDistributedValidationMean(
        DistributedTraining.ICommunicationBackend<T> distBackend,
        double localWeightedLossSum,
        long localSampleCount)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var stats = new Vector<T>(2);
        {
            Span<T> w = stats.AsWritableSpan();
            w[0] = numOps.FromDouble(localWeightedLossSum);
            w[1] = numOps.FromDouble(localSampleCount);
        }

        distBackend.AllReduce(stats, DistributedTraining.ReductionOperation.Sum);

        ReadOnlySpan<T> r = stats.AsSpan();
        double globalWeightedLoss = numOps.ToDouble(r[0]);
        double globalSamples = numOps.ToDouble(r[1]);
        return globalSamples > 0.0 ? numOps.FromDouble(globalWeightedLoss / globalSamples) : numOps.Zero;
    }

    /// <summary>
    /// Mean loss of the model on the validation loader (unshuffled, so every evaluation sees the same batches),
    /// logged to the training monitor as <c>validation_loss</c> at <paramref name="globalStep"/>. In distributed
    /// mode (<c>WorldSize &gt; 1</c>), shards validation batches across ranks and reduces the sample-weighted global
    /// loss via <c>AllReduce(Sum)</c>. The model's training mode is restored afterwards.
    /// </summary>
    private async Task<T> EvaluateStreamingValidationAsync(
        StreamingTrainingOptions<T, TInput, TOutput> options,
        long globalStep,
        string? monitorSessionId)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var model = _model ?? throw new InvalidOperationException("Validation requires a configured model.");
        var rawLoader = options.ValidationLoader ?? throw new InvalidOperationException("No validation loader configured.");
        var distBackend = ResolveStreamingDistributedBackend();
        bool isDistributedVal = distBackend is { WorldSize: > 1 };
        IStreamingDataLoader<T, TInput, TOutput> loader = isDistributedVal
            && rawLoader is not Data.Loaders.DistributedStreamingDataLoader<T, TInput, TOutput>
                ? new Data.Loaders.DistributedStreamingDataLoader<T, TInput, TOutput>(rawLoader, distBackend!)
                : rawLoader;
        var lossFunction = model.DefaultLossFunction;
        var network = model as NeuralNetworks.NeuralNetworkBase<T>;
        bool wasTraining = network?.IsTrainingMode ?? false;

        if (!loader.IsLoaded)
        {
            await loader.LoadAsync();
        }

        T sum = numOps.Zero;
        int batches = 0;
        double localWeightedLossSum = 0.0;
        long localSampleCount = 0L;
        try
        {
            await foreach (var (inputs, outputs) in loader.GetBatchesAsync(shuffle: false))
            {
                if (options.ValidationMaxBatches is int maxBatches && batches >= maxBatches)
                {
                    break;
                }

                if (inputs.Length == 0)
                {
                    if (isDistributedVal)
                    {
                        batches++;
                    }

                    continue;
                }

                var processedInputs = new TInput[inputs.Length];
                var processedOutputs = new TOutput[outputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    processedInputs[i] = _preprocessingPipeline is { IsFitted: true } pipeline
                        && pipeline.Transform(inputs[i]) is TInput transformed ? transformed : inputs[i];
                    processedOutputs[i] = _targetPipeline is { IsFitted: true } targets
                        ? targets.Transform(outputs[i])
                        : outputs[i];
                }

                T batchLoss;
                if (processedInputs[0] is Tensor<T> && processedOutputs[0] is Tensor<T>
                    && TryStackTensorBatch(processedInputs.Cast<Tensor<T>>().ToArray(), out var stackedX)
                    && TryStackTensorBatch(processedOutputs.Cast<Tensor<T>>().ToArray(), out var stackedY)
                    && stackedX is TInput batchX && stackedY is TOutput batchY)
                {
                    batchLoss = SampleLoss(model, lossFunction, batchX, batchY);
                }
                else
                {
                    T sampleSum = numOps.Zero;
                    for (int i = 0; i < processedInputs.Length; i++)
                    {
                        sampleSum = numOps.Add(sampleSum, SampleLoss(model, lossFunction, processedInputs[i], processedOutputs[i]));
                    }

                    batchLoss = numOps.Divide(sampleSum, numOps.FromDouble(processedInputs.Length));
                }

                sum = numOps.Add(sum, batchLoss);
                batches++;
                if (isDistributedVal)
                {
                    localWeightedLossSum += ConvertToDoubleScalar(batchLoss) * inputs.Length;
                    localSampleCount += inputs.Length;
                }
            }
        }
        finally
        {
            if (network is not null && wasTraining)
            {
                network.SetTrainingMode(true);
            }
        }

        T mean = isDistributedVal
            ? ComputeGlobalDistributedValidationMean(distBackend!, localWeightedLossSum, localSampleCount)
            : (batches > 0 ? numOps.Divide(sum, numOps.FromDouble(batches)) : numOps.Zero);
        if (_trainingMonitor is not null && monitorSessionId is not null)
        {
            _trainingMonitor.LogMetric(monitorSessionId, "validation_loss", mean, checked((int)globalStep));
        }

        return mean;
    }

    /// <summary>
    /// The held-out loss of one batch: the objective training minimizes. A tensor model is trained on
    /// <see cref="ILossFunction{T}.ComputeTapeLoss"/> over the batch's shapes -- class-index targets
    /// (<c>[B, S, V]</c> logits with <c>[B, S]</c> token ids, as PyTorch's <c>cross_entropy</c>), a per-row
    /// mean -- so validation scores the same tensors the same way. Flattening both to vectors broke both:
    /// class-index targets are shorter than the logits (the vector loss refused them), and a batch of rows
    /// read as one vector is one distribution, not a mean of per-row losses.
    /// </summary>
    internal static T SampleLoss(IFullModel<T, TInput, TOutput> model, ILossFunction<T> lossFunction, TInput x, TOutput y)
    {
        // Only the host scalar leaves this method. On a GPU engine Predict keeps its output on the device and the loss
        // adds more device results ([rows, vocabulary] for an LM head: 805 MB at 4096 x 49152). Without a scope they
        // were freed only when a garbage collection finalized them, and small managed wrappers rarely trigger one:
        // 16 validation batches filled a 12 GB device and every later training step waited on buffer frees.
        // A scope frees every GPU result created inside it, so Predict runs in one only for a model that keeps none of
        // them for a later call; a generation cache or carried state would be freed under the model. A loss that keeps
        // no device result between calls (LossFunctionBase.KeepsResultsBetweenCalls) has results that belong to
        // validation alone, and they are freed. Any other loss may cache one (a basis built on its first call, or a
        // caller's loss nested in a composite), so it runs with every scope suspended: neither a loss scope nor the
        // enclosing predict scope can free what it keeps.
        var releasing = model as NeuralNetworks.NeuralNetworkBase<T> is { ReleasesPredictResults: true } n ? n : null;
        bool lossKeepsNothing = lossFunction is LossFunctions.LossFunctionBase<T> { KeepsResultsBetweenCalls: false };
        // A scope frees GPU results only; host arrays were left to the garbage collector. On the CPU Predict returns a
        // fresh copy of its output and the loss allocates buffers of the same size, outside any arena. For an LM head
        // those are [rows, vocabulary] large-object-heap arrays (402 MB at 2048 x 49152), collected only by a full GC:
        // a 400-step CPU run validating 8 batches every 5 steps held 22.5 GB of heap of which 0.36 GB was live. Under
        // the same condition (only the host scalar leaves this method), the forward and the loss run in one top-level
        // arena with no output copy, and its Dispose returns every buffer to the cross-arena pool for the next batch.
        // Only when no arena is open: a nested arena does not pool, and its per-layer boundaries become copies.
        using var arena = releasing is not null && lossKeepsNothing && x is Tensor<T> && y is Tensor<T>
            && typeof(TOutput) == typeof(Tensor<T>) && Tensors.Helpers.TensorArena.Current is null
            ? Tensors.Helpers.TensorArena.Create()
            : null;
        using var predictScope = releasing is not null ? Tensors.Engines.TensorScope.Begin() : null;
        var prediction = arena is not null
            ? (TOutput)(object)releasing!.PredictIntoCallerArena((Tensor<T>)(object)x!, arena)
            : model.Predict(x);
        if (prediction is Tensor<T> predicted && y is Tensor<T> target)
        {
            if (lossKeepsNothing)
            {
                using var lossScope = Tensors.Engines.TensorScope.Begin();
                return lossFunction.ComputeTapeLoss(predicted, target).Data.Span[0];
            }

            using (Tensors.Engines.TensorScope.Suspend())
                return lossFunction.ComputeTapeLoss(predicted, target).Data.Span[0];
        }

        return lossFunction.CalculateLoss(
            ConversionsHelper.ConvertToVector<T, TOutput>(prediction),
            ConversionsHelper.ConvertToVector<T, TOutput>(y));
    }
}
