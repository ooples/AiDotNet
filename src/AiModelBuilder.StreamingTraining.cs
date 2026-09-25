using AiDotNet.CheckpointManagement;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <summary>
/// Step-level streaming training support (<see cref="ConfigureStreamingTraining"/>): resume, periodic held-out
/// validation, and the checkpoint cursor that makes a resumed run consume exactly the batches an uninterrupted run
/// would have.
/// </summary>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    internal const string StreamingEpochKey = "streaming.epoch";
    internal const string StreamingBatchInEpochKey = "streaming.batch_in_epoch";
    internal const string StreamingGlobalStepKey = "streaming.global_step";
    internal const string StreamingSeedKey = "streaming.seed";
    internal const string StreamingBatchSizeKey = "streaming.batch_size";

    private static Dictionary<string, T> StreamingCheckpointMetrics(T trainLoss, bool hasValidationLoss, T? validationLoss)
    {
        var metrics = new Dictionary<string, T> { ["loss"] = trainLoss };
        if (hasValidationLoss && validationLoss is not null)
        {
            metrics["validation_loss"] = validationLoss;
        }

        return metrics;
    }

    private static Dictionary<string, object> StreamingCheckpointMetadata(
        int epoch,
        int batchInEpoch,
        long globalStep,
        StreamingTrainingOptions<T, TInput, TOutput>? options,
        IStreamingDataLoader<T, TInput, TOutput> loader)
    {
        var metadata = new Dictionary<string, object>
        {
            [StreamingEpochKey] = epoch,
            [StreamingBatchInEpochKey] = batchInEpoch,
            [StreamingGlobalStepKey] = globalStep,
            [StreamingBatchSizeKey] = loader.BatchSize,
        };
        if (options?.Seed is int seed)
        {
            metadata[StreamingSeedKey] = seed;
        }

        return metadata;
    }

    /// <summary>
    /// Restores model state, optimizer state (including the learning-rate schedule) and the data cursor from the
    /// checkpoint manager's latest checkpoint. Returns (0, 0, 0) when there is no checkpoint yet, so the same
    /// configuration both starts and resumes a run.
    /// </summary>
    private (long GlobalStep, int Epoch, int BatchInEpoch) ResumeStreamingFromLatestCheckpoint(
        StreamingTrainingOptions<T, TInput, TOutput> options,
        IStreamingDataLoader<T, TInput, TOutput> loader)
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

        var latest = _checkpointManager.LoadLatestCheckpoint();
        if (latest is null)
        {
            return (0, 0, 0);
        }

        var metadata = latest.Metadata;
        long globalStep = ReadStreamingCursor(metadata, StreamingGlobalStepKey, latest.CheckpointId);
        int epoch = checked((int)ReadStreamingCursor(metadata, StreamingEpochKey, latest.CheckpointId));
        int batchInEpoch = checked((int)ReadStreamingCursor(metadata, StreamingBatchInEpochKey, latest.CheckpointId));
        int savedBatchSize = checked((int)ReadStreamingCursor(metadata, StreamingBatchSizeKey, latest.CheckpointId));

        if (!metadata.TryGetValue(StreamingSeedKey, out var savedSeedObj)
            || Convert.ToInt32(savedSeedObj) != options.Seed)
        {
            throw new InvalidOperationException(
                $"Cannot resume from checkpoint '{latest.CheckpointId}': it was written with shuffle seed " +
                $"'{(savedSeedObj ?? "none")}' but this run uses '{options.Seed}'. A different seed changes the data " +
                "order, so the skipped batches would not be the batches already trained on.");
        }

        if (savedBatchSize != loader.BatchSize)
        {
            throw new InvalidOperationException(
                $"Cannot resume from checkpoint '{latest.CheckpointId}': it was written with batch size " +
                $"{savedBatchSize} but the loader now uses {loader.BatchSize}.");
        }

        if (_checkpointManager is not CheckpointManager<T, TInput, TOutput> typedManager)
        {
            throw new NotSupportedException(
                "Resuming requires CheckpointManager<T, TInput, TOutput>, which stores the model's own state " +
                "(ICheckpointableModel) in a sidecar that can be restored into the configured model.");
        }

        if (!typedManager.RestoreModelState(latest.CheckpointId, _model))
        {
            throw new InvalidOperationException(
                $"Checkpoint '{latest.CheckpointId}' has no saved model state to resume from.");
        }

        var gradientOptimizer = _optimizer as GradientBasedOptimizerBase<T, TInput, TOutput>;
        ILearningRateScheduler? configuredScheduler = gradientOptimizer?.LearningRateScheduler;

        latest.RestoreOptimizer(_optimizer);

        if (options.UseConfiguredSchedulerOnResume && configuredScheduler is not null && gradientOptimizer is not null)
        {
            int restoredSchedulerStep = gradientOptimizer.LearningRateScheduler is LearningRateSchedulerBase restored
                ? restored.CurrentStep
                : checked((int)globalStep);
            FastForwardScheduler(configuredScheduler, restoredSchedulerStep);
            gradientOptimizer.SetLearningRateScheduler(configuredScheduler);
        }

        _checkpointManager.UpdateAutoSaveState(checked((int)globalStep));
        return (globalStep, epoch, batchInEpoch);
    }

    private static long ReadStreamingCursor(Dictionary<string, object> metadata, string key, string checkpointId)
    {
        if (!metadata.TryGetValue(key, out var value) || value is null)
        {
            throw new InvalidOperationException(
                $"Cannot resume from checkpoint '{checkpointId}': it has no '{key}' entry, so it was not written by " +
                "the streaming training loop.");
        }

        return Convert.ToInt64(value);
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

    /// <summary>
    /// Mean loss of the model on the validation loader (unshuffled, so every evaluation sees the same batches),
    /// logged to the training monitor as <c>validation_loss</c> at <paramref name="globalStep"/>. The model's
    /// training mode is restored afterwards.
    /// </summary>
    private async Task<T> EvaluateStreamingValidationAsync(
        StreamingTrainingOptions<T, TInput, TOutput> options,
        long globalStep,
        string? monitorSessionId)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var model = _model ?? throw new InvalidOperationException("Validation requires a configured model.");
        var loader = options.ValidationLoader ?? throw new InvalidOperationException("No validation loader configured.");
        var lossFunction = model.DefaultLossFunction;
        var network = model as NeuralNetworks.NeuralNetworkBase<T>;
        bool wasTraining = network?.IsTrainingMode ?? false;

        if (!loader.IsLoaded)
        {
            await loader.LoadAsync();
        }

        T sum = numOps.Zero;
        int batches = 0;
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
            }
        }
        finally
        {
            if (network is not null && wasTraining)
            {
                network.SetTrainingMode(true);
            }
        }

        T mean = batches > 0 ? numOps.Divide(sum, numOps.FromDouble(batches)) : numOps.Zero;
        if (_trainingMonitor is not null && monitorSessionId is not null)
        {
            _trainingMonitor.LogMetric(monitorSessionId, "validation_loss", mean, checked((int)globalStep));
        }

        return mean;
    }

    private static T SampleLoss(IFullModel<T, TInput, TOutput> model, ILossFunction<T> lossFunction, TInput x, TOutput y)
    {
        var prediction = model.Predict(x);
        return lossFunction.CalculateLoss(
            ConversionsHelper.ConvertToVector<T, TOutput>(prediction),
            ConversionsHelper.ConvertToVector<T, TOutput>(y));
    }
}
