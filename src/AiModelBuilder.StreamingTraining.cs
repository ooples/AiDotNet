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

        // Captured before the restore replaces it with the checkpoint's scheduler.
        ILearningRateScheduler? configuredScheduler =
            (_optimizer as GradientBasedOptimizerBase<T, TInput, TOutput>)?.LearningRateScheduler;

        latest.RestoreOptimizer(_optimizer);

        if (options.UseConfiguredSchedulerOnResume && configuredScheduler is not null
            && _optimizer is GradientBasedOptimizerBase<T, TInput, TOutput> gradientOptimizer)
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
