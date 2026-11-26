using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet;

/// <content>
/// LoRA adaptation shared by the in-memory and the streaming supervised paths.
/// </content>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>
    /// Reads the first sample of a streaming loader as the LoRA warm-up input: preprocessed the way training
    /// batches are, with a leading batch dimension of 1.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The loader yields no sample, or the preprocessing pipeline is not fitted yet, so the shape the layers will
    /// train on is not known.
    /// </exception>
    private async Task<TInput> PeekStreamingLoRAWarmupSampleAsync(
        IStreamingDataLoader<T, TInput, TOutput> loader,
        CancellationToken cancellationToken)
    {
        if (_preprocessingPipeline is { IsFitted: false })
        {
            throw new InvalidOperationException(
                $"{nameof(ConfigureLoRA)} has to size a LoRA target layer that takes its shape from its first input, " +
                "and on the streaming path that input is a preprocessed sample; the preprocessing pipeline is fitted " +
                "on the first training batch, which comes later. Give the layer its sizes in its constructor, or " +
                "pass a preprocessing pipeline that is already fitted.");
        }

        if (!loader.IsLoaded)
        {
            await loader.LoadAsync(cancellationToken).ConfigureAwait(false);
        }

        await foreach (var (peekInputs, _) in loader.GetBatchesAsync(shuffle: false, cancellationToken: cancellationToken))
        {
            if (peekInputs.Length == 0)
            {
                continue;
            }

            TInput sample = _preprocessingPipeline is { IsFitted: true } fittedPipeline
                && fittedPipeline.Transform(peekInputs[0]) is TInput transformed
                    ? transformed
                    : peekInputs[0];
            return sample is Tensor<T> sampleTensor
                && TryStackTensorBatch(new[] { sampleTensor }, out var stacked)
                && stacked is TInput batched
                    ? batched
                    : sample;
        }

        throw new InvalidOperationException(
            $"{nameof(ConfigureLoRA)} needs one sample to size its target layers, and the streaming loader yielded none.");
    }

    /// <summary>
    /// Swaps the configured LoRA adapters into the network's layers before training, so that only the adapter
    /// weights train. Does nothing when <see cref="ConfigureLoRA"/> was not called or the model is not a neural
    /// network.
    /// </summary>
    /// <param name="warmupSampleFactory">
    /// Supplies one input with a leading batch dimension of 1. It is called only when a LoRA target layer cannot
    /// declare its shape from its constructor arguments and needs one forward pass to size itself.
    /// </param>
    private async Task ApplyConfiguredLoRAAdaptersAsync(Func<Task<TInput>> warmupSampleFactory)
    {
        if (_loraConfiguration is null || _model is not NeuralNetworks.NeuralNetworkBase<T> neuralNetForLoRA)
        {
            return;
        }

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
            // Fetched outside the best-effort block below: a caller that cannot supply a sample must fail the
            // build, not be logged and skipped like a warm-up forward that threw.
            TInput warmupProbe = await warmupSampleFactory().ConfigureAwait(false);
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
}
