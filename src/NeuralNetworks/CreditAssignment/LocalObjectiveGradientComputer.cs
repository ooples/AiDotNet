using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>Result of one local-objective gradient computation.</summary>
internal sealed class LocalObjectiveGradients<T>
{
    public LocalObjectiveGradients(Dictionary<Tensor<T>, Tensor<T>> gradients, Tensor<T> outputLoss, Tensor<T> prediction, int hiddenLayers, int localObjectives)
    {
        Gradients = gradients; OutputLoss = outputLoss; Prediction = prediction; HiddenLayers = hiddenLayers; LocalObjectives = localObjectives;
    }

    /// <summary>Gradient of every network parameter that received one (keyed by parameter tensor reference).</summary>
    public Dictionary<Tensor<T>, Tensor<T>> Gradients { get; }

    /// <summary>The scalar output objective (what the step reports as its loss).</summary>
    public Tensor<T> OutputLoss { get; }

    /// <summary>The readout's prediction for the step (or the pre-head hidden state when the terminal readout loss was fused).</summary>
    public Tensor<T> Prediction { get; }

    /// <summary>Hidden trainable layers offered a local objective.</summary>
    public int HiddenLayers { get; }

    /// <summary>Hidden layers whose rule actually returned a local objective.</summary>
    public int LocalObjectives { get; }
}

/// <summary>
/// Computes one training step's gradients for an <see cref="ILocalObjectiveCreditRule{T}"/>: a single forward in which
/// the input of every trainable pre-readout layer is detached, a combined objective
/// <c>output_objective(readout(h_last)) + Σ_k local_objective_k(h_k)</c>, and one backward. Because each hidden
/// layer's output depends only on its own parameters, the gradient a hidden layer receives is exactly the gradient of
/// its own local objective; the readout layers receive the output objective plus any non-detached local objective
/// routed through them.
/// </summary>
internal static class LocalObjectiveGradientComputer<T>
{
    public static LocalObjectiveGradients<T> Compute(
        NeuralNetworkBase<T> network,
        Tensor<T> input,
        Tensor<T> target,
        ILocalObjectiveCreditRule<T> rule,
        LossFunctionBase<T> loss,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> alignTarget)
    {
        var layers = network.Layers;
        int readoutCount = rule.ReadoutLayerCount;
        if (readoutCount < 1 || readoutCount >= layers.Count)
        {
            throw new InvalidOperationException(
                $"Credit rule '{rule.Name}' asked for {readoutCount} readout layers, but the network has {layers.Count} " +
                "layers; the readout must be at least one layer and leave at least one layer before it.");
        }
        int readoutStart = layers.Count - readoutCount;
        int headIndex = layers.Count - 1;

        var paramsOf = new List<Tensor<T>>[layers.Count];
        for (int i = 0; i < layers.Count; i++)
            paramsOf[i] = CollectParameters(layers[i]);

        var hiddenIdx = new List<int>();
        for (int i = 0; i < readoutStart; i++)
            if (paramsOf[i].Count > 0) hiddenIdx.Add(i);
        if (hiddenIdx.Count == 0)
            throw new InvalidOperationException($"Credit rule '{rule.Name}' found no trainable layer before the readout.");

        // Everything that outlives the tape (detached activations) must not come from the step arena; see the
        // matching note in CreditAssignmentGradientComputer.
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        bool prevTraining = network.IsTrainingMode;
        network.SetTrainingMode(true);
        try
        {
            using var tape = new GradientTape<T>();
            IEngine engine = AiDotNetEngine.Current;

            var hiddenOut = new Tensor<T>?[layers.Count];
            var current = input;
            for (int i = 0; i < readoutStart; i++)
            {
                if (paramsOf[i].Count > 0 && i > 0)
                    // stop-gradient: layer i's params see nothing from below. StopGradient is a device copy with no
                    // backward (torch's detach); the host round-trip it replaces downloaded and re-uploaded every
                    // hidden activation each step on a GPU engine.
                    current = engine.StopGradient(current);
                current = layers[i].Forward(current);
                if (paramsOf[i].Count > 0) hiddenOut[i] = current;
            }

            Tensor<T> PreHeadReadout(Tensor<T> h)
            {
                var r = h;
                for (int i = readoutStart; i < headIndex; i++) r = layers[i].Forward(r);
                return r;
            }

            Tensor<T> Readout(Tensor<T> h) => layers[headIndex].Forward(PreHeadReadout(h));

            Tensor<T> LossFn(Tensor<T> p, Tensor<T> t) => ReduceToScalar(engine, loss.ComputeTapeLoss(p, t));

            // Any readout whose terminal layer implements IFusedReadoutLossLayer<T> (even when preceded by RMSNorm or
            // LayerNorm, i.e. readoutCount >= 2) fuses the [rows, vocabulary] head + cross-entropy without putting a
            // [rows, vocabulary] logits tensor on the tape per layer.
            var fusedHead = layers[headIndex] is LayerBase<T> { IsInitialized: false }
                ? null
                : layers[headIndex] as IFusedReadoutLossLayer<T>;
            var detachableFusedHead = fusedHead as IDetachableFusedReadoutLossLayer<T>;

            Tensor<int>? preparedIds = null;
            int preparedVocabSize = 0;
            FusedReadoutPreparedState<T> headPreparedState = default;

            // Evaluate the terminal layer's pre-head activation once. Running PreHeadReadout(current) first also
            // initializes any lazy pre-head layer before we inspect its parameters and detach/batching capabilities.
            var preHeadFinal = PreHeadReadout(current);
            for (int i = readoutStart; i < headIndex; i++)
                paramsOf[i] = CollectParameters(layers[i]);

            bool preHeadAllDetachable = true;
            bool preHeadBatchIndependent = true;
            for (int i = readoutStart; i < headIndex; i++)
            {
                bool hasParamsOrLazy = paramsOf[i].Count > 0 || layers[i] is LayerBase<T> { IsInitialized: false };
                if (layers[i] is IDetachablePreHeadLayer<T> detachablePreHead)
                {
                    if (!detachablePreHead.CanForwardWithDetachedParameters && hasParamsOrLazy)
                        preHeadAllDetachable = false;
                    if (!detachablePreHead.PreservesIndependentStackedBatches(preHeadFinal.Rank))
                        preHeadBatchIndependent = false;
                }
                else
                {
                    if (hasParamsOrLazy)
                        preHeadAllDetachable = false;
                    preHeadBatchIndependent = false;
                }
            }

            var preHeadDetachedStates = preHeadAllDetachable && headIndex > readoutStart
                ? new object?[headIndex - readoutStart]
                : Array.Empty<object?>();

            Tensor<T> PreHeadReadoutDetached(Tensor<T> h)
            {
                var r = h;
                for (int i = readoutStart; i < headIndex; i++)
                {
                    if (layers[i] is IDetachablePreHeadLayer<T> { CanForwardWithDetachedParameters: true } detachable)
                    {
                        int slot = i - readoutStart;
                        r = detachable.ForwardWithDetachedParameters(r, ref preHeadDetachedStates[slot]);
                    }
                    else
                    {
                        r = layers[i].Forward(r);
                    }
                }
                return r;
            }

            // When the active credit rule does not inspect the materialized [rows, vocabulary] prediction tensor in
            // BuildOutputObjective, fuse the terminal output objective as well so zero [rows, vocabulary] tensors are
            // allocated across the entire training step.
            bool canTryFuseTerminal = fusedHead is not null
                && rule is IOutputPredictionOptionalRule { UsesOutputPrediction: false };

            Tensor<T> prediction;
            Tensor<T> alignedTarget;
            Tensor<T> defaultLoss;

            if (canTryFuseTerminal && detachableFusedHead is not null)
            {
                preparedIds = detachableFusedHead.TryPrepareClassIndexTarget(
                    target, loss, preHeadFinal.Rank, preHeadFinal.Shape.ToArray(), out preparedVocabSize);
                if (preparedIds is not null
                    && detachableFusedHead.TryComputeFusedLossPrepared(
                        preHeadFinal, preparedIds, detachParameters: false,
                        ref headPreparedState) is { } fusedOut)
                {
                    prediction = preHeadFinal;
                    alignedTarget = target;
                    defaultLoss = ReduceToScalar(engine, fusedOut);
                }
                else
                {
                    prediction = layers[headIndex].Forward(preHeadFinal);
                    alignedTarget = alignTarget(prediction, target);
                    defaultLoss = LossFn(prediction, alignedTarget);
                }
            }
            else if (canTryFuseTerminal && fusedHead!.TryComputeFusedLoss(preHeadFinal, target, loss) is { } genericFusedOut)
            {
                prediction = preHeadFinal;
                alignedTarget = target;
                defaultLoss = ReduceToScalar(engine, genericFusedOut);
            }
            else
            {
                prediction = layers[headIndex].Forward(preHeadFinal);
                alignedTarget = alignTarget(prediction, target);
                defaultLoss = LossFn(prediction, alignedTarget);
                if (detachableFusedHead is null && layers[headIndex] is not LayerBase<T> { IsInitialized: false })
                {
                    fusedHead = layers[headIndex] as IFusedReadoutLossLayer<T>;
                    detachableFusedHead = fusedHead as IDetachableFusedReadoutLossLayer<T>;
                }
                if (detachableFusedHead is not null)
                {
                    preparedIds = detachableFusedHead.TryPrepareClassIndexTarget(
                        alignedTarget, loss, preHeadFinal.Rank, preHeadFinal.Shape.ToArray(), out preparedVocabSize);
                }
            }
            paramsOf[headIndex] = CollectParameters(layers[headIndex]);

            Tensor<T> ReadoutLossCore(Tensor<T> h, Tensor<T> t)
            {
                var preHead = PreHeadReadout(h);
                if (ReferenceEquals(t, alignedTarget) && preparedIds is not null && detachableFusedHead is not null
                    && detachableFusedHead.TryComputeFusedLossPrepared(
                        preHead, preparedIds, detachParameters: false,
                        ref headPreparedState) is { } preparedFused)
                {
                    return ReduceToScalar(engine, preparedFused);
                }
                if (fusedHead?.TryComputeFusedLoss(preHead, t, loss) is { } fused)
                    return ReduceToScalar(engine, fused);
                return LossFn(layers[headIndex].Forward(preHead), t);
            }

            Tensor<T> ReadoutLossDetachedFn(Tensor<T> h, Tensor<T> t)
            {
                if (preHeadAllDetachable && ReferenceEquals(t, alignedTarget) && preparedIds is not null
                    && detachableFusedHead is not null)
                {
                    var preHead = PreHeadReadoutDetached(h);
                    if (detachableFusedHead.TryComputeFusedLossPrepared(
                        preHead, preparedIds, detachParameters: true,
                        ref headPreparedState) is { } detachedFused)
                    {
                        return ReduceToScalar(engine, detachedFused);
                    }
                }

                // Universal exact fallback for any arbitrary readout stack: compute d/dh[ReadoutLoss(h, t)] on an
                // isolated inner tape and attach a straight-through linear surrogate on the outer tape so outer-tape
                // readout parameters receive identically zero gradient from this hidden layer.
                var hDetached = engine.StopGradient(h);
                Tensor<T> lossVal;
                Tensor<T> gradH;
                using (var innerTape = new GradientTape<T>())
                {
                    lossVal = ReadoutLossCore(hDetached, t);
                    var innerGrads = innerTape.ComputeGradients(lossVal, sources: new[] { hDetached });
                    innerTape.Retain(lossVal);
                    if (innerGrads.TryGetValue(hDetached, out var gh))
                    {
                        innerTape.Retain(gh);
                        gradH = gh;
                    }
                    else
                    {
                        gradH = new Tensor<T>(h.Shape.ToArray());
                    }
                }
                var dot = ReduceToScalar(engine, engine.TensorMultiply(h, gradH));
                var dotConst = engine.StopGradient(dot);
                return engine.TensorAdd(engine.StopGradient(lossVal), engine.TensorSubtract(dot, dotConst));
            }

            // Per-step cache for sampled candidate vocabulary readout (built lazily at most once per step).
            int cachedSampleMax = -1;
            bool cachedSampleDetach = false;
            int cachedSampleSeed = 0;
            Tensor<int>? cachedRemappedIds = null;
            Tensor<T>? cachedSampledWeight = null;
            Tensor<T>? cachedSampledBias = null;

            bool TryEnsureSampledReadoutCache(int maxCandidates, bool detachParams, int stepSeed)
            {
                if (preparedIds is null || detachableFusedHead is null
                    || preparedVocabSize < 2 || maxCandidates >= preparedVocabSize)
                    return false;
                if (cachedSampledWeight is not null && cachedSampleMax == maxCandidates
                    && cachedSampleDetach == detachParams && cachedSampleSeed == stepSeed)
                    return true;
                if (!detachableFusedHead.TryGetVocabularyRowsForSampling(
                    out var vocabRows, out var headBias, out bool vocabRowsAreInputMajor))
                    return false;

                BuildSampledCandidates(preparedIds, preparedVocabSize, maxCandidates, stepSeed,
                    out var candidateIds, out var remappedIds);
                int u = candidateIds.Length;

                if (detachParams)
                {
                    using (new NoGradScope<T>())
                    {
                        if (vocabRowsAreInputMajor)
                        {
                            cachedSampledWeight = engine.StopGradient(
                                engine.TensorIndexSelect(vocabRows, candidateIds, axis: 1));
                        }
                        else
                        {
                            var pickedRows = engine.TensorEmbeddingLookup<T, int>(vocabRows, candidateIds);
                            cachedSampledWeight = engine.StopGradient(engine.TensorTranspose(pickedRows));
                        }
                        cachedSampledBias = headBias is not null
                            ? engine.StopGradient(engine.TensorIndexSelect(headBias, candidateIds, axis: 0))
                            : new Tensor<T>(new[] { u });
                    }
                }
                else
                {
                    if (vocabRowsAreInputMajor)
                    {
                        cachedSampledWeight = engine.TensorIndexSelect(vocabRows, candidateIds, axis: 1);
                    }
                    else
                    {
                        var pickedRows = engine.TensorEmbeddingLookup<T, int>(vocabRows, candidateIds);
                        cachedSampledWeight = engine.TensorTranspose(pickedRows);
                    }
                    cachedSampledBias = headBias is not null
                        ? engine.TensorIndexSelect(headBias, candidateIds, axis: 0)
                        : new Tensor<T>(new[] { u });
                }

                cachedRemappedIds = remappedIds;
                cachedSampleMax = maxCandidates;
                cachedSampleDetach = detachParams;
                cachedSampleSeed = stepSeed;
                return true;
            }

            Tensor<T>? TrySampledReadoutLossFn(Tensor<T> h, Tensor<T> t, int maxCandidates, bool detachParams, int stepSeed)
            {
                if (!ReferenceEquals(t, alignedTarget) || !TryEnsureSampledReadoutCache(maxCandidates, detachParams, stepSeed))
                    return null;

                int rows = cachedRemappedIds!.Length;
                int inputDim = cachedSampledWeight!.Shape[0];

                if (detachParams && !preHeadAllDetachable)
                {
                    // Preserve the exact sampled-vocabulary objective (without falling back to full readout) while
                    // respecting any custom/subclassed pre-head ForwardTraced override on an isolated inner tape.
                    var hDetached = engine.StopGradient(h);
                    Tensor<T>? innerLossVal = null;
                    Tensor<T>? gradH = null;
                    using (var innerTape = new GradientTape<T>())
                    {
                        var preHeadInner = PreHeadReadout(hDetached);
                        if (preHeadInner.Rank >= 2
                            && preHeadInner.Shape[preHeadInner.Rank - 1] == inputDim
                            && preHeadInner.Length == rows * inputDim)
                        {
                            var flatInner = preHeadInner.Rank == 2
                                ? preHeadInner
                                : engine.Reshape(preHeadInner, new[] { rows, inputDim });
                            DenseLayer<T>.CountFusedReadoutLoss();
                            innerLossVal = ReduceToScalar(
                                engine,
                                engine.FusedLinearCrossEntropyWithLogits(
                                    flatInner, cachedSampledWeight!, cachedSampledBias!, cachedRemappedIds!));
                            var innerGrads = innerTape.ComputeGradients(innerLossVal, sources: new[] { hDetached });
                            innerTape.Retain(innerLossVal);
                            if (innerGrads.TryGetValue(hDetached, out var gh))
                            {
                                innerTape.Retain(gh);
                                gradH = gh;
                            }
                            else
                            {
                                gradH = new Tensor<T>(h.Shape.ToArray());
                            }
                        }
                    }
                    if (innerLossVal is null || gradH is null)
                        return null;
                    var dot = ReduceToScalar(engine, engine.TensorMultiply(h, gradH));
                    var dotConst = engine.StopGradient(dot);
                    return engine.TensorAdd(engine.StopGradient(innerLossVal), engine.TensorSubtract(dot, dotConst));
                }

                var preHead = detachParams ? PreHeadReadoutDetached(h) : PreHeadReadout(h);
                if (preHead.Rank < 2 || preHead.Shape[preHead.Rank - 1] != inputDim || preHead.Length != rows * inputDim)
                    return null;

                var flatHidden = preHead.Rank == 2 ? preHead : engine.Reshape(preHead, new[] { rows, inputDim });
                DenseLayer<T>.CountFusedReadoutLoss();
                var sampledLoss = engine.FusedLinearCrossEntropyWithLogits(
                    flatHidden, cachedSampledWeight, cachedSampledBias!, cachedRemappedIds);
                return ReduceToScalar(engine, sampledLoss);
            }

            var step = new LocalObjectiveStep<T>(
                input, alignedTarget, engine, Readout, LossFn, hiddenIdx.Count,
                ReadoutLossCore, ReadoutLossDetachedFn, TrySampledReadoutLossFn);

            var outputObjective = rule.BuildOutputObjective(step, prediction, defaultLoss) ?? defaultLoss;
            var combined = outputObjective;

            int locals = 0;
            bool handledByBatchedReadout = false;

            // Depth-stacked single-GEMM readout fast path: instead of launching m separate memory-bound
            // [rows, D] x [D, V] (or [D, U]) GEMMs and allocating m separate [D, V] dWeight buffers, concatenate all
            // selected intermediate hidden states [h_k0; ...; h_k(m-1)] along axis 0 and evaluate the shared pre-head
            // normalization + fused cross-entropy in ONE batched GEMM.
            if (rule is IBatchedLocalReadoutRule batchedRule
                && preparedIds is not null
                && detachableFusedHead is not null)
            {
                var mode = batchedRule.BatchedReadoutMode;
                bool detach = batchedRule.DetachReadoutOnHiddenLayers;
                bool progressive = batchedRule.UseDepthProgressiveWeight;
                int stride = batchedRule.LayerStride;
                double baseWeight = batchedRule.BaseLocalWeight;

                if (CanBatchReadout(mode, preHeadBatchIndependent, detach, preHeadAllDetachable, progressive))
                {
                    int lastHidden = hiddenIdx.Count - 1;
                    var selectedHidden = new List<int>(Math.Max(0, lastHidden));
                    bool allSameShape = true;
                    for (int k = 0; k < lastHidden; k++)
                    {
                        if (stride > 1 && ((lastHidden - 1 - k) % stride) != 0)
                            continue;
                        var hCandidate = hiddenOut[hiddenIdx[k]];
                        if (hCandidate is null || !SameShape(hCandidate, current))
                        {
                            allSameShape = false;
                            break;
                        }
                        selectedHidden.Add(k);
                    }

                    if (allSameShape)
                    {
                        int m = selectedHidden.Count;
                        if (m == 0)
                        {
                            handledByBatchedReadout = true;
                        }
                        else
                        {
                            var stackedInputs = new Tensor<T>[m];
                            int hCount = hiddenIdx.Count;
                            var scales = progressive ? ProgressiveScales(selectedHidden, m, hCount, baseWeight) : null;
                            for (int j = 0; j < m; j++)
                            {
                                int k = selectedHidden[j];
                                var hk = hiddenOut[hiddenIdx[k]]!;
                                if (scales is not null)
                                {
                                    // detach == true is guaranteed here. Rescale backward Jacobian by w_j while
                                    // preserving bit-identical forward values tilde_h == hk (zero ULP drift).
                                    if (scales[j].IsIdentity)
                                    {
                                        stackedInputs[j] = hk;
                                    }
                                    else
                                    {
                                        var scaled = engine.TensorMultiplyScalar(hk, scales[j].Value);
                                        var delta = engine.TensorSubtract(scaled, engine.StopGradient(scaled));
                                        stackedInputs[j] = engine.TensorAdd(engine.StopGradient(hk), delta);
                                    }
                                }
                                else
                                {
                                    stackedInputs[j] = hk;
                                }
                            }

                            var hStacked = m == 1
                                ? stackedInputs[0]
                                : engine.TensorConcatenate(stackedInputs, axis: 0);
                            Tensor<T>? batchedLoss = null;

                            if (mode == BatchedLocalReadoutMode.SampledVocabulary)
                            {
                                int stepSeed = batchedRule.AdvanceAndGetStepSeed(input);
                                if (preparedVocabSize >= 2
                                    && batchedRule.SampledMaxCandidates < preparedVocabSize
                                    && TryEnsureSampledReadoutCache(batchedRule.SampledMaxCandidates, detach, stepSeed))
                                {
                                    var preHeadStacked = detach ? PreHeadReadoutDetached(hStacked) : PreHeadReadout(hStacked);
                                    int singleRows = cachedRemappedIds!.Length;
                                    int totalRows = m * singleRows;
                                    int inputDim = cachedSampledWeight!.Shape[0];
                                    if (preHeadStacked.Rank >= 2
                                        && preHeadStacked.Shape[preHeadStacked.Rank - 1] == inputDim
                                        && preHeadStacked.Length == totalRows * inputDim)
                                    {
                                        var flatStacked = preHeadStacked.Rank == 2
                                            ? preHeadStacked
                                            : engine.Reshape(preHeadStacked, new[] { totalRows, inputDim });
                                        var tiledRemapped = TileIntIds(cachedRemappedIds, m);
                                        DenseLayer<T>.CountFusedReadoutLoss();
                                        batchedLoss = ReduceToScalar(
                                            engine,
                                            engine.FusedLinearCrossEntropyWithLogits(
                                                flatStacked, cachedSampledWeight, cachedSampledBias!, tiledRemapped));
                                    }
                                }
                            }

                            if (batchedLoss is null)
                            {
                                var preHeadStacked = detach ? PreHeadReadoutDetached(hStacked) : PreHeadReadout(hStacked);
                                var tiledIds = TileIntIds(preparedIds, m);
                                if (detachableFusedHead.TryComputeFusedLossPrepared(
                                    preHeadStacked, tiledIds, detachParameters: detach,
                                    ref headPreparedState) is { } fusedBatched)
                                {
                                    batchedLoss = ReduceToScalar(engine, fusedBatched);
                                }
                            }

                            if (batchedLoss is not null)
                            {
                                double scale = progressive ? m : m * baseWeight;
                                var scaledBatched = BitConverter.DoubleToInt64Bits(scale) == BitConverter.DoubleToInt64Bits(1.0)
                                    ? batchedLoss
                                    : engine.TensorMultiplyScalar(batchedLoss, MathHelper.GetNumericOperations<T>().FromDouble(scale));
                                combined = engine.TensorAdd(combined, scaledBatched);
                                locals = m;
                                handledByBatchedReadout = true;
                            }
                        }
                    }
                }
            }

            if (!handledByBatchedReadout)
            {
                for (int k = 0; k < hiddenIdx.Count; k++)
                {
                    int i = hiddenIdx[k];
                    var local = rule.BuildLocalObjective(new LocalObjectiveContext<T>(step, k, i, layers[i],
                        hiddenOut[i] ?? throw new InvalidOperationException($"Hidden layer {i} produced no output.")));
                    if (local is null) continue;
                    combined = engine.TensorAdd(combined, ReduceToScalar(engine, local));
                    locals++;
                }
            }

            var sources = new List<Tensor<T>>();
            var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var ps in paramsOf)
                foreach (var p in ps)
                    if (seen.Add(p)) sources.Add(p);
            var aux = rule.AuxiliaryParameters;
            foreach (var p in aux)
                if (seen.Add(p)) sources.Add(p);

            var all = tape.ComputeGradients(combined, sources: sources);

            var grads = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
            var auxGrads = new Dictionary<Tensor<T>, Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
            var auxSet = new HashSet<Tensor<T>>(aux, TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var kv in all)
            {
                if (auxSet.Contains(kv.Key)) auxGrads[kv.Key] = kv.Value;
                else if (seen.Contains(kv.Key)) grads[kv.Key] = kv.Value;
            }
            rule.ApplyAuxiliaryGradients(auxGrads);

            // The output objective and prediction outlive this tape (the caller reads the loss value and the
            // prediction's shape). Only the loss handed to ComputeGradients is kept automatically; on a GPU engine the
            // rest are released with the tape, so reading the loss threw "GPU intermediate of a finished GradientTape".
            tape.Retain(outputObjective);
            tape.Retain(prediction);
            return new LocalObjectiveGradients<T>(grads, outputObjective, prediction, hiddenIdx.Count, locals);
        }
        finally
        {
            network.SetTrainingMode(prevTraining);
        }
    }

    internal static void BuildSampledCandidates(
        Tensor<int> targetIds,
        int vocabSize,
        int maxCandidates,
        int stepSeed,
        out Tensor<int> candidateIds,
        out Tensor<int> remappedIds)
    {
        int rows = targetIds.Length;
        var map = new Dictionary<int, int>(Math.Min(vocabSize, Math.Max(rows, maxCandidates)));
        var candidates = new List<int>(Math.Min(vocabSize, Math.Max(rows, maxCandidates)));
        remappedIds = new Tensor<int>(new[] { rows });

        for (int r = 0; r < rows; r++)
        {
            int id = targetIds.GetFlat(r);
            if (!map.TryGetValue(id, out int localIdx))
            {
                localIdx = candidates.Count;
                map[id] = localIdx;
                candidates.Add(id);
            }
            remappedIds.SetFlat(r, localIdx);
        }

        int targetCount = Math.Min(vocabSize, Math.Max(candidates.Count, maxCandidates));
        uint state = unchecked((uint)stepSeed * 2654435761u) ^ 0x9E3779B9u;
        if (state == 0u) state = 0x6D2B79F5u;

        // Fill remaining slots with uniform distinct negative vocabulary indices; fall back to linear probe if dense.
        int attempts = 0;
        int maxRandomAttempts = (targetCount - candidates.Count) * 4;
        while (candidates.Count < targetCount && attempts < maxRandomAttempts)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            int neg = (int)(state % (uint)vocabSize);
            attempts++;
            if (!map.ContainsKey(neg))
            {
                map[neg] = candidates.Count;
                candidates.Add(neg);
            }
        }

        if (candidates.Count < targetCount)
        {
            int start = (int)(state % (uint)vocabSize);
            for (int offset = 0; offset < vocabSize && candidates.Count < targetCount; offset++)
            {
                int neg = (start + offset) % vocabSize;
                if (!map.ContainsKey(neg))
                {
                    map[neg] = candidates.Count;
                    candidates.Add(neg);
                }
            }
        }

        candidateIds = new Tensor<int>(candidates.ToArray(), new[] { candidates.Count });
    }

    private static bool SameShape(Tensor<T> a, Tensor<T> b)
    {
        if (a.Rank != b.Rank) return false;
        for (int axis = 0; axis < a.Rank; axis++)
            if (a.Shape[axis] != b.Shape[axis]) return false;
        return true;
    }

    private static Tensor<int> TileIntIds(Tensor<int> ids, int count)
    {
        if (count == 1) return ids;
        int rows = ids.Length;
        var tiled = new int[count * rows];
        for (int r = 0; r < rows; r++)
            tiled[r] = ids.GetFlat(r);
        for (int c = 1; c < count; c++)
            Array.Copy(tiled, 0, tiled, c * rows, rows);
        return new Tensor<int>(tiled, new[] { count * rows });
    }

    private static bool CanBatchReadout(
        BatchedLocalReadoutMode mode,
        bool preHeadBatchIndependent,
        bool detach,
        bool preHeadAllDetachable,
        bool progressive)
    {
        if (mode == BatchedLocalReadoutMode.None || !preHeadBatchIndependent)
            return false;
        return detach ? preHeadAllDetachable : !progressive;
    }

    private static Tensor<T> ReduceToScalar(IEngine engine, Tensor<T> t)
    {
        if (t.Shape.Length == 0) return t;
        var axes = Enumerable.Range(0, t.Shape.Length).ToArray();
        return engine.ReduceSum(t, axes, keepDims: false);
    }

    internal static List<Tensor<T>> CollectParameters(ILayer<T> layer)
    {
        var result = new List<Tensor<T>>();
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        Collect(layer, result, seen);
        return result;

        static void Collect(ILayer<T> l, List<Tensor<T>> acc, HashSet<Tensor<T>> seen)
        {
            if (l is ITrainableLayer<T> t)
                foreach (var p in t.GetTrainableParameters())
                    if (p is not null && p.Length > 0 && seen.Add(p)) acc.Add(p);
            if (l is LayerBase<T> lb)
                foreach (var sub in lb.GetSubLayers())
                    Collect(sub, acc, seen);
        }
    }

    /// <summary>
    /// The progressive per-layer gradient scale w_j = base * ((k_j + 1) / hiddenCount)^2 of each selected hidden layer,
    /// with whether it is exactly 1 (the input is then used as-is, so its forward value is untouched).
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("One scale per selected hidden layer (a handful), not one per tensor element.")]
    private static (T Value, bool IsIdentity)[] ProgressiveScales(List<int> selectedHidden, int count, int hiddenCount,
        double baseWeight)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var scales = new (T Value, bool IsIdentity)[count];
        for (int j = 0; j < count; j++)
        {
            double depthFrac = (selectedHidden[j] + 1.0) / hiddenCount;
            double w = baseWeight * depthFrac * depthFrac;
            scales[j] = (numOps.FromDouble(w),
                BitConverter.DoubleToInt64Bits(w) == BitConverter.DoubleToInt64Bits(1.0));
        }
        return scales;
    }
}
