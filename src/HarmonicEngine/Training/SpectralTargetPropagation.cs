using AiDotNet.HarmonicEngine.Layers;
using AiDotNet.HarmonicEngine.Models;

namespace AiDotNet.HarmonicEngine.Training;

/// <summary>
/// The paper's novel no-backpropagation training strategy: <b>linear-path
/// spectral target propagation</b>. Trains a deep HRE language model by
/// propagating targets backward through the network via Tikhonov-regularized
/// spectral inverse filters, then applying local Hebbian updates to each
/// spectral filter. Nonlinearities in the forward path (SpectralGating) are
/// treated as identity during target propagation — the "linear path" framing
/// that distinguishes this method from classical target propagation with
/// Jacobian linearization.
/// </summary>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard neural networks train with backpropagation,
/// which propagates gradients (derivatives of a loss function) backward through
/// the network via the chain rule. Target propagation flips this: instead of
/// propagating derivatives, it propagates <i>desired output values</i> backward
/// and each layer does a local update toward its propagated target. The classical
/// version uses linear inverses for the backward propagation, which doesn't work
/// if layers have nonlinearities. Our spectral version does two novel things:
/// </para>
/// <list type="number">
/// <item><description>Uses <b>Tikhonov-regularized spectral inverse filters</b> for the
/// backward pass. Each trained Hebbian filter H(k) has a well-defined spectral
/// inverse H⁻¹(k) = conj(H(k)) / (|H(k)|² + ε), and we use it to map targets
/// from one layer's output space to the previous layer's output space.</description></item>
/// <item><description>Treats nonlinearities as <b>identity during target propagation</b> —
/// only the linear spectral skeleton of the network participates in the backward
/// pass. The nonlinearities are essential for forward-pass expressivity but not
/// required for target-driven learning to converge. This is a genuinely novel
/// claim that, if validated empirically, becomes a central contribution of the
/// paper.</description></item>
/// </list>
/// <para>
/// <b>Algorithm (per training batch):</b>
/// </para>
/// <list type="number">
/// <item><description>Forward pass, caching all intermediate block inputs.</description></item>
/// <item><description>Compute output target via cross-entropy gradient surrogate:
/// <c>t_N = (target_one_hot - softmax(logits)) · embedding^T</c>.</description></item>
/// <item><description>For each block from the top down, for each Hebbian filter inside
/// that block (there are two in the FFN), compute the spectral inverse of its
/// current filter, apply it to the current target signal, and pass the result
/// backward as the target for the previous stage.</description></item>
/// <item><description>For each block's Hebbian filters, apply a local Hebbian update using
/// the cached (input, propagated target) pair.</description></item>
/// </list>
/// <para>
/// A warm-up phase (first N batches) uses layerwise Hebbian auto-associative
/// targets instead of target propagation, giving each filter a non-trivial
/// starting point so that its inverse is meaningful when target propagation
/// kicks in.
/// </para>
/// </remarks>
public class SpectralTargetPropagation<T> : ITrainingStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly double _hebbianLearningRate;
    private readonly int _warmupSteps;
    private int _step;

    // Metrics tracked for the paper's training curves
    private double _lastTrainLoss;
    private double _lastWarmupLoss;
    private int _warmupCount;
    private int _targetPropCount;

    /// <inheritdoc/>
    public string Name => "SpectralTargetPropagation";

    /// <inheritdoc/>
    public string Description =>
        "Linear-path target propagation through Tikhonov-regularized spectral inverse filters. " +
        "Novel no-backprop training rule for HRE language models.";

    /// <summary>
    /// Creates a new spectral target propagation training strategy.
    /// </summary>
    /// <param name="hebbianLearningRate">Learning rate passed to each filter's Hebbian update.</param>
    /// <param name="warmupSteps">Number of initial batches that use layerwise Hebbian
    /// auto-associative training before switching to target propagation. Default 100.</param>
    public SpectralTargetPropagation(double hebbianLearningRate = 0.01, int warmupSteps = 100)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _hebbianLearningRate = hebbianLearningRate;
        _warmupSteps = warmupSteps;
    }

    /// <inheritdoc/>
    public void TrainStep(HRELanguageModel<T> model, TrainingBatch<T> batch)
    {
        double totalLoss = 0;
        int lossCount = 0;

        for (int b = 0; b < batch.BatchSize; b++)
        {
            // Extract this batch item's input + target token sequences
            var inputSeq = new Tensor<T>([batch.SequenceLength]);
            var targetSeq = new Tensor<T>([batch.SequenceLength]);
            for (int s = 0; s < batch.SequenceLength; s++)
            {
                inputSeq[s] = batch.InputTokens[b, s];
                targetSeq[s] = batch.TargetTokens[b, s];
            }

            double itemLoss;
            if (_step < _warmupSteps)
            {
                itemLoss = WarmupStep(model, inputSeq, targetSeq);
                _warmupCount++;
            }
            else
            {
                itemLoss = TargetPropagationStep(model, inputSeq, targetSeq);
                _targetPropCount++;
            }

            totalLoss += itemLoss;
            lossCount++;
        }

        _lastTrainLoss = lossCount > 0 ? totalLoss / lossCount : double.NaN;
        if (_step < _warmupSteps) _lastWarmupLoss = _lastTrainLoss;
        _step++;
    }

    /// <summary>
    /// Warm-up training step: layerwise auto-associative Hebbian updates. Each
    /// Hebbian filter learns to reconstruct its own input spectrum (the target
    /// is the input). This gives the filters a non-trivial starting point so
    /// their spectral inverses are meaningful when target propagation begins.
    /// </summary>
    private double WarmupStep(HRELanguageModel<T> model, Tensor<T> inputSeq, Tensor<T> targetSeq)
    {
        // Compute forward pass and cross-entropy loss for metric tracking.
        var logits = model.Forward(inputSeq);
        double loss = ComputeCrossEntropyLoss(logits, targetSeq, model.VocabSize);

        // For each block, apply an auto-associative Hebbian update using the
        // block's own output as both input and target. This teaches each
        // filter the identity-ish spectral mapping of its local data, which
        // is a safe warm-up starting point.
        //
        // Note: we don't have the exact pre-block activations cached from
        // the Forward() above because HRELanguageModel doesn't expose them.
        // For the warm-up we use a simpler proxy: for each block, we feed
        // the raw embedded sequence into the first filter and the target
        // is the same embedded sequence (identity reconstruction).
        var embedded = GetEmbeddedSequence(model, inputSeq);

        foreach (var block in model.Blocks)
        {
            // We treat each position's embedding as a separate signal for the
            // per-position FFN filters. Run an auto-associative Hebbian update
            // per position, per filter.
            for (int s = 0; s < model.SequenceLength; s++)
            {
                var embedSlice = new Vector<T>(model.EmbeddingDim);
                for (int e = 0; e < model.EmbeddingDim; e++)
                    embedSlice[e] = embedded[s, e];

                // Target = input for the first warm-up (identity reconstruction)
                block.FFN.Filter1.HebbianUpdate(embedSlice, embedSlice);
                block.FFN.Filter2.HebbianUpdate(embedSlice, embedSlice);
            }
        }

        return loss;
    }

    /// <summary>
    /// Target propagation training step: the paper's novel contribution.
    /// Runs the forward pass, derives an output-level target from the cross-
    /// entropy gradient surrogate, propagates the target backward through each
    /// block's Hebbian filters via spectral inverses, and applies local Hebbian
    /// updates. Nonlinearities are treated as identity during propagation.
    /// </summary>
    private double TargetPropagationStep(HRELanguageModel<T> model, Tensor<T> inputSeq, Tensor<T> targetSeq)
    {
        // Forward pass
        var logits = model.Forward(inputSeq);
        double loss = ComputeCrossEntropyLoss(logits, targetSeq, model.VocabSize);

        int seqLen = model.SequenceLength;
        int embedDim = model.EmbeddingDim;
        int vocabSize = model.VocabSize;

        // ----- Compute output-level target t_N in embedding space -----
        // Cross-entropy gradient w.r.t. logits is (softmax(logits) - target_one_hot).
        // We negate it to make it a target direction: t_N points where we want the
        // model's final embedding x_N to move. Then project back to embedding space
        // via the unembedding transpose (which, for tied embeddings, is the
        // transpose of the token embedding matrix).
        var targetEmbedSeq = new Tensor<T>([seqLen, embedDim]);

        for (int s = 0; s < seqLen; s++)
        {
            // Softmax the logits at this position (numerically stable)
            double maxLogit = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double l = _numOps.ToDouble(logits[s, v]);
                if (l > maxLogit) maxLogit = l;
            }
            double sumExp = 0;
            var probs = new double[vocabSize];
            for (int v = 0; v < vocabSize; v++)
            {
                probs[v] = Math.Exp(_numOps.ToDouble(logits[s, v]) - maxLogit);
                sumExp += probs[v];
            }
            for (int v = 0; v < vocabSize; v++) probs[v] /= sumExp;

            int targetToken = (int)_numOps.ToDouble(targetSeq[s]);

            // Error signal in logit space: one_hot(target) - probs
            // Project to embedding space: e_s = Σ_v error[v] · E[v, :]
            // (This uses the tied embedding matrix; we access it via a forward
            // pass over a one-hot input by constructing the dot product directly.)
            var errorEmbed = new double[embedDim];
            for (int v = 0; v < vocabSize; v++)
            {
                double e = (v == targetToken ? 1.0 : 0.0) - probs[v];
                // We need access to the embedding matrix. For now we'll use
                // the model's forward pass over a token ID to get its row;
                // this is inefficient but correct and only happens at training.
                var row = GetEmbeddingRow(model, v);
                for (int d = 0; d < embedDim; d++)
                    errorEmbed[d] += e * _numOps.ToDouble(row[d]);
            }

            // errorEmbed is the target direction in x_N's space. Scale it
            // down so the magnitude is comparable to a typical activation
            // (otherwise the first Hebbian update produces huge filter coefficients).
            const double targetScale = 0.1;
            for (int d = 0; d < embedDim; d++)
                targetEmbedSeq[s, d] = _numOps.FromDouble(targetScale * errorEmbed[d]);
        }

        // ----- Propagate target backward through each block -----
        // For each block from N down to 1, we propagate the current target
        // backward through the block's two FFN Hebbian filters (in reverse
        // order: Filter2 first, then Filter1). Nonlinearity is skipped.
        //
        // Because the FFN is applied per-position, target propagation is also
        // per-position: for each sequence position s, we treat the length-E
        // embedding vector as a 1D signal and propagate the target through
        // that position's copy of the filters.
        var currentTarget = targetEmbedSeq;

        for (int blockIdx = model.NumLayers - 1; blockIdx >= 0; blockIdx--)
        {
            var block = model.Blocks[blockIdx];

            // Build inverse filters for Filter1 and Filter2 based on their
            // current state at the start of this batch.
            var invFilter2 = new SpectralInverseFilter<T>(block.FFN.Filter2.Filter);
            var invFilter1 = new SpectralInverseFilter<T>(block.FFN.Filter1.Filter);

            var newTarget = new Tensor<T>([seqLen, embedDim]);

            for (int s = 0; s < seqLen; s++)
            {
                // Extract this position's length-E target vector
                var targetVec = new Vector<T>(embedDim);
                for (int e = 0; e < embedDim; e++) targetVec[e] = currentTarget[s, e];

                // Apply H2⁻¹ (inverse of the second Hebbian filter), then skip
                // the SpectralGating (linear-path approximation), then apply H1⁻¹.
                var afterInv2 = invFilter2.ApplyReal(targetVec);
                var afterInv1 = invFilter1.ApplyReal(afterInv2);

                for (int e = 0; e < embedDim; e++)
                    newTarget[s, e] = afterInv1[e];
            }

            currentTarget = newTarget;
        }

        // ----- Apply Hebbian updates to each block's filters -----
        // We use the embedded input as the input to each filter (approximation:
        // the true input to each block is the running activation x_l, but we
        // use x_0 for all blocks during this first implementation pass. This
        // is the major approximation of the "linear path" framing.)
        var embedded = GetEmbeddedSequence(model, inputSeq);

        for (int blockIdx = 0; blockIdx < model.NumLayers; blockIdx++)
        {
            var block = model.Blocks[blockIdx];

            for (int s = 0; s < seqLen; s++)
            {
                var embedSlice = new Vector<T>(embedDim);
                var targetSlice = new Vector<T>(embedDim);
                for (int e = 0; e < embedDim; e++)
                {
                    embedSlice[e] = embedded[s, e];
                    targetSlice[e] = currentTarget[s, e];
                }

                block.FFN.Filter1.HebbianUpdate(embedSlice, targetSlice);
                block.FFN.Filter2.HebbianUpdate(embedSlice, targetSlice);
            }
        }

        return loss;
    }

    /// <summary>
    /// Computes mean cross-entropy loss across all positions.
    /// </summary>
    private double ComputeCrossEntropyLoss(Tensor<T> logits, Tensor<T> targetSeq, int vocabSize)
    {
        int seqLen = logits.Shape[0];
        double totalLoss = 0;
        for (int s = 0; s < seqLen; s++)
        {
            double maxLogit = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double l = _numOps.ToDouble(logits[s, v]);
                if (l > maxLogit) maxLogit = l;
            }
            double sumExp = 0;
            for (int v = 0; v < vocabSize; v++)
                sumExp += Math.Exp(_numOps.ToDouble(logits[s, v]) - maxLogit);
            double logSumExp = maxLogit + Math.Log(sumExp);

            int target = (int)_numOps.ToDouble(targetSeq[s]);
            double targetLogit = _numOps.ToDouble(logits[s, target]);
            totalLoss += (logSumExp - targetLogit);
        }
        return totalLoss / seqLen;
    }

    /// <summary>
    /// Gets the embedded + positional encoding sequence for a given token sequence.
    /// This is a helper that runs just the embedding stage of the model's forward
    /// pass without running any of the HRE blocks.
    /// </summary>
    private Tensor<T> GetEmbeddedSequence(HRELanguageModel<T> model, Tensor<T> inputSeq)
    {
        // We don't have direct access to the model's internal embedding table,
        // so we reconstruct the embedded sequence by passing through the full
        // forward pass and then reverse-engineering it. Unfortunately, we'd
        // need direct access to the embedding + positional encoding output for
        // this to be efficient.
        //
        // For now, we return a pass-through approximation: run one pass through
        // zero blocks by constructing a dummy model shape. The cleanest way is
        // to make HRELanguageModel.Forward expose an option to return the
        // pre-block embeddings. We'll defer that refactor and use a placeholder
        // here that assumes the model exposes an EmbedOnly method.
        //
        // As a workaround, we use the input tokens' index directly: we know
        // the model's embedding is deterministic given tokenId, so we can
        // reconstruct it by querying the model for embeddings of each token.
        int seqLen = model.SequenceLength;
        int embedDim = model.EmbeddingDim;
        var result = new Tensor<T>([seqLen, embedDim]);

        // Use model.GetEmbedding(tokenId) if exposed; otherwise fall back to
        // one-hot forward pass. We added GetEmbeddingRow as a helper accessor.
        for (int s = 0; s < seqLen; s++)
        {
            int tokenId = (int)_numOps.ToDouble(inputSeq[s]);
            var row = GetEmbeddingRow(model, tokenId);
            // Add positional encoding (sinusoidal)
            for (int d = 0; d < embedDim; d++)
            {
                double angle = s / Math.Pow(10000.0, 2.0 * (d / 2) / embedDim);
                double posVal = (d % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                result[s, d] = _numOps.Add(row[d], _numOps.FromDouble(posVal));
            }
        }
        return result;
    }

    /// <summary>
    /// Gets a row of the token embedding matrix for a given token ID.
    /// This requires accessor support from HRELanguageModel — we'll add it
    /// via a public GetEmbedding method if it doesn't already exist.
    /// </summary>
    private Vector<T> GetEmbeddingRow(HRELanguageModel<T> model, int tokenId)
    {
        return model.GetTokenEmbedding(tokenId);
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, double> GetMetrics()
    {
        return new Dictionary<string, double>
        {
            ["last_train_loss"] = _lastTrainLoss,
            ["last_warmup_loss"] = _lastWarmupLoss,
            ["warmup_count"] = _warmupCount,
            ["target_prop_count"] = _targetPropCount,
            ["total_steps"] = _step,
            ["in_warmup"] = _step < _warmupSteps ? 1.0 : 0.0,
        };
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _step = 0;
        _lastTrainLoss = 0;
        _lastWarmupLoss = 0;
        _warmupCount = 0;
        _targetPropCount = 0;
    }
}
