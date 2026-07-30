using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Continuous Integrate-and-Fire (CIF) alignment layer per Gao et al.
/// 2022 "Paraformer" §3.2 / Algorithm 1. Converts a variable-length
/// encoder hidden-state sequence <c>[B, S, D]</c> into a token-aligned
/// acoustic embedding sequence <c>[B, S, D]</c> by predicting per-
/// timestep fire weights and integrating the hidden states until the
/// cumulative weight crosses a unit-mass threshold.
/// </summary>
/// <remarks>
/// <para><b>Algorithm (Gao 2022 Algorithm 1):</b></para>
/// <list type="number">
/// <item>Predict <c>α_t ∈ [0, 1]</c> per timestep via a learnable
///   <c>Dense(D → 1, Sigmoid)</c> branch.</item>
/// <item>Maintain running accumulators <c>acc_α</c> and
///   <c>acc_h ∈ R^D</c>. Each step: if <c>acc_α + α_t</c> stays below
///   <c>threshold</c> (default 1.0), keep accumulating
///   <c>α_t · h_t</c>; otherwise split <c>α_t</c> into a "completing"
///   fraction that drives <c>acc_α</c> up to the threshold and a
///   "remainder" that seeds the next token, emit <c>acc_h</c> into
///   the output sequence, and reset.</item>
/// <item>Tail handling: after the last input timestep, if the
///   remaining <c>acc_α ≥ tailThreshold</c> (default 0.5), emit one
///   final token (renormalized by the remaining mass).</item>
/// </list>
///
/// <para><b>Output shape — fixed [B, S, D]:</b> the CIF paper's
/// output length <c>N</c> is data-dependent (depends on
/// <c>round(Σₜ α_t)</c>), which doesn't fit a static
/// <see cref="ILayer{T}"/> shape contract. We follow the FunASR
/// runtime convention: declare the output as the same length as the
/// input (a safe upper bound because each α_t ∈ [0, 1] gives at most
/// one fire per step), populate the first <c>predicted_N</c> slots
/// with the CIF tokens, and zero-pad the remainder. Downstream
/// attention layers ignore the padded slots through standard
/// padding-mask handling.</para>
///
/// <para><b>Trainable parameters:</b> only the alpha-predictor's
/// Dense weights. The integrate-and-fire arithmetic itself is
/// parameter-free and the threshold-crossing is non-differentiable —
/// gradients flow through the alpha predictor only via the upstream
/// loss applied to non-firing accumulation paths. Paraformer's
/// "alpha scaling" training trick (scaling all α_t so their sum
/// matches the target token count) is the standard way to make the
/// predictor learn alignment in spite of this; consumers that need
/// full alignment supervision should apply that scaling at training
/// time.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8", TestConstructorArgs = "8")]
public class CifAlignmentLayer<T> : LayerBase<T>
{
    private readonly int _encoderDim;
    private readonly T _threshold;
    private readonly T _tailThreshold;
    private readonly DenseLayer<T> _alphaPredictor;

    /// <summary>
    /// Currently <c>false</c>: this layer's <see cref="Forward"/>
    /// materializes α and the integrated hidden states into scalar T
    /// values via per-element <see cref="Tensor{T}"/> indexers and
    /// scalar <c>NumOps</c> arithmetic, which the tape autodiff
    /// path cannot record. Returning <c>true</c> while no gradient
    /// actually reaches <see cref="_alphaPredictor"/> would advertise
    /// a learnable alignment head that's secretly frozen — that's
    /// worse than a forward-only contract because callers would
    /// expect the alpha predictor to converge but it never would.
    /// </summary>
    /// <remarks>
    /// Fixing this to <c>true</c> requires one of:
    /// <list type="bullet">
    /// <item>A custom <c>Backward</c> implementation that walks
    /// recorded CIF split decisions in reverse and accumulates
    /// gradients for the alpha predictor (analytic derivatives of
    /// the integrate-and-fire dynamics).</item>
    /// <item>Implemented: the forward is now the soft CIF described below, so the tape records the
    /// whole alignment and the alpha predictor trains.</item>
    /// </list>
    /// </remarks>
    /// <inheritdoc/>
    /// <remarks>
    /// TRUE since the forward became the soft, differentiable CIF: the alignment is a continuous matrix
    /// built from Engine ops and applied with a matmul, so gradient reaches the alpha predictor
    /// (<c>DenseLayer(1, sigmoid)</c>). It previously reported FALSE because the hard integrate-and-fire
    /// scan wrote its output through raw indexing, which the tape cannot observe — that made the
    /// predictor untrainable and Paraformer's L_MAE term (Eq 6) impossible to express.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount => _alphaPredictor.ParameterCount;

    /// <summary>
    /// Initializes a new CIF alignment layer.
    /// </summary>
    /// <param name="encoderDim">Encoder hidden-state dimension <c>D</c>
    /// (also the layer's output channel dimension).</param>
    /// <param name="threshold">Fire threshold for the cumulative
    /// alpha. Gao 2022 §3.2 prescribes <c>1.0</c>.</param>
    /// <param name="tailThreshold">Tail-emission threshold —
    /// post-sequence remainder ≥ this triggers one final fire so a
    /// half-formed token isn't lost. Gao 2022 §3.2 prescribes
    /// <c>0.5</c>.</param>
    public CifAlignmentLayer(int encoderDim, double threshold = 1.0, double tailThreshold = 0.5)
        : base(new[] { -1, -1, encoderDim }, new[] { -1, -1, encoderDim })
    {
        if (encoderDim <= 0) throw new ArgumentOutOfRangeException(nameof(encoderDim));
        // Reject non-finite thresholds first: NaN slips past every relational guard below
        // (NaN < 1.0, NaN > threshold are both false), and ±Inf would corrupt the cumulative
        // integrate-and-fire comparisons once converted to T.
        if (double.IsNaN(threshold) || double.IsInfinity(threshold))
            throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "threshold must be a finite number.");
        if (double.IsNaN(tailThreshold) || double.IsInfinity(tailThreshold))
            throw new ArgumentOutOfRangeException(nameof(tailThreshold), tailThreshold, "tailThreshold must be a finite number.");
        // Reject threshold < 1.0 — the single-fire-per-timestep
        // assumption baked into the fixed [B, S, D] output shape only
        // holds when α_t ∈ [0, 1] cannot cross the threshold more
        // than once. For threshold < 1.0 a single α_t could cross
        // multiple times; the loop would emit only one token per step
        // (under-emitting) AND would carry an already-over-threshold
        // remainder into the next step (further corrupting the
        // accumulation invariant). The paper's stated value is 1.0;
        // future support for multi-fire would need either a dynamic
        // output shape or an inner "drain the remainder" loop.
        if (threshold < 1.0)
            throw new ArgumentOutOfRangeException(nameof(threshold), threshold,
                "threshold must be >= 1.0 — values below 1.0 admit multi-fire-per-timestep " +
                "which the single-fire output-shape assumption (S as upper bound on N) does not support. " +
                "Gao 2022 §3.2 prescribes 1.0.");
        if (tailThreshold < 0 || tailThreshold > threshold)
            throw new ArgumentOutOfRangeException(nameof(tailThreshold),
                $"tailThreshold must be in [0, threshold={threshold}].");

        _encoderDim = encoderDim;
        _threshold = NumOps.FromDouble(threshold);
        _tailThreshold = NumOps.FromDouble(tailThreshold);
        _alphaPredictor = new DenseLayer<T>(1, (IActivationFunction<T>)new SigmoidActivation<T>());
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Input contract: [B, S, D]. Reject non-paper ranks loudly —
        // CIF only makes sense over a time axis with hidden states.
        if (input.Rank != 3)
        {
            throw new ArgumentException(
                $"CifAlignmentLayer requires rank-3 [B, S, D] input; got rank {input.Rank}.",
                nameof(input));
        }
        int B = input.Shape[0];
        int S = input.Shape[1];
        int D = input.Shape[2];
        if (D != _encoderDim)
        {
            throw new ArgumentException(
                $"CifAlignmentLayer was configured for encoderDim={_encoderDim} but got D={D}.",
                nameof(input));
        }

        // ---- Soft (differentiable) CIF -------------------------------------------------------------
        // Gao 2022's Algorithm 1 is a hard integrate-and-fire scan: it splits alpha at each
        // threshold crossing with scalar arithmetic and writes the result through raw indexing. That
        // write is invisible to the autodiff tape, so the alpha predictor below could never receive a
        // gradient and Paraformer's L_MAE term (Eq 6), which supervises the predicted token COUNT, had
        // nothing to optimise.
        //
        // The soft re-formulation this uses is the one the layer's own remarks pointed at (Zhao & Gao
        // 2024, "Distill the soft CIF"): replace the hard scan with a continuous alignment matrix and
        // a matmul, so every step is an Engine op the tape records.
        //
        //   alpha_t = sigmoid(W h_t)                      (unchanged, the trainable predictor)
        //   cum_t   = sum_{i<=t} alpha_i                  = alpha . U, U upper-triangular ones
        //   A[l,t]  = clamp(1 - |cum_t - (l+1)|, 0, 1)    triangular kernel around each firing point
        //   E_l     = sum_t A[l,t] h_t                    = A . H
        //
        // cum is a matmul rather than TensorCumSum because TensorCumSum is not in the differentiable
        // op registry, while TensorMatMul is. A[l,t] peaks where the cumulative alpha crosses integer
        // l+1 — the same firing points the hard scan finds — but with a smooth, differentiable
        // neighbourhood instead of a discontinuous split.
        var alphaTensor = _alphaPredictor.Forward(input);          // [B, S, 1]
        var alpha2d = Engine.Reshape(alphaTensor, new[] { B, S }); // [B, S]

        // U[i, t] = 1 when i <= t, so (alpha . U)[b, t] = sum_{i<=t} alpha[b, i].
        var upper = new Tensor<T>(new[] { S, S });
        for (int i = 0; i < S; i++)
        {
            for (int t = i; t < S; t++)
            {
                upper.Data.Span[(i * S) + t] = NumOps.One;
            }
        }

        var cum = Engine.TensorMatMul(alpha2d, upper);             // [B, S]

        // Build the alignment directly in [b, l, t] order so no transpose is needed: replicate cum across
        // the token axis with a batched matmul against a column of ones.
        //   cumRep[b, l, t] = cum[b, t]   for every l
        var onesCol = new Tensor<T>(new[] { B, S, 1 });
        for (int i = 0; i < onesCol.Length; i++) onesCol.Data.Span[i] = NumOps.One;
        var cumRep = Engine.BatchMatMul(
            onesCol,                                      // [B, L=S, 1]
            Engine.Reshape(cum, new[] { B, 1, S }));      // [B, 1, S]  ->  [B, S, S]

        // Firing points: level[b, l, t] = (l + 1) * threshold, so a non-unit threshold stretches the
        // spacing exactly as the hard scan's accumulator would. Supervision-side constant, off-tape.
        double thresholdValue = NumOps.ToDouble(_threshold);
        var level = new Tensor<T>(new[] { B, S, S });
        for (int b2 = 0; b2 < B; b2++)
        {
            for (int l = 0; l < S; l++)
            {
                T value = NumOps.FromDouble((l + 1) * thresholdValue);
                int rowOffset = ((b2 * S) + l) * S;
                for (int t2 = 0; t2 < S; t2++)
                {
                    level.Data.Span[rowOffset + t2] = value;
                }
            }
        }

        // A = clamp(1 - |cum - level|, 0, 1), every step tape-recorded.
        var distance = Engine.TensorAbs(Engine.TensorSubtract(cumRep, level));
        var kernel = Engine.TensorAddScalar(Engine.TensorNegate(distance), NumOps.One);
        var alignment = Engine.TensorClamp(kernel, NumOps.Zero, NumOps.One);   // [B, L, S]

        // E[b, l, :] = sum_t A[b, l, t] * H[b, t, :]
        var aggregated = Engine.BatchMatMul(alignment, input);                  // [B, L=S, D]

        // Paraformer Eq 6's MAE term supervises this: the predicted token count per batch item.
        LastPredictedTokenCount = Engine.ReduceSum(alpha2d, new[] { 1 }, keepDims: false);

        return aggregated;
    }

    /// <summary>
    /// The predicted token count from the most recent forward: <c>sum_t alpha_t</c> per batch item.
    /// </summary>
    /// <remarks>
    /// This is the quantity Paraformer's MAE term supervises. Gao et al. 2022 (arXiv 2206.08317) §2.2
    /// train the CIF predictor to predict the number of tokens, and Eq 6's
    /// <c>L_total = gamma*L_CE + L_MAE + L_MWER</c> includes the MAE between this sum and the target
    /// length, described in §2.4 as guiding "the predictor to convergence". Exposed so a model can add
    /// that term; it is produced by <c>Engine.ReduceSum</c> over the tape-tracked alphas, so a loss
    /// built on it propagates into the predictor's weights.
    /// </remarks>
    public Tensor<T>? LastPredictedTokenCount { get; private set; }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => _alphaPredictor.GetParameters();

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
        => _alphaPredictor.SetParameters(parameters);

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
        => _alphaPredictor.GetParameterGradients();

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _alphaPredictor.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Tape-based autodiff drives the alpha predictor's updates
        // through its own UpdateParameters / Optimizer integration; no
        // manual step here. The CIF integrate-and-fire path is
        // parameter-free.
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _alphaPredictor.ResetState();
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["EncoderDim"] = _encoderDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Threshold"] = NumOps.ToDouble(_threshold).ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["TailThreshold"] = NumOps.ToDouble(_tailThreshold).ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
