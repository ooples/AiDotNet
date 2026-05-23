using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

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
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8", TestConstructorArgs = "8")]
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
    /// <item>A soft / differentiable CIF re-formulation (e.g.
    /// Zhao &amp; Gao 2024 "Distill the soft CIF") that replaces the
    /// hard threshold-crossing with a continuous accumulation matrix
    /// the tape can record through standard <c>Engine</c> ops.</item>
    /// </list>
    /// Tracked as the dedicated CIF-training follow-up — out of
    /// scope for the current Paraformer-shape-fix PR.
    /// </remarks>
    public override bool SupportsTraining => false;

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

        // Predict per-timestep fire weights via the Dense+Sigmoid head.
        // The alpha predictor is the layer's only trainable component;
        // we run it on the input *before* the CIF integrate-and-fire so
        // its gradient path (through the loss on aligned outputs) is
        // independent of the non-differentiable threshold crossing.
        var alphaTensor = _alphaPredictor.Forward(input);  // [B, S, 1]

        var output = new Tensor<T>(new[] { B, S, D });
        T thresh = _threshold;
        T tailThresh = _tailThreshold;

        // Per Gao 2022 Algorithm 1, executed per-batch independently:
        //   acc_α ← 0,  acc_h ← 0
        //   for t in 1..S:
        //     if acc_α + α_t >= θ:
        //       split α_t = α_t^c + α_t^r where α_t^c = θ − acc_α
        //       acc_h += α_t^c · h_t       // complete the current token
        //       emit acc_h                 // fire
        //       acc_α ← α_t^r,  acc_h ← α_t^r · h_t   // seed next
        //     else:
        //       acc_α += α_t,  acc_h += α_t · h_t
        //   if acc_α >= tail_θ:
        //     emit acc_h / acc_α          // renormalize partial token
        var accH = new T[D];
        for (int b = 0; b < B; b++)
        {
            T accAlpha = NumOps.Zero;
            for (int d = 0; d < D; d++) accH[d] = NumOps.Zero;
            int outIdx = 0;

            for (int t = 0; t < S && outIdx < S; t++)
            {
                T a = alphaTensor[b, t, 0];
                T proposedAcc = NumOps.Add(accAlpha, a);

                if (NumOps.GreaterThanOrEquals(proposedAcc, thresh))
                {
                    // Split alpha at the threshold-crossing.
                    T contribFraction = NumOps.Subtract(thresh, accAlpha);   // α_t^c
                    T remainderFraction = NumOps.Subtract(a, contribFraction); // α_t^r

                    // Complete the current token, emit it, then seed
                    // the next token with the remainder.
                    for (int d = 0; d < D; d++)
                    {
                        T h = input[b, t, d];
                        T completed = NumOps.Add(accH[d],
                            NumOps.Multiply(contribFraction, h));
                        output[b, outIdx, d] = completed;
                        accH[d] = NumOps.Multiply(remainderFraction, h);
                    }
                    accAlpha = remainderFraction;
                    outIdx++;
                }
                else
                {
                    // Standard accumulation step.
                    accAlpha = proposedAcc;
                    for (int d = 0; d < D; d++)
                    {
                        accH[d] = NumOps.Add(accH[d],
                            NumOps.Multiply(a, input[b, t, d]));
                    }
                }
            }

            // Tail emission per Gao 2022 §3.2 — a remainder above
            // tailThreshold gets renormalized into one final token so
            // the last partial fire isn't dropped on the floor.
            if (outIdx < S && NumOps.GreaterThanOrEquals(accAlpha, tailThresh))
            {
                T invAlpha = NumOps.GreaterThan(accAlpha, NumOps.Zero)
                    ? NumOps.Divide(NumOps.One, accAlpha)
                    : NumOps.Zero;
                for (int d = 0; d < D; d++)
                {
                    output[b, outIdx, d] = NumOps.Multiply(accH[d], invAlpha);
                }
                outIdx++;
            }

            // Remaining output slots [outIdx, S) stay zero — downstream
            // attention should mask them out via the standard padding-
            // mask path. Allocating with `new Tensor<T>(shape)`
            // zero-initializes by default(T), so nothing more to do.
        }

        return output;
    }

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
}
