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
    /// <c>true</c>: the alpha predictor is genuinely trained.
    /// </summary>
    /// <remarks>
    /// <para>This was previously <c>false</c> because <see cref="Forward"/> materialized α and
    /// the integrated hidden states into scalar <c>T</c> values through per-element indexers and
    /// <c>NumOps</c> arithmetic, which the gradient tape cannot record — so the alignment head
    /// was frozen at initialization.</para>
    /// <para><see cref="Forward"/> now takes the second route this remark used to prescribe: a
    /// continuous accumulation matrix recorded through standard <c>Engine</c> ops. The
    /// threshold crossings still decide the firing structure in a non-differentiable pass, but
    /// every emitted token is then formed as
    /// <c>out = (A ⊙ αB + C) · h</c> via <c>Engine.TensorMultiply</c>,
    /// <c>Engine.TensorAdd</c> and <c>Engine.TensorBatchMatMul</c>, so the tape records the whole
    /// contraction and gradients reach both <see cref="_alphaPredictor"/> and the encoder
    /// states automatically.</para>
    /// <para>One deliberate truncation remains: the completing fraction <c>α_t^c = θ − acc</c>
    /// enters as a constant, so the dependence of a fire on the ENTIRE preceding alpha sequence
    /// is not carried. Each frame's own alpha — the dominant term — is exact.</para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount => _alphaPredictor.ParameterCount;

    /// <summary>
    /// Whether the paper's training-time alpha scaling is applied. Defaults to <c>true</c>.
    /// </summary>
    /// <remarks>
    /// Dong &amp; Xu 2020 §3.2 scale the weights by <c>S~ / Σα</c> during training so the number
    /// of integrated embeddings is forced to match the target token count. Scaling only happens
    /// when <see cref="TargetTokenCount"/> is set and the layer is in training mode; inference is
    /// unaffected, matching the paper.
    /// </remarks>
    public bool AlphaScalingEnabled { get; set; }

    /// <summary>
    /// Weight <c>λ₂</c> on the quantity loss. Defaults to the paper's <c>1.0</c>.
    /// </summary>
    public double QuantityLossWeight { get; set; }

    /// <summary>
    /// Target token count <c>S~</c> for the current batch, set by the consumer before a training
    /// forward pass. <c>null</c> (the default) disables both alpha scaling and the quantity loss.
    /// </summary>
    /// <remarks>
    /// CIF's alignment supervision needs the label length, which a layer cannot infer from its
    /// input. Models that train CIF end-to-end should assign this from the target sequence before
    /// calling Forward, then read <see cref="LastQuantityLoss"/> and add
    /// <c>QuantityLossWeight * LastQuantityLoss</c> to their objective.
    /// </remarks>
    public int? TargetTokenCount { get; set; }

    /// <summary>
    /// The most recent <c>|Σα − S~|</c>, averaged over the batch, or zero when
    /// <see cref="TargetTokenCount"/> is unset. Multiply by <see cref="QuantityLossWeight"/> and
    /// add to the training objective.
    /// </summary>
    public T LastQuantityLoss { get; private set; } = MathHelper.GetNumericOperations<T>().Zero;

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
    /// <param name="alphaScalingEnabled">
    /// Whether to apply the paper's training-time alpha scaling. Dong &amp; Xu 2020 §3.2 multiply
    /// every weight by <c>S~ / Σα</c> so the integrated count matches the target token count.
    /// Defaults to <c>true</c> (the paper's strategy); requires <see cref="TargetTokenCount"/>.
    /// </param>
    /// <param name="quantityLossWeight">
    /// Weight <c>λ₂</c> on the quantity loss <c>|Σα − S~|</c>. Dong &amp; Xu 2020 use
    /// <c>1.0</c>, which is the default. Set to <c>0</c> to disable the term.
    /// </param>
    public CifAlignmentLayer(
        int encoderDim,
        double threshold = 1.0,
        double tailThreshold = 0.5,
        bool alphaScalingEnabled = true,
        double quantityLossWeight = 1.0)
        : base(new[] { -1, -1, encoderDim }, new[] { -1, -1, encoderDim })
    {
        AlphaScalingEnabled = alphaScalingEnabled;
        QuantityLossWeight = quantityLossWeight;
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
        // Input is the concatenated [h_{u-1} | h_u | h_{u+1}] window (3 x encoderDim), per
        // Dong & Xu 2020 — see the note in Forward on why the paper's conv1d + FC collapses
        // to a single affine map over that window.
        _alphaPredictor = new DenseLayer<T>(1, (IActivationFunction<T>)new SigmoidActivation<T>());

        // Register the predictor as a CHILD layer so recursive parameter discovery finds its
        // weights — the equivalent of PyTorch's nn.Module child registration.
        //
        // Without this, GetTrainableParameters() returned an EMPTY set while GetParameters()
        // reported 49 elements, because DenseLayer allocates its weights lazily on first Forward
        // and nothing ever registered them with the base layer. That mismatch was harmless while
        // SupportsTraining was false (the engine ignored the layer), but the moment the layer
        // became trainable it desynchronized the flat parameter vector from the tensor set the
        // tape and ParameterBuffer actually track — producing "Parameter[0] is NaN after
        // training" in every CIF consumer (SenseVoiceLarge, Paraformer, CIFEncoder).
        RegisterSubLayer(_alphaPredictor);
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
        //
        // Dong & Xu 2020 (arXiv:1905.11235) compute α_u from a WINDOW centred on h_u —
        // "pass a window centered at h_u (e.g. [h_{u-1}, h_u, h_{u+1}]) to a 1-dimensional
        // convolutional layer and then a fully connected layer with one output unit and a
        // sigmoid activation". This previously fed h_u alone, so α could not see its
        // neighbours — and a firing boundary is defined by the CHANGE between adjacent
        // frames, which is precisely the information a single-frame predictor cannot access.
        //
        // The window is materialized as the concatenation [h_{u-1} | h_u | h_{u+1}] with
        // zero padding at the sequence edges. A width-3 conv1d followed by a 1-unit FC, with
        // no activation between them, composes to a single affine map of that 3D-wide window,
        // so one Dense(1, sigmoid) over the concatenation is mathematically equivalent to the
        // paper's conv+FC pair while keeping this layer's parameter/gradient plumbing intact.
        var windowed = BuildAlphaWindow(input, B, S, D);
        var alphaTensor = _alphaPredictor.Forward(windowed);  // [B, S, 1]

        // A constant ones column, reused below for both the alpha sum and the broadcasts. Every
        // combination with it goes through Engine ops so the tape keeps tracking alpha.
        var onesColumn = new Tensor<T>(new[] { B, S, 1 });
        for (int i = 0; i < onesColumn.Length; i++) onesColumn[i] = NumOps.One;

        if (TargetTokenCount is int targetCount && targetCount > 0)
        {
            // Σα per batch as a tape-visible reduction: [B, 1, S] x [B, S, 1] = [B, 1, 1].
            // Computed ONLY when a target length is supplied. Building it unconditionally left a
            // tape node whose result never reached the output on the inference path, which is
            // both wasted work and a dangling contribution to gradient accumulation.
            var alphaRowForSum = Engine.Reshape(alphaTensor, [B, 1, S]);
            var alphaSum = Engine.TensorBatchMatMul(alphaRowForSum, onesColumn);   // [B, 1, 1]

            var target = new Tensor<T>(new[] { B, 1, 1 });
            for (int i = 0; i < target.Length; i++) target[i] = NumOps.FromDouble(targetCount);

            // Quantity loss |Σα − S~| (Dong & Xu 2020 §3.2, weighted by λ₂ = 1.0 by default).
            // Built from Engine ops so a consumer that folds it into the objective keeps a live
            // gradient path back to the alpha predictor.
            var quantityLoss = Engine.TensorAbs(Engine.TensorSubtract(alphaSum, target));
            T lossSum = NumOps.Zero;
            for (int i = 0; i < quantityLoss.Length; i++) lossSum = NumOps.Add(lossSum, quantityLoss[i]);
            LastQuantityLoss = B > 0 ? NumOps.Divide(lossSum, NumOps.FromDouble(B)) : NumOps.Zero;

            // Alpha scaling: multiply every weight by S~/Σα so the integrated count matches the
            // target. Training-time only, exactly as the paper specifies — at inference the
            // target length is unknown and the raw alphas decide the output length.
            if (AlphaScalingEnabled && IsTrainingMode)
            {
                // Guard the denominator. Sigma-alpha is a sum of non-negative firing weights, so
                // it approaches zero whenever the weight predictor's outputs collapse — which is
                // easy early in training, before the quantity loss has pulled the integrated
                // count toward the target. Dividing by it then produces an enormous ratio that
                // scales every weight, and d(1/u)/du = -1/u^2 makes the backward pass worse than
                // the forward.
                //
                // This is a latent hazard rather than an observed failure: the enclosing branch
                // only runs when TargetTokenCount is set, which the generated fixtures never do,
                // so it is not the cause of CIFEncoder's NaN.
                var denominatorFloor = new Tensor<T>(alphaSum.Shape.ToArray());
                for (int i = 0; i < denominatorFloor.Length; i++)
                    denominatorFloor[i] = NumOps.FromDouble(1e-6);

                var ratio = Engine.TensorDivide(
                    target,
                    Engine.TensorAdd(alphaSum, denominatorFloor));                 // [B, 1, 1]
                var ratioBroadcast = Engine.TensorBatchMatMul(onesColumn, ratio);  // [B, S, 1]
                alphaTensor = Engine.TensorMultiply(alphaTensor, ratioBroadcast);
            }
        }
        else
        {
            LastQuantityLoss = NumOps.Zero;
        }

        T thresh = _threshold;
        T tailThresh = _tailThreshold;

        // The integrate-and-fire output is a WEIGHTED SUM of the encoder states:
        //     out[b, j, :] = Σ_t W[b, j, t] · h[b, t, :]
        // and every weight is affine in that frame's own alpha:
        //     W[b, j, t] = A[b, j, t] · α_t + C[b, j, t]
        // The threshold crossings decide WHICH (j, t) pairs exist and with what A/C — that
        // decision is non-differentiable and is taken below from the alpha VALUES. But given
        // those decisions the output is an ordinary product, so building W with Engine ops and
        // contracting with a batched matmul lets the gradient tape record the whole thing
        // automatically: gradients reach the alpha predictor through A ⊙ αB, and the encoder
        // states through the matmul. This is the "continuous accumulation matrix ... standard
        // Engine ops" route this layer's own remarks prescribe, and it is what makes
        // SupportsTraining true rather than an advertised-but-frozen head.
        //
        // A and C are constants w.r.t. the tape; only αB carries alpha's gradient.
        var coeff = Zeroed(new[] { B, S, S });             // A — multiplies α_t
        var prefixCoeff = Zeroed(new[] { B, S, S });        // Bp — multiplies P_t = Σ_{s<t} α_s
        var constant = Zeroed(new[] { B, S, S });           // C — the fixed part

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
        for (int b = 0; b < B; b++)
        {
            T accAlpha = NumOps.Zero;
            int outIdx = 0;
            int fireCount = 0;

            // Frames feeding the token currently being accumulated, so the tail emission can
            // renormalize exactly the weights that formed it.
            var pending = new List<int>(S);

            for (int t = 0; t < S && outIdx < S; t++)
            {
                T a = alphaTensor[b, t, 0];
                T proposedAcc = NumOps.Add(accAlpha, a);

                if (NumOps.GreaterThanOrEquals(proposedAcc, thresh))
                {
                    // Split alpha at the threshold-crossing.
                    T contribFraction = NumOps.Subtract(thresh, accAlpha);   // α_t^c
                    T remainderFraction = NumOps.Subtract(a, contribFraction); // α_t^r

                    // α_t^c = θ − acc completes THIS token. It is a function of the PRIOR
                    // alphas — and that dependence is now carried exactly rather than dropped.
                    //
                    // The accumulator obeys acc_t = acc_{t-1} + α_t − (θ if fired) in BOTH
                    // branches, so it telescopes:
                    //     acc_{t-1} = P_t − F·θ,   P_t = Σ_{s<t} α_s,   F = fires before t
                    // which makes the split an exact affine function of the alphas:
                    //     α_t^c = θ(1 + F) − P_t
                    //     α_t^r = α_t − α_t^c = α_t + P_t − θ(1 + F)
                    // Expressing it through the prefix sum is what lets the tape see a fire's
                    // dependence on the ENTIRE preceding alpha sequence, instead of the earlier
                    // truncation that held α_t^c constant.
                    T thresholdTerm = NumOps.Multiply(thresh, NumOps.FromDouble(fireCount + 1));

                    // Completing fraction closes THIS token: no α_t term, −1 on the prefix sum.
                    constant[b, outIdx, t] = NumOps.Add(constant[b, outIdx, t], thresholdTerm);
                    prefixCoeff[b, outIdx, t] = NumOps.Subtract(prefixCoeff[b, outIdx, t], NumOps.One);

                    // Remainder seeds the NEXT token: +1 on α_t and +1 on the prefix sum.
                    if (outIdx + 1 < S)
                    {
                        coeff[b, outIdx + 1, t] = NumOps.Add(coeff[b, outIdx + 1, t], NumOps.One);
                        prefixCoeff[b, outIdx + 1, t] = NumOps.Add(prefixCoeff[b, outIdx + 1, t], NumOps.One);
                        constant[b, outIdx + 1, t] = NumOps.Subtract(constant[b, outIdx + 1, t], thresholdTerm);
                    }

                    accAlpha = remainderFraction;
                    fireCount++;
                    outIdx++;
                    pending.Clear();
                    pending.Add(t);
                }
                else
                {
                    // Standard accumulation step: the whole α_t lands on the current token.
                    accAlpha = proposedAcc;
                    coeff[b, outIdx, t] = NumOps.Add(coeff[b, outIdx, t], NumOps.One);
                    pending.Add(t);
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

                // Scale the weights already accumulated into this slot by 1/acc_α. The 1/acc_α
                // factor itself is held constant w.r.t. the tape — carrying its derivative would
                // couple the tail to every alpha that formed it.
                foreach (int t in pending)
                {
                    coeff[b, outIdx, t] = NumOps.Multiply(coeff[b, outIdx, t], invAlpha);
                    prefixCoeff[b, outIdx, t] = NumOps.Multiply(prefixCoeff[b, outIdx, t], invAlpha);
                    constant[b, outIdx, t] = NumOps.Multiply(constant[b, outIdx, t], invAlpha);
                }
                outIdx++;
            }

            // Remaining output slots [outIdx, S) stay zero — downstream
            // attention should mask them out via the standard padding-
            // mask path. Rows of A and C for those slots are all zero, so the
            // contraction below emits zeros there. Nothing more to do.
        }

        // Inclusive-minus-self prefix sum P_t = Σ_{s<t} α_s. A cumulative sum along the sequence
        // axis is O(S); the strictly-lower-triangular matmul this replaces built a [B, S, S]
        // operand and contracted it just to add up earlier alphas.
        var inclusivePrefix = Engine.TensorCumSum(alphaTensor, axis: 1);      // [B, S, 1]
        var prefix = Engine.TensorSubtract(inclusivePrefix, alphaTensor);     // exclusive

        // The contraction, algebraically rearranged to keep the tape off the S-by-S plane.
        //
        // Each output row is out_i = Σ_t W[i,t] · h_t with
        //     W[i,t] = A[i,t]·α_t + Bp[i,t]·P_t + C[i,t].
        // Because α_t and P_t do not depend on the OUTPUT index i, the broadcast copies of them
        // are redundant: (A ⊙ αB) · h = A · (α ⊙ h), and likewise for the prefix term. So
        //     out = A·(α ⊙ h) + Bp·(P ⊙ h) + C·h
        // which is identical arithmetic with the elementwise products moved from [B, S, S] down
        // to [B, S, D].
        //
        // That matters beyond memory. A, Bp and C are constants with respect to the tape -- the
        // firing structure is decided in the non-differentiable pass above -- so in this form
        // they are plain matmul operands and the only tape intermediates are [B, S, D]. The
        // previous form multiplied them INTO tape tensors, so the tape had to carry three
        // S-by-S intermediates per forward. Under an active TensorArena that was enough to make
        // the alpha predictor's gradient come back non-finite after a single training step,
        // while the forward stayed finite and the same model trained cleanly with no arena.
        var alphaBroadcast = Engine.TensorBroadcastTo(alphaTensor, [B, S, D]);
        var prefixBroadcast = Engine.TensorBroadcastTo(prefix, [B, S, D]);

        var alphaWeighted = Engine.TensorMultiply(alphaBroadcast, input);     // [B, S, D]
        var prefixWeighted = Engine.TensorMultiply(prefixBroadcast, input);   // [B, S, D]

        var output = Engine.TensorAdd(
            Engine.TensorAdd(
                Engine.TensorBatchMatMul(coeff, alphaWeighted),
                Engine.TensorBatchMatMul(prefixCoeff, prefixWeighted)),
            Engine.TensorBatchMatMul(constant, input));                       // [B, S, D]

        return output;
    }

    /// <inheritdoc/>
    /// <summary>
    /// Materializes the alpha predictor's context window: for each timestep <c>u</c>, the
    /// concatenation <c>[h_{u-1} | h_u | h_{u+1}]</c>, zero-padded at the sequence edges.
    /// </summary>
    /// <remarks>
    /// Dong &amp; Xu 2020 (arXiv:1905.11235) predict the firing weight from a window centred on
    /// <c>h_u</c> rather than from <c>h_u</c> alone, so the predictor can see the frame-to-frame
    /// change that marks a token boundary.
    /// </remarks>
    /// <summary>
    /// Zeroes a freshly allocated scratch tensor.
    /// </summary>
    /// <remarks>
    /// Several buffers in this layer are written SPARSELY and rely on every untouched element
    /// being zero: the firing coefficients are set only at positions that actually fire, the
    /// prefix-sum operand only below its diagonal, and the context window only where a neighbour
    /// exists. Relying on the allocator to hand back zeroed memory is not safe when a
    /// TensorArena is active, because pooled buffers carry whatever the previous tenant left in
    /// them. The stale values then flow straight into the matmuls that build the output weights.
    ///
    /// Measured: with an arena active, CIFEncoder's alpha predictor went non-finite after a
    /// single training step; with no arena the same model trained cleanly for twelve. That also
    /// explains the run-to-run variation -- the same binary failed 6, 5 or 4 of its 26 tests
    /// depending on what happened to be in the pool.
    /// </remarks>
    private static Tensor<T> Zeroed(int[] shape)
    {
        var tensor = new Tensor<T>(shape);
        var zero = MathHelper.GetNumericOperations<T>().Zero;
        for (int i = 0; i < tensor.Length; i++) tensor[i] = zero;
        return tensor;
    }

    private static Tensor<T> BuildAlphaWindow(Tensor<T> input, int B, int S, int D)
    {
        var windowed = Zeroed(new[] { B, S, 3 * D });

        for (int b = 0; b < B; b++)
        {
            for (int s = 0; s < S; s++)
            {
                int outBase = ((b * S) + s) * 3 * D;

                for (int offset = -1; offset <= 1; offset++)
                {
                    int src = s + offset;
                    int slot = (offset + 1) * D;

                    // Edge frames have no neighbour on one side; zero-pad so the window stays
                    // a fixed 3D width and the predictor's weights keep a stable meaning.
                    if (src < 0 || src >= S) continue;

                    int inBase = ((b * S) + src) * D;
                    for (int d = 0; d < D; d++)
                        windowed[outBase + slot + d] = input[inBase + d];
                }
            }
        }

        return windowed;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Delegates to the alpha predictor. The base implementation returns only tensors registered
    /// directly on THIS layer and does not recurse into children, so without this override a
    /// composite layer reports an empty trainable set: <c>GetParameters()</c> returned 49
    /// elements while <c>GetTrainableParameters()</c> returned none. That mismatch is invisible
    /// while <see cref="SupportsTraining"/> is false, but once the layer trains it desynchronizes
    /// the flat parameter vector from the tensor set the tape and <c>ParameterBuffer</c> track,
    /// which surfaced as "Parameter[0] is NaN after training" in every CIF consumer.
    /// </remarks>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters()
        => _alphaPredictor.GetTrainableParameters();

    /// <inheritdoc/>
    /// <remarks>
    /// Forwards buffer-backed views straight through to the alpha predictor so the tensors used
    /// during <see cref="Forward"/> are the same references the ParameterBuffer holds — the
    /// tape's reference-identity alignment check requires that.
    /// </remarks>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
        => _alphaPredictor.SetTrainableParameters(parameters);

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
