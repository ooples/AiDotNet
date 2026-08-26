using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A Conformer block — the convolution-augmented transformer block of Gulati et al. 2020
/// ("Conformer: Convolution-augmented Transformer for Speech Recognition", arXiv:2005.08100),
/// and the encoder block used by Google USM / Chirp (Zhang et al. 2023, arXiv:2303.01037 §2.1).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The block implements the paper's Equation 1 exactly — a "macaron" sandwich of two HALF-step
/// feed-forward modules around self-attention and a convolution module, every sub-module wrapped in
/// a pre-norm residual, closed by a final LayerNorm:
/// </para>
/// <code>
///   x̃  = x  + ½·FFN(LN(x))
///   x'  = x̃  + MHSA(LN(x̃))
///   x'' = x'  + Conv(LN(x'))
///   y   = LayerNorm(x'' + ½·FFN(LN(x'')))
/// </code>
/// <para>
/// The convolution module is the part that makes a Conformer a Conformer, and it is what the older
/// Dense-only "conformer" factories in this repository were missing entirely (their own comments
/// admitted the position-wise FFN stood in for a convolution that "was never a real convolution").
/// Per §2.2 of the paper it is: pointwise convolution with expansion factor 2 → GLU → 1-D DEPTHWISE
/// convolution → normalization → Swish → pointwise convolution.
/// </para>
/// <para>
/// The convolution module keeps the paper's BatchNorm. Substituting LayerNorm there — normalizing per
/// time position instead of per channel — is NOT equivalent: its 1/std backward blows up whenever the
/// convolution output at a position is near-degenerate, and it made the very first Adam step produce
/// NaN (parameter L2 34.4 → NaN). BatchNorm normalizes each channel across batch and time, which is
/// well-conditioned for convolution features, and it is what the paper specifies.
/// </para>
/// <para>
/// One deliberate, documented deviation from the paper:
/// </para>
/// <list type="bullet">
/// <item><b>Relative attention.</b> The paper (and USM) use Transformer-XL style relative sinusoidal
/// positional encoding. <see cref="MultiHeadAttentionLayer{T}"/> offers Rotary (RoPE) and ALiBi;
/// Rotary is selected because it is likewise a RELATIVE scheme — attention depends on position
/// differences, not absolute indices — which is the property the paper relies on for variable-length
/// speech.</item>
/// </list>
/// <para>
/// Input/output layout is <c>[..., time, dim]</c> (time-major, matching the rest of the ASR stack).
/// The convolution module transposes internally to the <c>[batch, channels, time]</c> layout the
/// Conv1D primitives expect and transposes back, so the block is a drop-in for a transformer block.
/// All shape work routes through <see cref="LayerBase{T}.Engine"/> so the gradient tape records the
/// graph — no manual backward, same contract as the other composite blocks.
/// </para>
/// <para><b>For Beginners:</b> A plain transformer block is good at relating distant parts of a
/// sequence but weak at fine local detail. Speech needs both: phonemes are local, sentences are
/// global. A Conformer block adds a small convolution — which looks at neighbouring time frames — in
/// between the attention and the feed-forward parts, so the model captures local sound patterns and
/// long-range context at once.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, NormalizesInput = true, Cost = ComputeCost.High,
    TestInputShape = "4, 16", TestConstructorArgs = "16, 2, 2, 3")]
// Roles are the block's own documented layout, quoted from the remarks above: "Input/output layout is
// [..., time, dim] (time-major, matching the rest of the ASR stack)". Batch is optional because that
// "..." is genuinely empty at the rank the block is tested at ([LayerProperty(TestInputShape = "4, 16")]
// is [time, dim]) and genuinely one axis deep above it; ConvolutionModule folds every leading axis into
// one ("for (int i = 0; i < rank - 2; i++) batch *= input.Shape[i];") so both run the same code.
//
// SHAPE-PRESERVING, and structurally so rather than by arithmetic coincidence: Equation 1 is four
// pre-norm RESIDUAL adds against the running activation, and a residual add cannot resize its operand.
// The convolution module is the only sub-module that changes layout at all, and it explicitly restores
// the caller's shape on the way out ("Engine.Reshape(outTd, input.Shape.ToArray())") -- its 2*D channel
// expansion is halved again by the GLU before that point, so even the interior width comes back to D.
// The closing LayerNorm rescales values only.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ConformerBlockLayer<T> : LayerBase<T>, IShapeContract
{

    /// <summary>Construction state, retained so the layer can be rebuilt exactly rather than inferred from its shape.</summary>
    private readonly int _maxSequenceLength;

    /// <summary>Construction state, retained so the layer can be rebuilt exactly rather than inferred from its shape.</summary>
    private readonly double _ropeTheta;
    private readonly int _modelDim;
    private readonly int _numHeads;
    private readonly int _ffnExpansionFactor;
    private readonly int _convKernelSize;
    private readonly int _attentionGroupSize;

    // Macaron feed-forward #1 (half-step residual).
    private readonly LayerNormalizationLayer<T> _ffn1Norm;
    private readonly FullyConnectedLayer<T> _ffn1Expand;
    private readonly FullyConnectedLayer<T> _ffn1Project;

    // Multi-head self-attention with relative (rotary) positional encoding.
    private readonly LayerNormalizationLayer<T> _attnNorm;
    private readonly MultiHeadAttentionLayer<T> _attention;

    // Convolution module.
    private readonly LayerNormalizationLayer<T> _convNorm;
    private readonly Conv1DLayer<T> _convPointwiseExpand;   // -> 2*dim channels, feeds the GLU
    private readonly DepthwiseConv1DLayer<T> _convDepthwise;
    private readonly BatchNormalizationLayer<T> _convInnerNorm;
    private readonly Conv1DLayer<T> _convPointwiseProject;  // -> dim channels

    // Macaron feed-forward #2 (half-step residual) + closing LayerNorm.
    private readonly LayerNormalizationLayer<T> _ffn2Norm;
    private readonly FullyConnectedLayer<T> _ffn2Expand;
    private readonly FullyConnectedLayer<T> _ffn2Project;
    private readonly LayerNormalizationLayer<T> _outputNorm;

    private readonly T _half;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a Conformer block.
    /// </summary>
    /// <param name="modelDim">Model (encoder) dimension d_model.</param>
    /// <param name="numHeads">Number of attention heads; must divide <paramref name="modelDim"/>.</param>
    /// <param name="ffnExpansionFactor">Feed-forward expansion factor. Paper default 4.</param>
    /// <param name="convKernelSize">
    /// Depthwise convolution kernel size. The Conformer paper uses 31/32; USM reports a 5-frame
    /// kernel for its Conformer encoders (arXiv:2303.01037 §2.1).
    /// </param>
    /// <param name="ropeTheta">Rotary base frequency for the relative positional encoding.</param>
    /// <param name="maxSequenceLength">Maximum sequence length the positional encoding supports.</param>
    /// <param name="attentionGroupSize">
    /// Number of adjacent projected frames combined into one attention position. EfficientConformer
    /// uses three in its first stage; ordinary Conformer keeps the default value of one.
    /// </param>
    public ConformerBlockLayer(
        [LayerState] int modelDim,
        [LayerState] int numHeads,
        [LayerState] int ffnExpansionFactor = 4,
        [LayerState] int convKernelSize = 5,
        [LayerState] double ropeTheta = 10000.0,
        [LayerState] int maxSequenceLength = 2048,
        [LayerState] int attentionGroupSize = 1)
        : base(new[] { -1, modelDim }, new[] { -1, modelDim })
    {
        _maxSequenceLength = maxSequenceLength;
        _ropeTheta = ropeTheta;
        if (modelDim <= 0) throw new ArgumentOutOfRangeException(nameof(modelDim), "modelDim must be positive.");
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");
        if (modelDim % numHeads != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        if (ffnExpansionFactor <= 0) throw new ArgumentOutOfRangeException(nameof(ffnExpansionFactor), "ffnExpansionFactor must be positive.");
        if (convKernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(convKernelSize), "convKernelSize must be positive.");
        if (attentionGroupSize <= 0) throw new ArgumentOutOfRangeException(nameof(attentionGroupSize), "attentionGroupSize must be positive.");

        _modelDim = modelDim;
        _numHeads = numHeads;
        _ffnExpansionFactor = ffnExpansionFactor;
        _convKernelSize = convKernelSize;
        _attentionGroupSize = attentionGroupSize;
        _half = NumOps.FromDouble(0.5);

        var swish = (Interfaces.IActivationFunction<T>)new SwishActivation<T>();
        var identity = (Interfaces.IActivationFunction<T>)new IdentityActivation<T>();
        int ffnDim = modelDim * ffnExpansionFactor;

        // Every sub-layer below is constructed with EXPLICIT input dimensions rather than relying on
        // lazy first-forward resolution. That is required for Clone/deserialize: SetParameters slices
        // the flat vector by each sub-layer's ParameterCount, and an unresolved lazy layer reports no
        // parameters yet — so a fresh clone would mis-slice and silently receive the wrong weights
        // (Clone_ShouldProduceIdenticalOutput / Clone_AfterTraining_ShouldPreserveLearnedWeights).
        // All the widths are known here, so there is no reason to defer them.

        // --- Macaron FFN #1: LN -> Linear(d -> d*ff) -> Swish -> Linear(-> d) ---
        _ffn1Norm = new LayerNormalizationLayer<T>(modelDim);
        _ffn1Expand = new FullyConnectedLayer<T>(modelDim, ffnDim, swish);
        _ffn1Project = new FullyConnectedLayer<T>(ffnDim, modelDim, identity);

        // --- Self-attention with RELATIVE positional encoding ---
        _attnNorm = new LayerNormalizationLayer<T>(modelDim);
        _attention = new MultiHeadAttentionLayer<T>(numHeads, modelDim / numHeads)
            .ConfigureAttentionGrouping(attentionGroupSize);
        _attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, ropeTheta, maxSequenceLength);

        // --- Convolution module (paper §2.2) ---
        // Pointwise conv to 2*dim so the GLU can gate one half with the other, then a depthwise
        // conv over time, normalization, Swish, and a pointwise conv back to dim.
        _convNorm = new LayerNormalizationLayer<T>(modelDim);
        _convPointwiseExpand = new Conv1DLayer<T>(inputChannels: modelDim, outputChannels: modelDim * 2, kernelSize: 1, dilation: 1, stride: 1, padding: 0, activation: identity);
        _convDepthwise = new DepthwiseConv1DLayer<T>(modelDim, convKernelSize, multiplier: 1, stride: 1, padding: convKernelSize / 2, activation: identity);
        // BatchNorm, exactly as the paper specifies (§2.2). This is NOT interchangeable with LayerNorm
        // here: LayerNorm normalizes per time position, and its 1/std backward blows up when the
        // convolution output at a position is near-degenerate — substituting it made the first Adam
        // step produce NaN (param L2 34.4 -> NaN, "Parameter[0] is NaN after training"). BatchNorm
        // normalizes each channel across batch AND time, which is well-conditioned for conv features.
        // Applied on the [B, T, D] view below so it acts per channel, matching BatchNorm1d semantics.
        _convInnerNorm = new BatchNormalizationLayer<T>(modelDim);
        _convPointwiseProject = new Conv1DLayer<T>(inputChannels: modelDim, outputChannels: modelDim, kernelSize: 1, dilation: 1, stride: 1, padding: 0, activation: identity);

        // --- Macaron FFN #2 + closing LayerNorm ---
        _ffn2Norm = new LayerNormalizationLayer<T>(modelDim);
        _ffn2Expand = new FullyConnectedLayer<T>(modelDim, ffnDim, swish);
        _ffn2Project = new FullyConnectedLayer<T>(ffnDim, modelDim, identity);
        _outputNorm = new LayerNormalizationLayer<T>(modelDim);

        RegisterSubLayer(_ffn1Norm);
        RegisterSubLayer(_ffn1Expand);
        RegisterSubLayer(_ffn1Project);
        RegisterSubLayer(_attnNorm);
        RegisterSubLayer(_attention);
        RegisterSubLayer(_convNorm);
        RegisterSubLayer(_convPointwiseExpand);
        RegisterSubLayer(_convDepthwise);
        RegisterSubLayer(_convInnerNorm);
        RegisterSubLayer(_convPointwiseProject);
        RegisterSubLayer(_ffn2Norm);
        RegisterSubLayer(_ffn2Expand);
        RegisterSubLayer(_ffn2Project);
        RegisterSubLayer(_outputNorm);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // x̃ = x + ½·FFN(LN(x))
        var ffn1 = _ffn1Project.Forward(_ffn1Expand.Forward(_ffn1Norm.Forward(input)));
        var x1 = Engine.TensorAdd(input, Engine.TensorMultiplyScalar(ffn1, _half));

        // x' = x̃ + MHSA(LN(x̃))
        var attn = _attention.Forward(_attnNorm.Forward(x1));
        var x2 = Engine.TensorAdd(x1, attn);

        // x'' = x' + Conv(LN(x'))
        var conv = ConvolutionModule(_convNorm.Forward(x2));
        var x3 = Engine.TensorAdd(x2, conv);

        // y = LayerNorm(x'' + ½·FFN(LN(x'')))
        var ffn2 = _ffn2Project.Forward(_ffn2Expand.Forward(_ffn2Norm.Forward(x3)));
        var x4 = Engine.TensorAdd(x3, Engine.TensorMultiplyScalar(ffn2, _half));
        return _outputNorm.Forward(x4);
    }

    /// <summary>
    /// Convolution module of the Conformer block (Gulati et al. 2020 §2.2):
    /// pointwise conv (×2 expansion) → GLU → depthwise conv → norm → Swish → pointwise conv.
    /// </summary>
    /// <remarks>
    /// The Conv1D primitives operate on <c>[batch, channels, time]</c>, while the block works in
    /// <c>[..., time, dim]</c>, so the tensor is transposed in and back out. Everything routes through
    /// the engine, keeping the whole module differentiable on the tape.
    /// </remarks>
    private Tensor<T> ConvolutionModule(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        int time = input.Shape[rank - 2];
        int dim = input.Shape[rank - 1];
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input.Shape[i];

        // [..., T, D] -> [B, T, D] -> [B, D, T]
        var btd = Engine.Reshape(input, new[] { batch, time, dim });
        var bdt = Engine.TensorPermute(btd, new[] { 0, 2, 1 });

        // Pointwise conv to 2*D channels, then GLU: split the channels in half and gate.
        var expanded = _convPointwiseExpand.Forward(bdt);          // [B, 2D, T]
        var gateA = Engine.TensorSlice(expanded, new[] { 0, 0, 0 }, new[] { batch, dim, time });
        var gateB = Engine.TensorSlice(expanded, new[] { 0, dim, 0 }, new[] { batch, dim, time });
        var gated = Engine.TensorMultiply(gateA, Engine.Sigmoid(gateB));   // GLU: a ⊙ σ(b)

        // Depthwise conv over time, then normalization + Swish (paper order), then pointwise back.
        var depthwise = _convDepthwise.Forward(gated);             // [B, D, T]

        // Normalize per time position: transpose to [B, T, D] so LayerNorm acts on the feature axis.
        var normedTd = _convInnerNorm.Forward(Engine.TensorPermute(depthwise, new[] { 0, 2, 1 }));
        var activated = Engine.Swish(normedTd);
        var backToBdt = Engine.TensorPermute(activated, new[] { 0, 2, 1 });

        var projected = _convPointwiseProject.Forward(backToBdt);  // [B, D, T]

        // [B, D, T] -> [B, T, D] -> original rank
        var outTd = Engine.TensorPermute(projected, new[] { 0, 2, 1 });
        // Rank-3 is the native Conformer contract, and outTd already has exactly the requested
        // [B,T,D] shape. Returning it directly avoids wrapping an arena-backed permutation in a
        // redundant reshape view whose storage can be recycled while the enclosing residual block
        // still needs it. Higher-rank callers still need their flattened leading axes restored.
        return rank == 3 ? outTd : Engine.Reshape(outTd, input.Shape.ToArray());
    }

    private ILayer<T>[] OrderedSubLayers => new ILayer<T>[]
    {
        _ffn1Norm, _ffn1Expand, _ffn1Project,
        _attnNorm, _attention,
        _convNorm, _convPointwiseExpand, _convDepthwise, _convInnerNorm, _convPointwiseProject,
        _ffn2Norm, _ffn2Expand, _ffn2Project, _outputNorm
    };

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var parts = new List<T>((int)ParameterCount);
        foreach (var layer in OrderedSubLayers)
        {
            var g = layer.GetParameterGradients();
            for (int i = 0; i < g.Length; i++) parts.Add(g[i]);
        }
        return new Vector<T>(parts.ToArray());
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in OrderedSubLayers) layer.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in OrderedSubLayers) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in OrderedSubLayers) layer.ResetState();
    }

    /// <summary>
    /// Persists the constructor configuration so the block can be rebuilt during deserialization /
    /// Clone. Sub-layer WEIGHTS travel in the flat parameter vector; this metadata only carries the
    /// structure. Reconstruction lives in <c>DeserializationHelper.CreateConformerBlockLayer</c>.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["ModelDim"] = _modelDim.ToString(ci);
        metadata["NumHeads"] = _numHeads.ToString(ci);
        metadata["FfnExpansionFactor"] = _ffnExpansionFactor.ToString(ci);
        metadata["ConvKernelSize"] = _convKernelSize.ToString(ci);
        metadata["AttentionGroupSize"] = _attentionGroupSize.ToString(ci);
        metadata["RopeTheta"] = _attention.RopeTheta.ToString(ci);
        metadata["PositionalMaxSequenceLength"] = _attention.PositionalMaxSequenceLength.ToString(ci);
        return metadata;
    }
}
