using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A single Time-MoE transformer block: multi-head self-attention + Mixture-of-Experts FFN,
/// each wrapped in pre-norm + residual. Per Shi et al. 2024 "Time-MoE: Billion-Scale Time Series
/// Foundation Models with Mixture of Experts".
/// </summary>
/// <remarks>
/// <para>
/// Forward sequence (pre-norm, GPT-style):
/// <list type="number">
/// <item><description>norm1(input) → self-attention → residual add with input → x</description></item>
/// <item><description>norm2(x) → MoE-FFN (top-k expert dispatch) → residual add with x → output</description></item>
/// </list>
/// </para>
/// <para>
/// The MoE routes each token (each row of the flattened [B·numPatches, hiddenDim] tensor)
/// independently through top-k experts, per paper. Each expert is a 2-layer Dense FFN with
/// GELU (hiddenDim → intermediateSize → hiddenDim). Load-balancing is enabled with weight
/// 0.01.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric element type.</typeparam>
[LayerCategory(LayerCategory.Transformer)]
[LayerCategory(LayerCategory.MixtureOfExperts)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.Routing)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.High, TestInputShape = "4, 8", TestConstructorArgs = "8, 2, 16, 4, 2")]
// SHAPE-PRESERVING, and ForwardTraced proves it rather than the architecture implying it: both
// sublayers land in `Engine.TensorAdd(input, attn)` and `Engine.TensorAdd(x, moe)`, and a residual
// add is only defined when the two operands already agree, so the output shape is the input shape.
// Nothing else to write - matching input/output layouts make the generator derive Same(role) per axis.
//
// Deliberately NOT [ElementWiseShape]: that shorthand claims EVERY rank down to 1, and the
// self-attention sublayer needs a sequence axis as well as a feature axis. SetParameters says so in
// its own words - "MHA requires rank>=2 (sequence + features)" - so rank 1 is not this layer's to
// accept. Rank 3 is declared because the class remarks describe the MoE routing "each row of the
// flattened [B*numPatches, hiddenDim] tensor", i.e. a batched sequence.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class TimeMoEBlockLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _intermediateSize;
    private readonly int _numExperts;
    private readonly int _topK;

    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly MultiHeadAttentionLayer<T> _selfAttention;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly MixtureOfExpertsLayer<T> _moe;

    /// <summary>
    /// Surfaces the MoE router's load-balancing auxiliary loss for the current
    /// (most recent) forward pass. The enclosing training loop can add this to
    /// the main task loss per Shi et al. 2024 §3.2 to prevent expert collapse.
    /// Returns zero before any forward has been executed. Exposes a scalar —
    /// keeps the router/expert implementation an internal detail of this layer.
    /// </summary>
    public T GetAuxiliaryLoss() =>
        _moe.UseAuxiliaryLoss ? _moe.ComputeAuxiliaryLoss() : NumOps.Zero;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Persists the constructor arguments so the deserializer can rebuild this layer
    /// at the same shape. Without this override the layer falls back to default values
    /// and SetParameters throws "expected N parameters across sublayers, got M" because
    /// MoE expert/intermediate counts drive total parameter count.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["HiddenDim"] = _hiddenDim.ToString(ci);
        metadata["NumHeads"] = _numHeads.ToString(ci);
        metadata["IntermediateSize"] = _intermediateSize.ToString(ci);
        metadata["NumExperts"] = _numExperts.ToString(ci);
        metadata["TopK"] = _topK.ToString(ci);
        return metadata;
    }

    /// <summary>
    /// Initializes a new <see cref="TimeMoEBlockLayer{T}"/>.
    /// </summary>
    /// <param name="hiddenDim">Per-token hidden dimension (the feature dim of the block input / output).</param>
    /// <param name="numHeads">Number of self-attention heads. hiddenDim must be divisible by numHeads.</param>
    /// <param name="intermediateSize">Per-expert FFN intermediate (expanded) width.</param>
    /// <param name="numExperts">Number of experts in the MoE FFN.</param>
    /// <param name="topK">Number of experts to route each token through (sparse dispatch).</param>
    public TimeMoEBlockLayer(int hiddenDim, int numHeads, int intermediateSize, int numExperts, int topK)
        : base(new[] { hiddenDim }, new[] { hiddenDim })
    {
        if (hiddenDim < 1) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (numHeads < 1) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (hiddenDim % numHeads != 0)
            throw new ArgumentException(
                $"hiddenDim ({hiddenDim}) must be divisible by numHeads ({numHeads}).",
                nameof(numHeads));
        if (intermediateSize < 1) throw new ArgumentOutOfRangeException(nameof(intermediateSize));
        if (numExperts < 1) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (topK < 1 || topK > numExperts)
            throw new ArgumentOutOfRangeException(nameof(topK),
                $"topK must be in [1, numExperts]; got topK={topK}, numExperts={numExperts}.");

        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _intermediateSize = intermediateSize;
        _numExperts = numExperts;
        _topK = topK;

        _norm1 = new LayerNormalizationLayer<T>();

        // sequenceLength=1 is the placeholder used by TransformerEncoderLayer; the attention
        // layer supports any rank and reshapes internally.
        _selfAttention = new MultiHeadAttentionLayer<T>(numHeads, (hiddenDim) / (numHeads), 
            activationFunction: new GELUActivation<T>() as IActivationFunction<T>);

        _norm2 = new LayerNormalizationLayer<T>();

        // Build numExperts experts, each a 2-layer Dense FFN: hiddenDim -> intermediateSize
        // (GELU) -> hiddenDim. ExpertLayer wraps a list of sub-layers and applies them
        // sequentially; we compose each expert from two DenseLayers.
        var experts = new List<ILayer<T>>(numExperts);
        for (int e = 0; e < numExperts; e++)
        {
            var innerLayers = new List<ILayer<T>>
            {
                new DenseLayer<T>(
                    outputSize: intermediateSize,
                    activationFunction: new GELUActivation<T>()),
                new DenseLayer<T>(
                    outputSize: hiddenDim,
                    activationFunction: null),
            };
            experts.Add(new ExpertLayer<T>(
                innerLayers,
                new[] { hiddenDim },
                new[] { hiddenDim }));
        }

        // Router: dense projection from per-token hidden → per-expert score.
        var router = new DenseLayer<T>(
            outputSize: numExperts,
            activationFunction: null);

        _moe = new MixtureOfExpertsLayer<T>(
            experts: experts,
            router: router,
            inputShape: new[] { hiddenDim },
            outputShape: new[] { hiddenDim },
            topK: topK,
            activationFunction: null,
            useLoadBalancing: true,
            loadBalancingWeight: NumOps.FromDouble(0.01));

        RegisterSubLayer(_norm1);
        RegisterSubLayer(_selfAttention);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_moe);
    }

    /// <summary>
    /// Sizes the four sub-layers from <c>_hiddenDim</c>, which is the only thing that determines
    /// their weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the one fact about the block that cannot be derived: its children are sized by the
    /// block's HIDDEN width, not by the shape of the tensor arriving at its input, and nothing
    /// outside the block knows that. It used to live inside a hand-written
    /// <c>SetParameters</c> — so a restore materialized the children, while <c>ParameterCount</c>
    /// asked the same unmaterialized children and answered a smaller number. The saved and loaded
    /// blocks then disagreed by 2,424 values.
    /// </para>
    /// <para>
    /// Stating it here instead makes it a SHAPE fact, declared once, and count / read / restore
    /// all inherit it — none of them has to re-derive the parameter layout, which is what the
    /// three hand-written surfaces were really for.
    /// </para>
    /// <para>
    /// MultiHeadAttention requires rank >= 2 (sequence + features), so it gets a length-1
    /// sequence; its weight count depends only on <c>_hiddenDim</c> either way.
    /// </para>
    /// </remarks>
    protected override IReadOnlyList<(LayerBase<T>? Child, TensorShape InputShape)> DeclaredSubLayerShapes()
    {
        if (_declaredSubLayerShapes is not null) return _declaredSubLayerShapes;

        var hidden = ShapeOf(_hiddenDim);
        _declaredSubLayerShapes = new (LayerBase<T>?, TensorShape)[]
        {
            (_norm1,         hidden),
            (_selfAttention, ShapeOf(1, _hiddenDim)),
            (_norm2,         hidden),
            (_moe,           hidden),
        };
        return _declaredSubLayerShapes;
    }

    /// <summary>Cached <see cref="DeclaredSubLayerShapes"/>; the children and _hiddenDim never change.</summary>
    private (LayerBase<T>? Child, TensorShape InputShape)[]? _declaredSubLayerShapes;

    protected override void EnsureInitialized()
    {
        EnsureSubLayersRegistered();

        // The per-child IsShapeResolved guards were redundant -- ResolveFromShape opens with
        // `if (IsShapeResolved) return;` -- and going through the base instead means a shape walker
        // or a deferring constructor gets dimensions without paying for weights, which the direct
        // ResolveFromShape calls did not allow.
        BringUpDeclaredSubLayers();

        base.EnsureInitialized();
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Pre-norm + attention + residual
        var normed1 = _norm1.Forward(input);
        var attn = _selfAttention.Forward(normed1);
        var x = Engine.TensorAdd(input, attn);

        // Pre-norm + MoE-FFN + residual. MoE internally flattens rank>=3 inputs to
        // [B*numPatches, hiddenDim], routes per-token through top-k experts, and reshapes
        // back to the original input shape.
        var normed2 = _norm2.Forward(x);
        var moe = _moe.Forward(normed2);
        var output = Engine.TensorAdd(x, moe);
        return output;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _selfAttention.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _moe.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _norm1.ResetState();
        _selfAttention.ResetState();
        _norm2.ResetState();
        _moe.ResetState();
    }
}
