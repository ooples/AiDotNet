using AiDotNet.Diffusion.Attention;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.NeuralNetworks.Layers;

// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// One transformer block of Wonder3D's cross-domain U-Net (<c>BasicMVTransformerBlock</c> with
/// <c>multiview_attention</c> and <c>cd_attention_mid</c>): the U-Net's self-attention made multi-view, followed by
/// cross-domain attention between the normal and color branches.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The input batch holds, for each object, its <c>NumViews</c> normal-domain views followed later by the same views in
/// the color domain: <c>[normal views of every object, color views of every object]</c>, the order of Wonder3D's
/// <c>reshape_to_cd_input</c>.
/// </para>
/// <para>
/// <b>Multi-view self-attention</b> (<c>MVAttnProcessor</c>): the wrapped block's own query, key and value
/// projections, with every view of an object reading the keys and values of all its views
/// (<see cref="DiffusionAttention{T}.ForwardJointTokens"/> with a group of <c>NumViews</c>).
/// </para>
/// <para>
/// The wrapped block is the U-Net's LDM <c>SpatialTransformer</c> (its default) or a standalone
/// <see cref="DiffusionAttention{T}"/>. In the transformer the multi-view self-attention is <c>attn1</c> over the
/// <c>NumViews</c> views of one domain, and the cross-domain attention runs inside each transformer block right after
/// it, before <c>attn2</c> and the feed-forward, as <c>BasicMVTransformerBlock.forward</c> orders them
/// (<c>attn1</c>, then <c>attn_joint_mid</c> when <c>cd_attention_mid</c>, then <c>attn2</c>, then <c>ff</c>;
/// Wonder3D <c>mvdiffusion/models/transformer_mv2d.py</c>). Around a standalone self-attention block it follows it.
/// </para>
/// <para>
/// <b>Cross-domain attention</b> (<c>attn_joint_mid</c>, <c>JointAttnProcessor</c>): <c>x + Attn(LayerNorm(x))</c>
/// where a token of view v in one domain attends over the tokens of view v in BOTH domains,
/// <c>key = cat([key_normal, key_color], dim=1)</c> for both, with its own projections whose output weight starts at
/// zero (<c>nn.init.zeros_(attn_joint_mid.to_out[0].weight)</c>), so a freshly built block is the plain multi-view
/// block.
/// </para>
/// <para><b>For Beginners:</b> The first step lets the six views look at one another so they agree on the object;
/// the second lets each view's normal map and color image look at each other so the shape and the colors agree.</para>
/// </remarks>
// A DECORATOR of the U-Net's self-attention block, shape-preserving like the block it wraps: the multi-view pass
// joins and splits tokens of the same [B, C, H, W] input, and the cross-domain pass is a residual of that shape.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input,
    Note = "Image form: the batch is [normal views, color views], each NumViews long per object.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class Wonder3DAttentionBlock<T> : LayerBase<T>, IAttentionBlockDecorator<T>, IShapeContract,
    IContextualAttentionBlock<T>
{
    private readonly ILayer<T> _inner;
    private readonly LayerNormalizationLayer<T> _crossDomainNorm;
    private readonly FlashAttentionLayer<T> _crossDomainAttention;

    /// <summary>Construction state: the 'channels' the block was built with.</summary>
    private readonly int _channels;

    /// <summary>Construction state: the 'numViews' the block was built with.</summary>
    private readonly int _numViews;

    /// <summary>Construction state: the 'numHeads' the block was built with.</summary>
    private readonly int _numHeads;

    /// <summary>Construction state: the 'seed' the block was built with.</summary>
    private readonly int? _seed;

    /// <summary>Wraps a U-Net attention block.</summary>
    /// <param name="inner">The U-Net's SpatialTransformer, or a self-attention block.</param>
    /// <param name="channels">The block's channel width.</param>
    /// <param name="numViews">The views per object.</param>
    /// <param name="numHeads">The cross-domain attention's heads (the U-Net's).</param>
    /// <param name="seed">The cross-domain projections' initialization seed, or null.</param>
    public Wonder3DAttentionBlock(ILayer<T> inner, int channels, int numViews, int numHeads, int? seed = null)
        : base(inner?.GetInputShape() ?? throw new ArgumentNullException(nameof(inner)), inner.GetOutputShape())
    {
        if (inner is not DiffusionAttention<T> && inner is not MultiViewTransformerLayer<T>)
            throw new ArgumentException(
                $"Wonder3D's attention wraps the U-Net's SpatialTransformer or a DiffusionAttention block; got {inner.GetType().Name}.",
                nameof(inner));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels), "The width must be positive.");
        if (numViews <= 0) throw new ArgumentOutOfRangeException(nameof(numViews), "The view count must be positive.");
        if (numHeads <= 0 || channels % numHeads != 0)
            throw new ArgumentException($"The width {channels} must split into {numHeads} heads.", nameof(numHeads));
        _inner = inner;
        _channels = channels;
        _numViews = numViews;
        _numHeads = numHeads;
        _seed = seed;

        // The block's tokens per view, from the wrapped block's declared [1, H·W, C] shape; the joint sequence of a
        // view pair is twice that. Flash attention reads the real length from its input either way.
        var innerShape = inner.GetInputShape();
        int tokens = innerShape.Length >= 2 && innerShape[1] > 0 ? innerShape[1] : 1;
        _crossDomainNorm = new LayerNormalizationLayer<T>(channels, 1e-5);
        _crossDomainAttention = new FlashAttentionLayer<T>(2 * tokens, channels, numHeads)
        {
            ZeroInitializeOutputProjection = true,
        };
        if (seed.HasValue) _crossDomainAttention.RandomSeed = seed.Value;
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => (_inner as IShapeContract)?.OutputAxesFor(inputRank);

    /// <inheritdoc/>
    public ILayer<T> Inner => _inner;

    /// <summary>The views per object.</summary>
    public int NumViews => _numViews;

    /// <summary>The channel width.</summary>
    public int Channels => _channels;

    /// <summary>The cross-domain attention's LayerNorm (<c>norm_joint_mid</c>).</summary>
    internal LayerNormalizationLayer<T> CrossDomainNorm => _crossDomainNorm;

    /// <summary>The cross-domain attention (<c>attn_joint_mid</c>).</summary>
    internal FlashAttentionLayer<T> CrossDomainAttention => _crossDomainAttention;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Multi-view self-attention, then cross-domain attention (no cross-attention context).</summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Run(input, null);

    /// <inheritdoc/>
    Tensor<T> IContextualAttentionBlock<T>.Forward(Tensor<T> input, Tensor<T>? context) => Run(input, context);

    // The transformer runs whole with its self-attention grouped by view and the cross-domain attention inside each
    // block; a standalone self-attention block is followed by it.
    private Tensor<T> Run(Tensor<T> input, Tensor<T>? context)
    {
        if (IsInferringShapes)
            return _inner is MultiViewTransformerLayer<T> inferred ? inferred.Forward(input, context) : _inner.Forward(input);
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Shape.Length != 4)
            throw new ArgumentException($"Wonder3D attention takes [batch, channels, height, width]; got rank {input.Shape.Length}.",
                nameof(input));
        int batch = input.Shape[0];
        if (batch % (2 * _numViews) != 0)
            throw new ArgumentException(
                $"Wonder3D attention needs every object's {_numViews} views in both domains, a batch that is a multiple of " +
                $"{2 * _numViews}; got {batch}.", nameof(input));
        if (_inner is MultiViewTransformerLayer<T> transformer)
            return transformer.Forward(input, context, _numViews, CrossDomainTokens);
        var attention = (DiffusionAttention<T>)_inner;
        return PostProcess(attention.ForwardJointTokens(input, _numViews));
    }

    /// <summary>
    /// The cross-domain attention: <c>x + Attn(LayerNorm(x))</c> with view i of the normal half and view i of the color
    /// half joined into one sequence.
    /// </summary>
    /// <param name="innerOutput">The multi-view block's output, <c>[2M, C, H, W]</c>.</param>
    public Tensor<T> PostProcess(Tensor<T> innerOutput)
    {
        if (innerOutput is null) throw new ArgumentNullException(nameof(innerOutput));
        if (innerOutput.Shape.Length != 4 || innerOutput.Shape[0] % 2 != 0)
            throw new ArgumentException("Cross-domain attention takes [2M, C, H, W]: a normal half and a color half.",
                nameof(innerOutput));
        var e = Engine;
        int n = innerOutput.Shape[0], c = innerOutput.Shape[1], h = innerOutput.Shape[2], w = innerOutput.Shape[3];
        // [N, C, H, W] -> tokens [N, H·W, C].
        var tokens = e.Reshape(e.TensorPermute(innerOutput, new[] { 0, 2, 3, 1 }), new[] { n, h * w, c });
        var attended = CrossDomainTokens(tokens);
        return e.TensorPermute(e.Reshape(attended, new[] { n, h, w, c }), new[] { 0, 3, 1, 2 });
    }

    /// <summary>
    /// The cross-domain attention over tokens <c>[2M, HW, C]</c>: <c>x + Attn(LayerNorm(x))</c> with view i of the
    /// normal half and view i of the color half joined into one sequence.
    /// </summary>
    internal Tensor<T> CrossDomainTokens(Tensor<T> tokens)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Shape.Length != 3 || tokens.Shape[0] % 2 != 0)
            throw new ArgumentException("Cross-domain attention takes [2M, HW, C]: a normal half and a color half.", nameof(tokens));
        var e = Engine;
        int n = tokens.Shape[0], count = tokens.Shape[1], c = tokens.Shape[2];
        int half = n / 2;
        var normalized = _crossDomainNorm.Forward(e.Reshape(tokens, new[] { n * count, c }));
        // [2, M, HW, C] -> [M, 2, HW, C] -> [M, 2·HW, C]: the normal tokens of a view, then its color tokens.
        var paired = e.Reshape(
            e.TensorPermute(e.Reshape(normalized, new[] { 2, half, count, c }), new[] { 1, 0, 2, 3 }),
            new[] { half, 2 * count, c });
        var attended = _crossDomainAttention.Forward(paired);
        var split = e.Reshape(
            e.TensorPermute(e.Reshape(attended, new[] { half, 2, count, c }), new[] { 1, 0, 2, 3 }),
            new[] { n, count, c });
        return e.TensorAdd(tokens, split);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _inner.ResetState();
        _crossDomainNorm.ResetState();
        _crossDomainAttention.ResetState();
    }
}
