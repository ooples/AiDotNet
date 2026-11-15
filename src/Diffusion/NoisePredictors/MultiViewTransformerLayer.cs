using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// MVDream's <c>SpatialTransformer3D</c>: the U-Net transformer whose self-attention runs over the tokens of all the
/// views of one object at once.
/// </summary>
/// <remarks>
/// <para>
/// Block for block the official <c>SpatialTransformer3D</c> / <c>BasicTransformerBlock3D</c>
/// (<c>bytedance/MVDream</c>, <c>mvdream/ldm/modules/attention.py</c>, with <c>use_linear_in_transformer</c>), over
/// <c>x [B·F, C, H, W]</c> holding F views per object:
/// </para>
/// <code>
/// h = proj_in(GroupNorm32(x))                        as [B·F, HW, C] tokens
/// per block:
///   h = (B·F, HW, C) -> (B, F·HW, C);  h = attn1(norm1(h)) + h;  -> (B·F, HW, C)   3D self-attention
///   h = attn2(norm2(h), context) + h                                                  per-view cross-attention
///   h = ff(norm3(h)) + h                                                              GEGLU, 4C inner width
/// out = x + proj_out(h)                              proj_out starts at zero (zero_module)
/// </code>
/// <para>
/// With image-prompt tokens the cross-attention is ImageDream's decoupled attention
/// (<see cref="ImagePromptCrossAttentionLayer{T}"/>), whose context is the text tokens followed by the image tokens.
/// Without a context the cross-attention is skipped, as the shared U-Net skips its cross-attention when it has no
/// conditioning.
/// </para>
/// </remarks>
[ElementWiseShape(Note = "SpatialTransformer3D is a residual block; it preserves [B, C, H, W].")]
internal sealed partial class MultiViewTransformerLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _contextDimension;
    private readonly int _headCount;
    private readonly int _framesPerObject;
    private readonly int _imageTokens;
    private readonly SpatialTransformerProjection _projection;

    private readonly GroupNormalizationLayer<T> _norm;
    private readonly DenseLayer<T> _projectionIn;
    private readonly List<Block> _blocks = new();
    private readonly DenseLayer<T> _projectionOut;

    /// <summary>One <c>BasicTransformerBlock3D</c>.</summary>
    internal sealed class Block
    {
        public Block(LayerNormalizationLayer<T> norm1, DiffusionAttentionLayer<T> attention1, LayerNormalizationLayer<T> norm2,
            ILayer<T> attention2, LayerNormalizationLayer<T> norm3, DenseLayer<T> feedForwardIn, DenseLayer<T> feedForwardOut)
        {
            Norm1 = norm1;
            Attention1 = attention1;
            Norm2 = norm2;
            Attention2 = attention2;
            Norm3 = norm3;
            FeedForwardIn = feedForwardIn;
            FeedForwardOut = feedForwardOut;
        }

        public LayerNormalizationLayer<T> Norm1 { get; }

        /// <summary>The 3D self-attention (<c>attn1</c>).</summary>
        public DiffusionAttentionLayer<T> Attention1 { get; }

        public LayerNormalizationLayer<T> Norm2 { get; }

        /// <summary>The per-view cross-attention (<c>attn2</c>): a <see cref="DiffusionAttentionLayer{T}"/>, or an
        /// <see cref="ImagePromptCrossAttentionLayer{T}"/> with image-prompt tokens.</summary>
        public ILayer<T> Attention2 { get; }

        public LayerNormalizationLayer<T> Norm3 { get; }

        /// <summary>GEGLU's <c>ff.net.0.proj</c>: C to 2 · 4C (value then gate).</summary>
        public DenseLayer<T> FeedForwardIn { get; }

        /// <summary><c>ff.net.2</c>: 4C to C.</summary>
        public DenseLayer<T> FeedForwardOut { get; }
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>The transformer blocks, in order.</summary>
    internal IReadOnlyList<Block> Blocks => _blocks;

    /// <summary>The number of attention heads.</summary>
    public int HeadCount => _headCount;

    /// <param name="channels">The feature width C (LDM's <c>in_channels</c>, and the inner width n_heads · d_head).</param>
    /// <param name="contextDimension">The cross-attention context width; zero for self-attention in <c>attn2</c>.</param>
    /// <param name="options">The multi-view options (heads, depth, frames per object, image-prompt tokens).</param>
    public MultiViewTransformerLayer(int channels, int contextDimension, MultiViewUNetOptions options)
        : base([channels, -1, -1], [channels, -1, -1])
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (contextDimension < 0) throw new ArgumentOutOfRangeException(nameof(contextDimension));
        if (options is null) throw new ArgumentNullException(nameof(options));
        options.Validate();
        _channels = channels;
        _contextDimension = contextDimension;
        _headCount = options.HeadsFor(channels);
        _framesPerObject = options.FramesPerObject;
        _imageTokens = contextDimension > 0 ? options.ImagePromptTokens : 0;
        _projection = options.Projection;

        // LDM Normalize: GroupNorm(32, C, eps=1e-6); a width that 32 does not divide takes the largest group count
        // that does, as the shared U-Net's other norms do.
        _norm = new GroupNormalizationLayer<T>(DiffusionResBlock<T>.ComputeNumGroups(channels, 32), channels, 1e-6);
        _projectionIn = ImagePromptCrossAttentionLayer<T>.Linear(channels, channels, bias: true);
        int attendedContext = contextDimension > 0 ? contextDimension : channels;
        int inner = checked(channels * 4);
        for (int d = 0; d < options.TransformerDepth; d++)
        {
            ILayer<T> attention2 = _imageTokens > 0
                ? new ImagePromptCrossAttentionLayer<T>(channels, attendedContext, _headCount, _imageTokens, options.ImagePromptWeight)
                : new DiffusionAttentionLayer<T>(channels, attendedContext, _headCount,
                    outputBiasInitialization: AttentionBiasInitialization.TorchDefault);
            _blocks.Add(new Block(
                new LayerNormalizationLayer<T>(channels),
                new DiffusionAttentionLayer<T>(channels, channels, _headCount,
                    outputBiasInitialization: AttentionBiasInitialization.TorchDefault),
                new LayerNormalizationLayer<T>(channels),
                attention2,
                new LayerNormalizationLayer<T>(channels),
                ImagePromptCrossAttentionLayer<T>.Linear(channels, checked(inner * 2), bias: true),
                ImagePromptCrossAttentionLayer<T>.Linear(inner, channels, bias: true)));
        }

        // zero_module(nn.Linear): the transformer starts as the identity.
        _projectionOut = new DenseLayer<T>(channels, (IActivationFunction<T>?)null, InitializationStrategies<T>.Zero);
        _projectionOut.ResolveShapesOnly([channels]);

        foreach (var layer in ParameterLayers()) RegisterSubLayer(layer);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Forward(input, null);

    /// <summary>
    /// Runs the transformer over <paramref name="input"/> <c>[B·F, C, H, W]</c> with an optional cross-attention
    /// <paramref name="context"/> <c>[B·F or 1, L, contextDimension]</c>.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T>? context)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 4)
            throw new ArgumentException("MultiViewTransformerLayer requires [B·F, C, H, W].", nameof(input));
        if (input.Shape[1] != _channels)
            throw new ArgumentException($"Expected {_channels} channels, got {input.Shape[1]}.", nameof(input));

        int batch = input.Shape[0], height = input.Shape[2], width = input.Shape[3];
        int positions = height * width;
        int frames = _framesPerObject > 0 ? _framesPerObject : batch;
        if (batch % frames != 0)
            throw new ArgumentException(
                $"The batch ({batch}) must be a whole number of objects of {frames} views (MVDream's num_frames).", nameof(input));
        int objects = batch / frames;

        var tokens = Engine.TensorPermute(_norm.Forward(input), [0, 2, 3, 1]).Contiguous();
        tokens = Engine.Reshape(tokens, [batch, positions, _channels]);
        tokens = _projectionIn.Forward(tokens);
        var crossContext = context is null ? null : ContextFor(context, batch);

        foreach (var block in _blocks)
        {
            // "(b f) l c -> b (f l) c": the frames of one object are consecutive in the batch, so this is a reshape.
            var joined = Engine.Reshape(tokens, [objects, frames * positions, _channels]);
            var normed1 = block.Norm1.Forward(joined);
            joined = Engine.TensorAdd(block.Attention1.Forward(normed1, normed1), joined);
            tokens = Engine.Reshape(joined, [batch, positions, _channels]);

            if (crossContext is not null || _contextDimension == 0)
            {
                var normed2 = block.Norm2.Forward(tokens);
                var attended = block.Attention2 switch
                {
                    ImagePromptCrossAttentionLayer<T> decoupled => decoupled.Forward(normed2, crossContext ?? normed2),
                    DiffusionAttentionLayer<T> plain => plain.Forward(normed2, crossContext ?? normed2),
                    _ => throw new InvalidOperationException("Unexpected cross-attention layer.")
                };
                tokens = Engine.TensorAdd(attended, tokens);
            }

            tokens = Engine.TensorAdd(FeedForward(block, block.Norm3.Forward(tokens), batch, positions), tokens);
        }

        tokens = _projectionOut.Forward(tokens);
        var spatial = Engine.TensorPermute(Engine.Reshape(tokens, [batch, height, width, _channels]), [0, 3, 1, 2]).Contiguous();
        var output = Engine.TensorAdd(spatial, input);
        ResolveShapes([_channels, height, width], [_channels, height, width]);
        return output;
    }

    // GEGLU (LDM attention.py): proj(x).chunk(2) -> value * gelu(gate), exact erf GELU, then the 4C -> C projection.
    private Tensor<T> FeedForward(Block block, Tensor<T> x, int batch, int positions)
    {
        int inner = _channels * 4;
        var rows = Engine.Reshape(x, [batch * positions, _channels]);
        var projected = block.FeedForwardIn.Forward(rows);
        var value = Engine.TensorNarrow(projected, 1, 0, inner).Contiguous();
        var gate = Engine.TensorNarrow(projected, 1, inner, inner).Contiguous();
        var gated = Engine.TensorMultiply(value, ActivationHelper.ExactGelu(gate, Engine));
        return Engine.Reshape(block.FeedForwardOut.Forward(gated), [batch, positions, _channels]);
    }

    private Tensor<T> ContextFor(Tensor<T> context, int batch)
    {
        var batched = context.Rank switch
        {
            2 => Engine.Reshape(context, [1, context.Shape[0], context.Shape[1]]),
            3 => context,
            _ => throw new ArgumentException("The context must be [L, D] or [B, L, D].", nameof(context))
        };
        if (_contextDimension > 0 && batched.Shape[2] != _contextDimension)
            throw new ArgumentException($"Expected context width {_contextDimension}, got {batched.Shape[2]}.", nameof(context));
        if (batched.Shape[0] == batch) return batched;
        if (batched.Shape[0] != 1)
            throw new ArgumentException($"Expected one shared context or one per view ({batch}), got {batched.Shape[0]}.", nameof(context));
        return Engine.TensorBroadcastTo(batched, [batch, batched.Shape[1], batched.Shape[2]]).Contiguous();
    }

    /// <summary>
    /// Loads the official names under <paramref name="prefix"/> (for example
    /// <c>model.diffusion_model.input_blocks.1.1.</c>): <c>norm</c>, <c>proj_in</c>,
    /// <c>transformer_blocks.{d}.{norm1, attn1, norm2, attn2, norm3, ff.net.0.proj, ff.net.2}</c> and <c>proj_out</c>.
    /// </summary>
    internal IReadOnlyList<string> LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source, string prefix)
        => LoadTorchWeightsCore(source, prefix);

    /// <summary>The names <see cref="LoadTorchWeights"/> reads under <paramref name="prefix"/>, in order.</summary>
    internal IReadOnlyList<string> LdmTensorNames(string prefix)
    {
        var names = new List<string>();
        void Attention(string name, bool decoupled)
        {
            names.Add(name + "to_q.weight");
            names.Add(name + "to_k.weight");
            names.Add(name + "to_v.weight");
            if (decoupled)
            {
                names.Add(name + "to_k_ip.weight");
                names.Add(name + "to_v_ip.weight");
            }
            names.Add(name + "to_out.0.weight");
            names.Add(name + "to_out.0.bias");
        }

        names.AddRange(LdmCheckpoint.Pair(prefix + "norm"));
        names.AddRange(LdmCheckpoint.Pair(prefix + "proj_in"));
        for (int d = 0; d < _blocks.Count; d++)
        {
            string b = $"{prefix}transformer_blocks.{d}.";
            names.AddRange(LdmCheckpoint.Pair(b + "norm1"));
            Attention(b + "attn1.", decoupled: false);
            names.AddRange(LdmCheckpoint.Pair(b + "norm2"));
            Attention(b + "attn2.", _blocks[d].Attention2 is ImagePromptCrossAttentionLayer<T>);
            names.AddRange(LdmCheckpoint.Pair(b + "norm3"));
            names.AddRange(LdmCheckpoint.Pair(b + "ff.net.0.proj"));
            names.AddRange(LdmCheckpoint.Pair(b + "ff.net.2"));
        }
        names.AddRange(LdmCheckpoint.Pair(prefix + "proj_out"));
        return names;
    }

    private IReadOnlyList<string> LoadTorchWeightsCore(AiDotNet.ModelLoading.TorchTensorSource source, string prefix)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        var engine = AiDotNetEngine.Current;
        var names = new List<string>();
        void GroupNorm(GroupNormalizationLayer<T> norm, string name)
        {
            var gamma = norm.GetGammaTensor();
            var beta = norm.GetBetaTensor();
            source.CopyTo(name + ".weight", new[] { gamma.Length }, gamma);
            source.CopyTo(name + ".bias", new[] { beta.Length }, beta);
            engine.InvalidatePersistentTensor(gamma);
            engine.InvalidatePersistentTensor(beta);
            names.Add(name + ".weight");
            names.Add(name + ".bias");
        }
        void LayerNorm(LayerNormalizationLayer<T> norm, string name)
        {
            AiDotNet.TextToSpeech.CodecBased.TorchParameters.LayerNorm(engine, norm, source, name + ".weight", name + ".bias");
            names.Add(name + ".weight");
            names.Add(name + ".bias");
        }
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs, int[]? weightShape = null)
        {
            if (weightShape is null)
                ImagePromptCrossAttentionLayer<T>.TorchParametersLinear(engine, layer, inputs, outputs, source, name + ".weight", name + ".bias");
            else
                AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(engine, layer, inputs, outputs, source, name + ".weight", name + ".bias", weightShape);
            names.Add(name + ".weight");
            names.Add(name + ".bias");
        }
        // SD 1.x's proj_in / proj_out are 1x1 convolutions: the same [C, C] map stored as [C, C, 1, 1].
        int[]? projectionShape = _projection == SpatialTransformerProjection.Convolution
            ? new[] { _channels, _channels, 1, 1 }
            : null;

        int inner = _channels * 4;
        GroupNorm(_norm, prefix + "norm");
        Linear(_projectionIn, prefix + "proj_in", _channels, _channels, projectionShape);
        for (int d = 0; d < _blocks.Count; d++)
        {
            var block = _blocks[d];
            string b = $"{prefix}transformer_blocks.{d}.";
            LayerNorm(block.Norm1, b + "norm1");
            names.AddRange(block.Attention1.LoadTorchWeights(source, b + "attn1."));
            LayerNorm(block.Norm2, b + "norm2");
            names.AddRange(block.Attention2 switch
            {
                ImagePromptCrossAttentionLayer<T> decoupled => decoupled.LoadTorchWeights(source, b + "attn2."),
                DiffusionAttentionLayer<T> plain => plain.LoadTorchWeights(source, b + "attn2."),
                _ => throw new InvalidOperationException("Unexpected cross-attention layer.")
            });
            LayerNorm(block.Norm3, b + "norm3");
            Linear(block.FeedForwardIn, b + "ff.net.0.proj", _channels, inner * 2);
            Linear(block.FeedForwardOut, b + "ff.net.2", inner, _channels);
        }
        Linear(_projectionOut, prefix + "proj_out", _channels, _channels, projectionShape);
        return names;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in ParameterLayers()) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in ParameterLayers()) layer.SetTrainingMode(isTraining);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in ParameterLayers()) layer.ResetState();
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        metadata["Channels"] = _channels.ToString(invariant);
        metadata["ContextDimension"] = _contextDimension.ToString(invariant);
        metadata["HeadCount"] = _headCount.ToString(invariant);
        metadata["Depth"] = _blocks.Count.ToString(invariant);
        metadata["FramesPerObject"] = _framesPerObject.ToString(invariant);
        metadata["ImageTokens"] = _imageTokens.ToString(invariant);
        return metadata;
    }

    private IEnumerable<ILayer<T>> ParameterLayers()
    {
        yield return _norm;
        yield return _projectionIn;
        foreach (var block in _blocks)
        {
            yield return block.Norm1;
            yield return block.Attention1;
            yield return block.Norm2;
            yield return block.Attention2;
            yield return block.Norm3;
            yield return block.FeedForwardIn;
            yield return block.FeedForwardOut;
        }
        yield return _projectionOut;
    }
}
