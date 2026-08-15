using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// The released Upscale-A-Video Transformer3D block.
/// </summary>
/// <remarks>
/// It combines a temporal <c>(3,1,1)</c> ResNet, per-frame spatial/text
/// transformer processing, zero-initialized temporal attention, GEGLU, and an
/// outer residual. The implementation accepts arbitrary frame counts.
/// </remarks>
public sealed class VideoTransformer3DLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _contextDimension;
    private readonly int _headCount;
    private readonly int _spatialSize;
    private readonly bool _onlyCrossAttention;

    private readonly GroupNormalizationLayer<T> _temporalNorm1;
    private readonly TemporalConv3DLayer<T> _temporalConv1;
    private readonly GroupNormalizationLayer<T> _temporalNorm2;
    private readonly TemporalConv3DLayer<T> _temporalConv2;
    private readonly GroupNormalizationLayer<T> _outerNorm;
    private readonly DenseLayer<T> _projectionIn;
    private readonly LayerNormalizationLayer<T> _attentionNorm1;
    private readonly DiffusionAttentionLayer<T> _attention1;
    private readonly LayerNormalizationLayer<T> _attentionNorm2;
    private readonly DiffusionAttentionLayer<T> _crossAttention;
    private readonly LayerNormalizationLayer<T> _temporalAttentionNorm;
    private readonly DiffusionAttentionLayer<T> _temporalAttention;
    private readonly LayerNormalizationLayer<T> _feedForwardNorm;
    private readonly DenseLayer<T> _feedForwardGate;
    private readonly DenseLayer<T> _feedForwardValue;
    private readonly DenseLayer<T> _feedForwardDown;
    private readonly DenseLayer<T> _projectionOut;

    /// <summary>Gets whether the first attention sublayer also attends to text.</summary>
    public bool OnlyCrossAttention => _onlyCrossAttention;

    /// <summary>Gets whether the block contains the released temporal attention path.</summary>
    public bool UsesTemporalAttention => true;

    /// <summary>Gets whether temporal attention starts as a zero residual adapter.</summary>
    public bool TemporalAttentionIsZeroInitialized =>
        _temporalAttention.IsOutputProjectionZeroInitialized;

    /// <summary>
    /// Gets the temporal adapters optimized while the pretrained spatial transformer is frozen.
    /// </summary>
    public IReadOnlyList<ILayer<T>> TemporalTrainingLayers { get; }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override long ParameterCount => ParameterLayers().Sum(layer => layer.ParameterCount);

    /// <summary>Creates the released Transformer3D block.</summary>
    public VideoTransformer3DLayer(
        int channels,
        int contextDimension,
        int headCount,
        int spatialSize,
        bool onlyCrossAttention)
        : base([channels, -1, -1, -1], [channels, -1, -1, -1])
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (contextDimension <= 0) throw new ArgumentOutOfRangeException(nameof(contextDimension));
        if (headCount <= 0 || channels % headCount != 0)
            throw new ArgumentException("Head count must evenly divide channels.", nameof(headCount));
        if (spatialSize <= 0) throw new ArgumentOutOfRangeException(nameof(spatialSize));
        _channels = channels;
        _contextDimension = contextDimension;
        _headCount = headCount;
        _spatialSize = spatialSize;
        _onlyCrossAttention = onlyCrossAttention;

        int groups = ComputeGroups(channels, 32);
        _temporalNorm1 = new GroupNormalizationLayer<T>(groups, channels, 1e-6);
        _temporalConv1 = new TemporalConv3DLayer<T>(channels, channels, kernelDepth: 3);
        _temporalNorm2 = new GroupNormalizationLayer<T>(groups, channels, 1e-6);
        _temporalConv2 = new TemporalConv3DLayer<T>(channels, channels, kernelDepth: 3);
        _outerNorm = new GroupNormalizationLayer<T>(groups, channels, 1e-6);
        _projectionIn = Linear(channels, channels, new IdentityActivation<T>());

        _attentionNorm1 = new LayerNormalizationLayer<T>(channels);
        _attention1 = new DiffusionAttentionLayer<T>(
            channels, onlyCrossAttention ? contextDimension : channels, headCount);
        _attentionNorm2 = new LayerNormalizationLayer<T>(channels);
        _crossAttention = new DiffusionAttentionLayer<T>(channels, contextDimension, headCount);
        _temporalAttentionNorm = new LayerNormalizationLayer<T>(channels);
        _temporalAttention = new DiffusionAttentionLayer<T>(
            channels, channels, headCount, zeroOutputProjection: true);

        int feedForwardDimension = checked(channels * 4);
        _feedForwardNorm = new LayerNormalizationLayer<T>(channels);
        _feedForwardGate = Linear(channels, feedForwardDimension, new GELUActivation<T>());
        _feedForwardValue = Linear(channels, feedForwardDimension, new IdentityActivation<T>());
        _feedForwardDown = Linear(feedForwardDimension, channels, new IdentityActivation<T>());
        _projectionOut = Linear(channels, channels, new IdentityActivation<T>());

        TemporalTrainingLayers = Array.AsReadOnly<ILayer<T>>([
            _temporalNorm1,
            _temporalConv1,
            _temporalNorm2,
            _temporalConv2,
            _temporalAttentionNorm,
            _temporalAttention
        ]);

        foreach (var layer in ParameterLayers()) RegisterSubLayer(layer);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input) =>
        throw new InvalidOperationException(
            "VideoTransformer3DLayer requires text encoder states [B,N,contextDim].");

    /// <summary>Runs the block for an NCFHW video and text encoder states.</summary>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> context)
    {
        if (input.Rank != 5)
            throw new ArgumentException("VideoTransformer3DLayer requires [B,C,F,H,W].", nameof(input));
        if (input.Shape[1] != _channels)
            throw new ArgumentException($"Expected {_channels} channels, got {input.Shape[1]}.", nameof(input));

        int batch = input.Shape[0];
        int frames = input.Shape[2];
        int height = input.Shape[3];
        int width = input.Shape[4];
        var contextBatch = NormalizeContext(context, batch);

        // Transformer3DModel's VSR temporal ResNet before spatial projection.
        var hidden = NormalizeVideo(_temporalNorm1, input);
        hidden = Engine.TensorSiLU(hidden);
        hidden = _temporalConv1.Forward(hidden);
        hidden = NormalizeVideo(_temporalNorm2, hidden);
        hidden = Engine.TensorSiLU(hidden);
        hidden = _temporalConv2.Forward(hidden);
        hidden = Engine.TensorAdd(input, hidden);
        var outerResidual = hidden;

        // Per-frame GroupNorm and linear projection to spatial tokens.
        var frameBatch = FlattenFrames(hidden);
        frameBatch = _outerNorm.Forward(frameBatch);
        var bhwc = Engine.TensorPermute(frameBatch, [0, 2, 3, 1]).Contiguous();
        var tokens = Engine.Reshape(bhwc, [batch * frames, height * width, _channels]);
        tokens = _projectionIn.Forward(tokens);
        var repeatedContext = RepeatContextForFrames(contextBatch, frames);

        // attn1 is self-attention except at only_cross_attention stages.
        var normed1 = _attentionNorm1.Forward(tokens);
        var attention1 = _onlyCrossAttention
            ? _attention1.Forward(normed1, repeatedContext)
            : _attention1.Forward(normed1);
        tokens = Engine.TensorAdd(tokens, attention1);

        // The released BasicTransformerBlock always has the second text cross attention.
        var normed2 = _attentionNorm2.Forward(tokens);
        tokens = Engine.TensorAdd(tokens, _crossAttention.Forward(normed2, repeatedContext));

        // Zero-initialized temporal attention at every spatial token.
        var bfs = Engine.Reshape(tokens, [batch, frames, height * width, _channels]);
        var bsf = Engine.TensorPermute(bfs, [0, 2, 1, 3]).Contiguous();
        var temporalTokens = Engine.Reshape(
            bsf, [batch * height * width, frames, _channels]);
        var temporalNormed = _temporalAttentionNorm.Forward(temporalTokens);
        temporalTokens = Engine.TensorAdd(
            temporalTokens, _temporalAttention.Forward(temporalNormed));
        bsf = Engine.Reshape(
            temporalTokens, [batch, height * width, frames, _channels]);
        bfs = Engine.TensorPermute(bsf, [0, 2, 1, 3]).Contiguous();
        tokens = Engine.Reshape(bfs, [batch * frames, height * width, _channels]);

        // GEGLU feed-forward residual.
        var feedForwardInput = _feedForwardNorm.Forward(tokens);
        var flatFeedForward = Engine.Reshape(
            feedForwardInput, [batch * frames * height * width, _channels]);
        var gate = _feedForwardGate.Forward(flatFeedForward);
        var value = _feedForwardValue.Forward(flatFeedForward);
        var feedForward = _feedForwardDown.Forward(Engine.TensorMultiply(gate, value));
        feedForward = Engine.Reshape(
            feedForward, [batch * frames, height * width, _channels]);
        tokens = Engine.TensorAdd(tokens, feedForward);

        tokens = _projectionOut.Forward(tokens);
        bhwc = Engine.Reshape(tokens, [batch * frames, height, width, _channels]);
        frameBatch = Engine.TensorPermute(bhwc, [0, 3, 1, 2]).Contiguous();
        hidden = RestoreFrames(frameBatch, batch, frames);
        var output = Engine.TensorAdd(outerResidual, hidden);
        ResolveShapes(
            [_channels, frames, height, width],
            [_channels, frames, height, width]);
        return output;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var values = new List<T>();
        foreach (var layer in ParameterLayers())
        {
            var parameters = layer.GetParameters();
            for (int i = 0; i < parameters.Length; i++) values.Add(parameters[i]);
        }
        return new Vector<T>(values.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, got {parameters.Length}.", nameof(parameters));
        int offset = 0;
        foreach (var layer in ParameterLayers())
        {
            int count = checked((int)layer.ParameterCount);
            var slice = new Vector<T>(count);
            for (int i = 0; i < count; i++) slice[i] = parameters[offset + i];
            layer.SetParameters(slice);
            offset += count;
        }
    }

    /// <inheritdoc />
    public override Vector<T> GetParameterGradients()
    {
        var values = new List<T>();
        foreach (var layer in ParameterLayers())
        {
            var gradients = layer.GetParameterGradients();
            for (int i = 0; i < gradients.Length; i++) values.Add(gradients[i]);
        }
        return new Vector<T>(values.ToArray());
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
    public override LayerBase<T> Clone()
    {
        var clone = new VideoTransformer3DLayer<T>(
            _channels, _contextDimension, _headCount, _spatialSize, _onlyCrossAttention);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ContextDimension"] = _contextDimension.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["HeadCount"] = _headCount.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["SpatialSize"] = _spatialSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["OnlyCrossAttention"] = _onlyCrossAttention.ToString();
        return metadata;
    }

    private DenseLayer<T> Linear(
        int inputDimension, int outputDimension, IActivationFunction<T> activation)
    {
        var layer = new DenseLayer<T>(outputDimension, activation, InitializationStrategies<T>.Lazy);
        layer.ResolveShapesOnly([inputDimension]);
        return layer;
    }

    private Tensor<T> NormalizeVideo(GroupNormalizationLayer<T> norm, Tensor<T> video)
    {
        int batch = video.Shape[0];
        int frames = video.Shape[2];
        int height = video.Shape[3];
        int width = video.Shape[4];
        var folded = Engine.Reshape(video, [batch, _channels, frames * height, width]);
        return Engine.Reshape(
            norm.Forward(folded), [batch, _channels, frames, height, width]);
    }

    private Tensor<T> FlattenFrames(Tensor<T> video)
    {
        int batch = video.Shape[0];
        int frames = video.Shape[2];
        var bfchw = Engine.TensorPermute(video, [0, 2, 1, 3, 4]).Contiguous();
        return Engine.Reshape(
            bfchw, [batch * frames, _channels, video.Shape[3], video.Shape[4]]);
    }

    private Tensor<T> RestoreFrames(Tensor<T> frameBatch, int batch, int frames)
    {
        var bfchw = Engine.Reshape(
            frameBatch, [batch, frames, _channels, frameBatch.Shape[2], frameBatch.Shape[3]]);
        return Engine.TensorPermute(bfchw, [0, 2, 1, 3, 4]).Contiguous();
    }

    private Tensor<T> NormalizeContext(Tensor<T> context, int batch)
    {
        Tensor<T> contextBatch = context.Rank switch
        {
            2 => Engine.Reshape(context, [1, context.Shape[0], context.Shape[1]]),
            3 => context,
            _ => throw new ArgumentException("Text context must be [N,D] or [B,N,D].", nameof(context))
        };
        if (contextBatch.Shape[2] != _contextDimension)
            throw new ArgumentException(
                $"Expected text width {_contextDimension}, got {contextBatch.Shape[2]}.", nameof(context));
        if (contextBatch.Shape[0] == batch) return contextBatch;
        if (contextBatch.Shape[0] != 1)
            throw new ArgumentException(
                $"Expected one shared context or batch {batch}, got {contextBatch.Shape[0]}.", nameof(context));
        return Engine.TensorBroadcastTo(
            contextBatch, [batch, contextBatch.Shape[1], _contextDimension]);
    }

    private Tensor<T> RepeatContextForFrames(Tensor<T> context, int frames)
    {
        int batch = context.Shape[0];
        int sequence = context.Shape[1];
        var expanded = Engine.Reshape(context, [batch, 1, sequence, _contextDimension]);
        expanded = Engine.TensorBroadcastTo(
            expanded, [batch, frames, sequence, _contextDimension]);
        return Engine.Reshape(expanded, [batch * frames, sequence, _contextDimension]);
    }

    private IEnumerable<ILayer<T>> ParameterLayers()
    {
        yield return _temporalNorm1;
        yield return _temporalConv1;
        yield return _temporalNorm2;
        yield return _temporalConv2;
        yield return _outerNorm;
        yield return _projectionIn;
        yield return _attentionNorm1;
        yield return _attention1;
        yield return _attentionNorm2;
        yield return _crossAttention;
        yield return _temporalAttentionNorm;
        yield return _temporalAttention;
        yield return _feedForwardNorm;
        yield return _feedForwardGate;
        yield return _feedForwardValue;
        yield return _feedForwardDown;
        yield return _projectionOut;
    }

    private static int ComputeGroups(int channels, int target)
    {
        for (int groups = System.Math.Min(channels, target); groups >= 1; groups--)
            if (channels % groups == 0) return groups;
        return 1;
    }
}
