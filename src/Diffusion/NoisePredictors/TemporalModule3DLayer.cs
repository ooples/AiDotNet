using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Initialization;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Released Upscale-A-Video temporal module: temporal 3D ResNet, inflated
/// spatial ResNet, and a zero-initialized residual projection.
/// </summary>
/// <remarks>
/// The released configuration disables both optional transformer-attention
/// branches. Consequently this layer contains the exact convolutional path and
/// remains independent of the number of frames supplied at runtime.
/// </remarks>
[ElementWiseShape(Note = "Residual temporal module preserves NCFHW shape at every frame count.")]
public sealed class TemporalModule3DLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _timeEmbeddingDim;
    private readonly int _spatialSize;
    private readonly GroupNormalizationLayer<T> _temporalNorm1;
    private readonly TemporalConv3DLayer<T> _temporalConv1;
    private readonly DenseLayer<T> _temporalTimeProjection;
    private readonly GroupNormalizationLayer<T> _temporalNorm2;
    private readonly TemporalConv3DLayer<T> _temporalConv2;
    private readonly DiffusionResBlock<T> _spatialResBlock;
    private readonly TemporalConv3DLayer<T> _shiftProjection;

    /// <summary>Gets the released first temporal kernel depth.</summary>
    public int FirstTemporalKernelDepth => _temporalConv1.KernelDepth;

    /// <summary>Gets the released second temporal kernel depth.</summary>
    public int SecondTemporalKernelDepth => _temporalConv2.KernelDepth;

    /// <summary>Gets whether optional transformer attention is present.</summary>
    public bool UsesTransformerAttention => false;

    /// <summary>Gets whether the residual output projection starts at zero.</summary>
    public bool UsesZeroInitializedOutputProjection => _shiftProjection.IsZeroInitialized;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the paper/released-code temporal module.</summary>
    public TemporalModule3DLayer(int channels, int timeEmbeddingDim, int spatialSize)
        : base([channels, -1, -1, -1], [channels, -1, -1, -1])
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (timeEmbeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(timeEmbeddingDim));
        if (spatialSize <= 0) throw new ArgumentOutOfRangeException(nameof(spatialSize));
        _channels = channels;
        _timeEmbeddingDim = timeEmbeddingDim;
        _spatialSize = spatialSize;

        int groups = ComputeGroups(channels, 32);
        _temporalNorm1 = new GroupNormalizationLayer<T>(groups, channels, 1e-6);
        _temporalConv1 = new TemporalConv3DLayer<T>(channels, channels, kernelDepth: 5);
        _temporalTimeProjection = new DenseLayer<T>(
            channels,
            (IActivationFunction<T>)new IdentityActivation<T>(),
            InitializationStrategies<T>.Lazy);
        _temporalTimeProjection.ResolveShapesOnly([timeEmbeddingDim]);
        _temporalNorm2 = new GroupNormalizationLayer<T>(groups, channels, 1e-6);
        _temporalConv2 = new TemporalConv3DLayer<T>(channels, channels, kernelDepth: 3);
        _spatialResBlock = new DiffusionResBlock<T>(
            channels, channels, spatialSize, timeEmbeddingDim, numGroups: 32, epsilon: 1e-6);
        _shiftProjection = new TemporalConv3DLayer<T>(
            channels, channels, kernelDepth: 1, zeroInitialize: true);

        RegisterSubLayer(_temporalNorm1);
        RegisterSubLayer(_temporalConv1);
        RegisterSubLayer(_temporalTimeProjection);
        RegisterSubLayer(_temporalNorm2);
        RegisterSubLayer(_temporalConv2);
        RegisterSubLayer(_spatialResBlock);
        RegisterSubLayer(_shiftProjection);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => throw new InvalidOperationException(
            "TemporalModule3DLayer requires an explicit diffusion timestep embedding.");

    /// <summary>Runs the temporal module with a projected diffusion timestep embedding.</summary>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> timeEmbedding)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (timeEmbedding is null) throw new ArgumentNullException(nameof(timeEmbedding));
        if (input.Rank != 5)
            throw new ArgumentException("TemporalModule3DLayer requires [B,C,F,H,W].", nameof(input));
        if (input.Shape[1] != _channels)
            throw new ArgumentException($"Expected {_channels} channels, got {input.Shape[1]}.", nameof(input));

        int batch = input.Shape[0];
        int frames = input.Shape[2];
        var timeBatch = NormalizeTimeEmbedding(timeEmbedding, batch);

        // ResnetBlock3DCNN: GN → SiLU → Conv(5,1,1) → time projection
        // → GN → SiLU → Conv(3,1,1) + identity.
        var hidden = NormalizeVideo(_temporalNorm1, input);
        hidden = Engine.TensorSiLU(hidden);
        hidden = _temporalConv1.Forward(hidden);
        var timeProjection = _temporalTimeProjection.Forward(Engine.TensorSiLU(timeBatch));
        hidden = Engine.TensorAdd(
            hidden, Engine.Reshape(timeProjection, [batch, _channels, 1, 1, 1]));
        hidden = NormalizeVideo(_temporalNorm2, hidden);
        hidden = Engine.TensorSiLU(hidden);
        hidden = _temporalConv2.Forward(hidden);
        hidden = Engine.TensorAdd(input, hidden);

        // Inflated spatial ResNet: flatten B×F, run the exact 2D diffusion
        // ResBlock, then restore NCFHW. No host per-frame loop and no fixed F.
        var frameBatch = FlattenFrames(hidden);
        var repeatedTime = RepeatTimeForFrames(timeBatch, frames);
        frameBatch = _spatialResBlock.Forward(frameBatch, repeatedTime);
        hidden = RestoreFrames(frameBatch, batch, frames);

        // Released module's zero_module(InflatedConv3d(1×1)) makes the whole
        // temporal branch an identity at initialization.
        hidden = _shiftProjection.Forward(hidden);
        var output = Engine.TensorAdd(input, hidden);
        ResolveShapes(
            [_channels, input.Shape[2], input.Shape[3], input.Shape[4]],
            [_channels, output.Shape[2], output.Shape[3], output.Shape[4]]);
        return output;
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
        var clone = new TemporalModule3DLayer<T>(_channels, _timeEmbeddingDim, _spatialSize);
        var parameters = GetParameters();
        if (parameters.Length > 0) clone.SetParameters(parameters);
        return clone;
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["TimeEmbeddingDim"] = _timeEmbeddingDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["SpatialSize"] = _spatialSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["AttentionBlockTypes"] = string.Empty;
        return metadata;
    }

    private Tensor<T> NormalizeVideo(GroupNormalizationLayer<T> norm, Tensor<T> video)
    {
        int batch = video.Shape[0];
        int frames = video.Shape[2];
        int height = video.Shape[3];
        int width = video.Shape[4];
        var folded = Engine.Reshape(video, [batch, _channels, frames * height, width]);
        var normalized = norm.Forward(folded);
        return Engine.Reshape(normalized, [batch, _channels, frames, height, width]);
    }

    private Tensor<T> NormalizeTimeEmbedding(Tensor<T> timeEmbedding, int batch)
    {
        Tensor<T> timeBatch = timeEmbedding.Rank switch
        {
            1 => Engine.Reshape(timeEmbedding, [1, timeEmbedding.Shape[0]]),
            2 => timeEmbedding,
            _ => throw new ArgumentException("Time embedding must be [D] or [B,D].", nameof(timeEmbedding))
        };
        if (timeBatch.Shape[1] != _timeEmbeddingDim)
            throw new ArgumentException(
                $"Expected time embedding width {_timeEmbeddingDim}, got {timeBatch.Shape[1]}.",
                nameof(timeEmbedding));
        if (timeBatch.Shape[0] == batch) return timeBatch;
        if (timeBatch.Shape[0] != 1)
            throw new ArgumentException(
                $"Expected one shared time embedding or batch {batch}, got {timeBatch.Shape[0]}.",
                nameof(timeEmbedding));
        return Engine.TensorBroadcastTo(timeBatch, [batch, _timeEmbeddingDim]);
    }

    private Tensor<T> FlattenFrames(Tensor<T> video)
    {
        int batch = video.Shape[0];
        int frames = video.Shape[2];
        var bfchw = Engine.TensorPermute(video, [0, 2, 1, 3, 4]).Contiguous();
        return Engine.Reshape(bfchw, [batch * frames, _channels, video.Shape[3], video.Shape[4]]);
    }

    private Tensor<T> RestoreFrames(Tensor<T> frameBatch, int batch, int frames)
    {
        var bfchw = Engine.Reshape(
            frameBatch, [batch, frames, _channels, frameBatch.Shape[2], frameBatch.Shape[3]]);
        return Engine.TensorPermute(bfchw, [0, 2, 1, 3, 4]).Contiguous();
    }

    private Tensor<T> RepeatTimeForFrames(Tensor<T> timeBatch, int frames)
    {
        int batch = timeBatch.Shape[0];
        var expanded = Engine.Reshape(timeBatch, [batch, 1, _timeEmbeddingDim]);
        expanded = Engine.TensorBroadcastTo(expanded, [batch, frames, _timeEmbeddingDim]);
        return Engine.Reshape(expanded, [batch * frames, _timeEmbeddingDim]);
    }

    private IEnumerable<ILayer<T>> ParameterLayers()
    {
        yield return _temporalNorm1;
        yield return _temporalConv1;
        yield return _temporalTimeProjection;
        yield return _temporalNorm2;
        yield return _temporalConv2;
        yield return _spatialResBlock;
        yield return _shiftProjection;
    }

    private static int ComputeGroups(int channels, int target)
    {
        for (int groups = System.Math.Min(channels, target); groups >= 1; groups--)
            if (channels % groups == 0) return groups;
        return 1;
    }
}
