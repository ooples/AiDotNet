using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements one pre-normalized InternImage block with a deformable spatial
/// mixer and a GELU feed-forward network, each protected by a residual path.
/// </summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerCategory(LayerCategory.Residual)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 8, 8", TestConstructorArgs = "1, 1")]
[AutoParameters]
public sealed partial class InternImageBlockLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _groups;
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly DeformableConvolutionalLayer<T> _dcn;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly ConvolutionalLayer<T> _expand;
    private readonly ActivationLayer<T> _gelu;
    private readonly ConvolutionalLayer<T> _project;
    private readonly LayerBase<T>[] _parameterLayers;
    private readonly LayerBase<T>[] _allLayers;
    private Vector<T>? _pendingParameters;

    /// <summary>Creates an InternImage block for a fixed channel width.</summary>
    public InternImageBlockLayer(int channels, int groups = 1)
        : base([channels, -1, -1], [channels, -1, -1])
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (groups <= 0 || channels % groups != 0)
            throw new ArgumentOutOfRangeException(nameof(groups), groups, "Groups must divide channels.");

        _channels = channels;
        _groups = groups;
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        _norm1 = new LayerNormalizationLayer<T>(channels, epsilon: 1e-6);
        _dcn = new DeformableConvolutionalLayer<T>(
            channels, kernelSize: 3, stride: 1, padding: 1,
            groups: groups, deformGroups: groups, useModulation: true);
        _norm2 = new LayerNormalizationLayer<T>(channels, epsilon: 1e-6);
        _expand = new ConvolutionalLayer<T>(channels * 4, 1, 1, 0, identity);
        _gelu = new ActivationLayer<T>((IActivationFunction<T>)new GELUActivation<T>());
        _project = new ConvolutionalLayer<T>(channels, 1, 1, 0, identity);

        _parameterLayers = [_norm1, _dcn, _norm2, _expand, _project];
        _allLayers = [_norm1, _dcn, _norm2, _expand, _gelu, _project];
        foreach (var layer in _allLayers) RegisterSubLayer(layer);
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 3;
        var x = unbatched ? Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]) : input;
        if (x.Rank != 4 || x.Shape[1] != _channels)
            throw new ArgumentException($"InternImage block expects [B,{_channels},H,W].", nameof(input));
        if (!IsShapeResolved)
            ResolveShapes([_channels, x.Shape[2], x.Shape[3]], [_channels, x.Shape[2], x.Shape[3]]);

        if (_pendingParameters is not null)
        {
            // Lazy convolutions need the runtime spatial/channel shape before their
            // parameter tensors exist. Materialize once off-tape, replay the serialized
            // vector, then execute the real forward with the restored weights.
            using (var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                ForwardCore(x);
            var pending = _pendingParameters;
            _pendingParameters = null;
            SetParameters(pending);
        }

        var output = ForwardCore(x);
        return unbatched ? Engine.Reshape(output, [output.Shape[1], output.Shape[2], output.Shape[3]]) : output;
    }

    private Tensor<T> ForwardCore(Tensor<T> x)
    {
        var spatial = _dcn.Forward(NormalizeChannels(_norm1, x));
        var output = Engine.TensorAdd(x, spatial);

        var ffn = NormalizeChannels(_norm2, output);
        ffn = _gelu.Forward(_expand.Forward(ffn));
        ffn = _project.Forward(ffn);
        return Engine.TensorAdd(output, ffn);
    }

    private Tensor<T> NormalizeChannels(LayerNormalizationLayer<T> norm, Tensor<T> input)
    {
        var channelsLast = Engine.TensorPermute(input, [0, 2, 3, 1]);
        var normalized = norm.Forward(channelsLast);
        return Engine.TensorPermute(normalized, [0, 3, 1, 2]);
    }

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in _allLayers) layer.SetTrainingMode(isTraining);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameterGradients() => Concatenate(_parameterLayers, gradients: true);

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in _parameterLayers) layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in _parameterLayers) layer.ClearGradients();
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in _allLayers) layer.ResetState();
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString();
        metadata["Groups"] = _groups.ToString();
        return metadata;
    }

    private static Vector<T> Concatenate(IEnumerable<LayerBase<T>> layers, bool gradients)
    {
        var values = new List<T>();
        foreach (var layer in layers)
        {
            var vector = gradients ? layer.GetParameterGradients() : layer.GetParameters();
            for (int i = 0; i < vector.Length; i++) values.Add(vector[i]);
        }
        return new Vector<T>(values.ToArray());
    }
}
