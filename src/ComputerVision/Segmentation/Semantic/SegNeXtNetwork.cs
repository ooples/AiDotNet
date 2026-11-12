using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>SegNeXt's network configuration (Guo et al. 2022; mmsegmentation's MSCAN and LightHamHead).</summary>
internal sealed record SegNeXtConfiguration
{
    public int InputChannels { get; init; } = 3;

    /// <summary>MSCAN's stage widths (Tiny: 32/64/160/256).</summary>
    public int[] EmbedDims { get; init; } = { 32, 64, 160, 256 };

    /// <summary>MSCA blocks per stage (Tiny: 3/3/5/2).</summary>
    public int[] Depths { get; init; } = { 3, 3, 5, 2 };

    /// <summary>The MLP's expansion per stage (8/8/4/4).</summary>
    public int[] MlpRatios { get; init; } = { 8, 8, 4, 4 };

    /// <summary>Stochastic depth, rising linearly from 0 over all blocks to this rate (0.1; Large 0.3).</summary>
    public double DropPathRate { get; init; } = 0.1;

    /// <summary>Dropout inside the MLP (MSCAN's drop_rate, 0).</summary>
    public double MlpDropout { get; init; }

    /// <summary>The head's channels after the align convolution (256; Base 512, Large 1024).</summary>
    public int HeadChannels { get; init; } = 256;

    /// <summary>The Hamburger's channels (equal to <see cref="HeadChannels"/> in every official config).</summary>
    public int HamChannels { get; init; } = 256;

    public int Classes { get; init; } = 150;

    /// <summary>NMF's rank R (MD_R, 16).</summary>
    public int MatrixRank { get; init; } = 16;

    /// <summary>NMF multiplicative-update steps in training (6) and evaluation (7).</summary>
    public int TrainSteps { get; init; } = 6;
    public int EvalSteps { get; init; } = 7;

    /// <summary>The head's GroupNorm groups (32).</summary>
    public int NormGroups { get; init; } = 32;

    /// <summary>Dropout2d before the classifier (0.1).</summary>
    public double HeadDropout { get; init; } = 0.1;
}

/// <summary>A depth-wise convolution with a rectangular kernel and per-axis padding (MSCA's strip convolutions).</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 2, 5, 5", TestConstructorArgs = "2, 1, 3, 0, 1")]
[ElementWiseShape(Note = "Same-padded depth-wise convolution: every MSCA strip pads (k - 1) / 2 on its long axis, so the map keeps its shape.")]
[AutoParameters]
internal sealed partial class StripDepthwiseConvolution<T> : LayerBase<T>
{
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _weight;

    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;

    private readonly int _channels, _kernelHeight, _kernelWidth, _padHeight, _padWidth;

    public override bool SupportsTraining => true;

    public StripDepthwiseConvolution(int channels, int kernelHeight, int kernelWidth, int padHeight, int padWidth)
        : base([channels], [channels])
    {
        _channels = channels;
        _kernelHeight = kernelHeight;
        _kernelWidth = kernelWidth;
        _padHeight = padHeight;
        _padWidth = padWidth;
        _weight = new Tensor<T>([channels, 1, kernelHeight, kernelWidth]);
        _bias = new Tensor<T>([channels]);
        // A depth-wise kernel's fan-in and fan-out are both its area; SegNeXtNetwork replaces these with the official
        // initialization or a checkpoint.
        InitializeLayerWeights(_weight, kernelHeight * kernelWidth, kernelHeight * kernelWidth);
        RegisterTrainableParameter(_weight, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    public int KernelArea => _kernelHeight * _kernelWidth;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 4 || input.Shape[1] != _channels)
            throw new ArgumentException($"Expected [B, {_channels}, H, W], got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        var y = Engine.DepthwiseConv2D(input, _weight, [1, 1], [_padHeight, _padWidth]);
        return Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(_bias, [1, _channels, 1, 1]), y.Shape.ToArray()));
    }

    public void Load(double[] weight, double[] bias)
    {
        for (int i = 0; i < weight.Length; i++) _weight[i] = NumOps.FromDouble(weight[i]);
        for (int i = 0; i < bias.Length; i++) _bias[i] = NumOps.FromDouble(bias[i]);
        Engine.InvalidatePersistentTensor(_weight);
        Engine.InvalidatePersistentTensor(_bias);
    }

    public override void UpdateParameters(T learningRate)
    {
        var gradient = GetParameterGradients();
        int w = _weight.Length;
        if (gradient.Length != w + _bias.Length) return;
        for (int i = 0; i < w; i++) _weight[i] = NumOps.Subtract(_weight[i], NumOps.Multiply(learningRate, gradient[i]));
        for (int i = 0; i < _bias.Length; i++) _bias[i] = NumOps.Subtract(_bias[i], NumOps.Multiply(learningRate, gradient[w + i]));
        Engine.InvalidatePersistentTensor(_weight);
        Engine.InvalidatePersistentTensor(_bias);
    }

    public override void ResetState()
    {
    }
}

/// <summary>A learned per-channel scale (MSCA's layer scale, initialized to 0.01).</summary>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 2, 3, 3", TestConstructorArgs = "2")]
[ElementWiseShape(Note = "A per-channel scale changes no axis.")]
[AutoParameters]
internal sealed partial class ChannelScale<T> : LayerBase<T>
{
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _scale;

    private readonly int _channels;

    public override bool SupportsTraining => true;

    /// <param name="channels">The channel count.</param>
    /// <param name="initialValue">Every channel's starting scale (MSCA's layer_scale_init_value, 0.01).</param>
    public ChannelScale(int channels, double initialValue = 0.01)
        : base([channels], [channels])
    {
        _channels = channels;
        _scale = Tensor<T>.CreateDefault([channels], NumOps.FromDouble(initialValue));
        RegisterTrainableParameter(_scale, PersistentTensorRole.Weights);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 4 || input.Shape[1] != _channels)
            throw new ArgumentException($"Expected [B, {_channels}, H, W], got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        return Engine.TensorMultiply(input, Engine.TensorBroadcastTo(Engine.Reshape(_scale, [1, _channels, 1, 1]), input.Shape.ToArray()));
    }

    public void Load(double[] values)
    {
        for (int i = 0; i < values.Length; i++) _scale[i] = NumOps.FromDouble(values[i]);
        Engine.InvalidatePersistentTensor(_scale);
    }

    public override void UpdateParameters(T learningRate)
    {
        var gradient = GetParameterGradients();
        if (gradient.Length != _channels) return;
        for (int i = 0; i < _channels; i++) _scale[i] = NumOps.Subtract(_scale[i], NumOps.Multiply(learningRate, gradient[i]));
        Engine.InvalidatePersistentTensor(_scale);
    }

    public override void ResetState()
    {
    }
}
