using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using System.Linq;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// VGG-16 with batch normalization as a feature-pyramid backbone, cut the way CRAFT cuts it.
/// </summary>
/// <remarks>
/// <para>
/// Thirteen 3x3 convolutions in five blocks (64, 128, 256, 512, 512 channels), each followed by batch norm
/// and ReLU, with 2x2 max pooling after the first four blocks (Simonyan and Zisserman 2015, configuration D).
/// CRAFT (Baek et al. 2019) replaces pool5 and the classifier with fc6/fc7 as convolutions: a 3x3 stride-1
/// max pool, a 3x3 convolution with dilation 6 to 1024 channels, and a 1x1 convolution to 1024 channels.
/// </para>
/// <para>
/// Features are returned finest first, [relu2_2, relu3_3, relu4_3, relu5_3, fc7], at strides 2, 4, 8, 16 and
/// 16, the five taps CRAFT's U-Net decoder merges.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Very Deep Convolutional Networks for Large-Scale Image Recognition",
    "https://arxiv.org/abs/1409.1556",
    Year = 2015,
    Authors = "Karen Simonyan, Andrew Zisserman")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class VGG16BNBackbone<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private static readonly int[][] BlockChannels =
    {
        new[] { 64, 64 },
        new[] { 128, 128 },
        new[] { 256, 256, 256 },
        new[] { 512, 512, 512 },
        new[] { 512, 512, 512 },
    };

    private readonly List<(ConvolutionalLayer<T> Conv, BatchNormalizationLayer<T> Norm)[]> _blocks;
    private readonly List<MaxPoolingLayer<T>> _pools;
    private readonly DilatedConvolutionalLayer<T> _fc6;
    private readonly ConvolutionalLayer<T> _fc7;
    private readonly IActivationFunction<T> _relu = new ReLUActivation<T>();

    /// <summary>Whether the backbone's weights are frozen.</summary>
    public bool IsFrozen { get; private set; }

    /// <summary>The backbone name.</summary>
    public string Name => "VGG16-BN";

    /// <summary>Channels of each returned feature map, finest first.</summary>
    public IReadOnlyList<int> OutputChannels { get; } = new[] { 128, 256, 512, 512, 1024 };

    /// <summary>Stride of each returned feature map, finest first.</summary>
    public IReadOnlyList<int> Strides => new[] { 2, 4, 8, 16, 16 };

    /// <summary>Creates the backbone.</summary>
    /// <param name="options">Input channels; defaults to three.</param>
    public VGG16BNBackbone(VGG16BNBackboneOptions? options = null)
        : base(DetectionBackboneArchitecture<T>.Create((options ??= new VGG16BNBackboneOptions()).InChannels),
              new MeanSquaredErrorLoss<T>())
    {
        options.Validate();
        _blocks = BlockChannels
            .Select(block => block
                .Select(channels => (new ConvolutionalLayer<T>(channels, 3, 1, 1, (IActivationFunction<T>?)null),
                                     new BatchNormalizationLayer<T>()))
                .ToArray())
            .ToList();
        _pools = Enumerable.Range(0, 4).Select(_ => new MaxPoolingLayer<T>(2, 2)).ToList();
        _fc6 = new DilatedConvolutionalLayer<T>(1024, 3, 6, 1, 6);
        _fc7 = new ConvolutionalLayer<T>(1024, 1, 1, 0, (IActivationFunction<T>?)null);
        EnsureArchitectureInitialized();

        // Layers are created in training mode; a new backbone predicts, so its batch norms must start on
        // running statistics. Otherwise a rebuilt copy (Clone, Deserialize) normalizes with batch
        // statistics and disagrees with the model it was copied from.
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>(5);
        var x = input;
        for (int b = 0; b < _blocks.Count; b++)
        {
            if (b > 0) x = _pools[b - 1].Forward(x);
            foreach (var (conv, norm) in _blocks[b])
                x = _relu.Activate(norm.Forward(conv.Forward(x)));
            if (b >= 1) features.Add(x);
        }

        // CRAFT's slice5: pool5 at stride 1 keeps the resolution; fc6 and fc7 follow without activation.
        x = CvTensorOps<T>.MaxPoolPadded(x, 3, 1, 1);
        features.Add(_fc7.Forward(_fc6.Forward(x)));
        return features;
    }

    /// <inheritdoc/>
    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <inheritdoc/>
    public void WriteParameters(BinaryWriter writer)
    {
        foreach (var layer in Layers.OfType<LayerBase<T>>()) BackboneSerialization.WriteLayerParameters(writer, layer);
    }

    /// <inheritdoc/>
    public void ReadParameters(BinaryReader reader)
    {
        foreach (var layer in Layers.OfType<LayerBase<T>>()) BackboneSerialization.ReadLayerParameters(reader, layer);
    }

    /// <summary>Freezes the backbone.</summary>
    public virtual void Freeze() => IsFrozen = true;

    /// <summary>Unfreezes the backbone.</summary>
    public virtual void Unfreeze() => IsFrozen = false;

    /// <summary>CRAFT's canonical training resolution.</summary>
    public (int Height, int Width) GetExpectedInputSize() => (768, 768);

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) => ExtractFeatures(input)[^1];

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        foreach (var block in _blocks)
            foreach (var (conv, norm) in block) { Layers.Add(conv); Layers.Add(norm); }
        Layers.Add(_fc6);
        Layers.Add(_fc7);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = Name,
        AdditionalInfo = new Dictionary<string, object>
        {
            ["BackboneName"] = Name,
            ["OutputChannels"] = OutputChannels,
            ["Strides"] = Strides
        }
    };

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) =>
        throw new NotSupportedException($"{GetType().Name}: detection backbones train as part of a parent detector.");

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException($"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones.");
}