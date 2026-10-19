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
/// VGG-16 (configuration D, no batch normalization) as a feature-pyramid backbone, cut the way EAST cuts it.
/// </summary>
/// <remarks>
/// <para>
/// Thirteen 3x3 convolutions with ReLU in five blocks (64, 128, 256, 512, 512 channels), each block followed by 2x2
/// max pooling (Simonyan and Zisserman 2015). EAST (Zhou et al. 2017, §3.2) extracts "feature maps after pooling-2 to
/// pooling-5", returned here finest first at strides 4, 8, 16 and 32.
/// </para>
/// <para>
/// Parameter names follow torchvision's <c>vgg16().features</c>, so <see cref="LoadTorchWeights"/> reads ImageNet
/// weights, and <see cref="InitializeLikeReference"/> is torchvision's initialization (Kaiming normal with fan-out,
/// zero biases).
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
public partial class VGG16Backbone<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private static readonly int[][] BlockChannels =
    {
        new[] { 64, 64 },
        new[] { 128, 128 },
        new[] { 256, 256, 256 },
        new[] { 512, 512, 512 },
        new[] { 512, 512, 512 },
    };

    private readonly int _inChannels;
    private readonly List<ConvolutionalLayer<T>[]> _blocks;
    private readonly List<MaxPoolingLayer<T>> _pools;

    /// <summary>Whether the backbone's weights are frozen.</summary>
    public bool IsFrozen { get; private set; }

    /// <summary>The backbone name.</summary>
    public string Name => "VGG16";

    /// <summary>Channels of each returned feature map (pool2 to pool5), finest first.</summary>
    public IReadOnlyList<int> OutputChannels { get; } = new[] { 128, 256, 512, 512 };

    /// <summary>Stride of each returned feature map, finest first.</summary>
    public IReadOnlyList<int> Strides => new[] { 4, 8, 16, 32 };

    /// <summary>Creates the backbone, initialized as torchvision initializes VGG.</summary>
    /// <param name="options">Input channels; defaults to three.</param>
    public VGG16Backbone(VGG16BackboneOptions? options = null)
        : base(DetectionBackboneArchitecture<T>.Create(VGG16BackboneOptions.OrDefault(options).InChannels),
              new MeanSquaredErrorLoss<T>())
    {
        var resolved = VGG16BackboneOptions.OrDefault(options);
        resolved.Validate();
        _inChannels = resolved.InChannels;
        int inputs = _inChannels;
        _blocks = new List<ConvolutionalLayer<T>[]>();
        foreach (var block in BlockChannels)
        {
            var convs = new ConvolutionalLayer<T>[block.Length];
            for (int i = 0; i < block.Length; i++)
            {
                convs[i] = ConvolutionalLayer<T>.WithInputDepth(inputs, block[i], 3, 1, padding: 1,
                    activationFunction: new ReLUActivation<T>(),
                    initializationStrategy: new AiDotNet.Initialization.ZeroInitializationStrategy<T>());
                inputs = block[i];
            }
            _blocks.Add(convs);
        }
        _pools = Enumerable.Range(0, 5).Select(_ => new MaxPoolingLayer<T>(2, 2)).ToList();
        EnsureArchitectureInitialized();
        InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(resolved.Seed ?? 42));
    }

    /// <summary>The convolutions in order, with torchvision's <c>features</c> indices.</summary>
    private IEnumerable<(ConvolutionalLayer<T> Conv, int Index, int Inputs, int Outputs)> Convolutions()
    {
        int index = 0, inputs = _inChannels;
        for (int b = 0; b < _blocks.Count; b++)
        {
            foreach (var conv in _blocks[b])
            {
                int outputs = BlockChannels[b][Array.IndexOf(_blocks[b], conv)];
                yield return (conv, index, inputs, outputs);
                inputs = outputs;
                index += 2;          // conv, ReLU
            }
            index += 1;              // max pool
        }
    }

    /// <summary>Loads torchvision <c>vgg16</c> weights (<c>features.{i}.weight</c> and <c>.bias</c>), given a reader of a
    /// named tensor of a given shape.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        foreach (var (conv, index, inputs, outputs) in Convolutions())
        {
            var weight = read($"features.{index}.weight", new[] { outputs, inputs, 3, 3 });
            var bias = read($"features.{index}.bias", new[] { outputs });
            conv.SetParameters(new Vector<T>(weight.Concat(bias).Select(NumOps.FromDouble).ToArray()));
        }
    }

    /// <summary>torchvision's VGG initialization: <c>kaiming_normal_(mode="fan_out", nonlinearity="relu")</c>, zero bias.</summary>
    public void InitializeLikeReference(Random random)
    {
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, b) => a * b);
            var values = new double[count];
            if (name.EndsWith(".weight", StringComparison.Ordinal))
            {
                double std = Math.Sqrt(2.0 / (shape[0] * shape[2] * shape[3]));
                for (int i = 0; i < count; i++) values[i] = std * Normal();
            }
            return values;
        });
    }

    /// <inheritdoc/>
    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>(4);
        var x = input;
        for (int b = 0; b < _blocks.Count; b++)
        {
            foreach (var conv in _blocks[b]) x = conv.Forward(x);
            x = _pools[b].Forward(x);
            if (b >= 1) features.Add(x);
        }
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

    /// <summary>EAST's training crop (512x512, §3.5).</summary>
    public (int Height, int Width) GetExpectedInputSize() => (512, 512);

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) => ExtractFeatures(input)[^1];

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        foreach (var conv in _blocks.SelectMany(block => block)) Layers.Add(conv);
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
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
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException($"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones.");
}
