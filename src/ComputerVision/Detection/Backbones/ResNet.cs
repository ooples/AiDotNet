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
/// ResNet backbone network for feature extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> ResNet (Residual Network) is a foundational architecture
/// that introduced skip connections to enable training of very deep networks. It's widely
/// used as a backbone for detection models like Faster R-CNN.</para>
///
/// <para>Key features:
/// - Residual blocks with skip connections prevent gradient vanishing
/// - Multiple variants: ResNet-18, 34, 50, 101, 152
/// - Bottleneck blocks (3 convolutions) for deeper networks
/// </para>
///
/// <para>Reference: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Deep Residual Learning for Image Recognition",
    "https://arxiv.org/abs/1512.03385",
    Year = 2016,
    Authors = "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
[TaskVariantOf(typeof(AiDotNet.NeuralNetworks.ResNetNetwork<>), "detection backbone")]
public partial class ResNet<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    // UpdateParameters delegated straight to SetParameters. The base does that now.
    private readonly ResNetVariant _variant;
    private ResNetTrunk<T>? _trunk;
    private readonly ResNetTrunkConfiguration _trunkConfiguration;

    public bool IsFrozen { get; private set; }
    public string Name => $"ResNet-{GetLayerCount(_variant)}";
    public IReadOnlyList<int> OutputChannels { get; }
    public IReadOnlyList<int> Strides => new[] { 4, 8, 16, 32 };

    /// <summary>The convolutional trunk, shared with <see cref="ResNetNetwork{T}"/>.</summary>
    internal ResNetTrunk<T> Trunk => _trunk ?? throw new InvalidOperationException("The backbone is not initialized.");

    /// <summary>
    /// Creates a new ResNet backbone: the classifier's trunk (torchvision's ResNet without its pooling and fully
    /// connected layer), producing the C2–C5 feature pyramid.
    /// </summary>
    public ResNet(ResNetBackboneOptions? options = null)
        : base(DetectionBackboneArchitecture<T>.Create((options ??= new ResNetBackboneOptions()).InChannels),
              new MeanSquaredErrorLoss<T>())
    {
        _variant = options.Variant;
        _trunkConfiguration = ResNetTrunkConfiguration.For(options.Variant, options.InChannels);
        OutputChannels = _trunkConfiguration.OutputChannels;
        InitializeLayers();
        // A backbone starts in inference mode: its parent detector switches it to training when it trains, and the
        // base default (training) would otherwise normalize every detection with the batch's own statistics.
        SetTrainingMode(false);
    }

    private static int GetLayerCount(ResNetVariant variant) => variant switch
    {
        ResNetVariant.ResNet18 => 18,
        ResNetVariant.ResNet34 => 34,
        ResNetVariant.ResNet50 => 50,
        ResNetVariant.ResNet101 => 101,
        ResNetVariant.ResNet152 => 152,
        _ => throw new NotSupportedException($"ResNet variant {variant} is not supported.")
    };

    private static Tensor<T> Batched(Tensor<T> input)
    {
        // Backbones accept a single [C,H,W] image or a batched [N,C,H,W] tensor.
        if (input.Shape.Length == 3)
            return input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        if (input.Shape.Length != 4)
            throw new ArgumentException(
                $"ResNet expects a [C,H,W] or [N,C,H,W] image tensor, but got rank-{input.Shape.Length} " +
                $"[{string.Join(",", input.Shape.ToArray())}].", nameof(input));
        return input;
    }

    public List<Tensor<T>> ExtractFeatures(Tensor<T> input) => Trunk.Forward(Batched(input));

    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <summary>
    /// Returns the stem output and each residual stage's output (the C2..C5 pyramid), keyed by name.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var x = Batched(input);
        var activations = new Dictionary<string, Tensor<T>> { ["Stem"] = Trunk.Stem(x).Clone() };
        var features = Trunk.Forward(x);
        for (int i = 0; i < features.Count; i++)
            activations[$"Stage{i + 1}_C{i + 2}"] = features[i].Clone();
        return activations;
    }

    /// <summary>Loads a torchvision ResNet state dict (its <c>fc</c> entries are not read).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Trunk.LoadTorchWeights(read);

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write((int)_variant);
        writer.Write(Layers.Count);
        foreach (var layer in Layers) BackboneSerialization.WriteLayerParameters(writer, (LayerBase<T>)layer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        var variant = (ResNetVariant)reader.ReadInt32();
        int layerCount = reader.ReadInt32();
        if (variant != _variant)
            throw new InvalidOperationException($"ResNet variant mismatch: expected {_variant}, got {variant}.");
        if (layerCount != Layers.Count)
            throw new InvalidOperationException($"ResNet layer count mismatch: expected {Layers.Count}, got {layerCount}.");
        foreach (var layer in Layers) BackboneSerialization.ReadLayerParameters(reader, (LayerBase<T>)layer);
    }

    public virtual void Freeze() => IsFrozen = true;
    public virtual void Unfreeze() => IsFrozen = false;
    public (int Height, int Width) GetExpectedInputSize() => (640, 640);

    protected override Tensor<T> PredictCore(Tensor<T> input) => ExtractFeatures(input)[3];

    protected override void InitializeLayers()
    {
        // The base constructor calls this before the configuration is set; the constructor calls it again.
        if (_trunkConfiguration is null || _trunk is not null) return;
        var layers = new List<LayerBase<T>>();
        _trunk = new ResNetTrunk<T>(Engine, layers, _trunkConfiguration);
        _trunk.InitializeLikeReference(LayerInitializationSeedScope.NextRandom());
        Layers.AddRange(layers);
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain (a
    /// projection shortcut reads its block's input).</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

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

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) =>
        throw new NotSupportedException(
            $"{GetType().Name}: detection backbones train as part of a parent detector " +
            "(FasterRCNN, YOLOv8, DETR, …) which orchestrates the joint forward/backward pass. " +
            "Train the parent detection model instead.");

    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones. " +
            "Use ReadParameters(BinaryReader) on a fresh instance.");
}
