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
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

using System.Linq;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// EfficientNet backbone for efficient feature extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Reference: Tan et al., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    "https://arxiv.org/abs/1905.11946",
    Year = 2019,
    Authors = "Mingxing Tan, Quoc V. Le")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
[TaskVariantOf(typeof(AiDotNet.NeuralNetworks.EfficientNetNetwork<>), "detection backbone")]
public partial class EfficientNet<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private readonly EfficientNetVariant _variant;
    private readonly EfficientNetTrunkConfiguration _trunkConfiguration;
    private EfficientNetTrunk<T>? _trunk;

    public bool IsFrozen { get; private set; }
    public string Name => $"EfficientNet-{_variant}";
    public IReadOnlyList<int> OutputChannels { get; }
    public IReadOnlyList<int> Strides => new[] { 4, 8, 16, 32 };

    /// <summary>The convolutional trunk, shared with <see cref="EfficientNetNetwork{T}"/>.</summary>
    internal EfficientNetTrunk<T> Trunk => _trunk ?? throw new InvalidOperationException("The backbone is not initialized.");

    /// <summary>
    /// Creates a new EfficientNet backbone: the classifier's trunk without its head convolution, producing the C2–C5
    /// pyramid from the ends of stages 2, 3, 5 and 7 (strides 4, 8, 16 and 32), as EfficientDet reads it.
    /// </summary>
    public EfficientNet(EfficientNetBackboneOptions? options = null)
        : base(DetectionBackboneArchitecture<T>.Create((options ??= new EfficientNetBackboneOptions()).InChannels),
              new MeanSquaredErrorLoss<T>())
    {
        _variant = options.Variant;
        var (width, depth) = EfficientNetTrunkConfiguration.MultipliersOf(options.Variant);
        _trunkConfiguration = new EfficientNetTrunkConfiguration(width, depth, options.InChannels, IncludeHead: false);
        OutputChannels = _trunkConfiguration.PyramidChannels;
        InitializeLayers();
        // A backbone starts in inference mode: its parent detector switches it to training when it trains, and the
        // base default (training) would otherwise normalize every detection with the batch's own statistics.
        SetTrainingMode(false);
    }

    private static Tensor<T> Batched(Tensor<T> input)
    {
        if (input.Shape.Length == 3)
            return input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        if (input.Shape.Length != 4)
            throw new ArgumentException(
                $"EfficientNet expects a [C,H,W] or [N,C,H,W] image tensor, but got rank-{input.Shape.Length}.", nameof(input));
        return input;
    }

    /// <summary>Draws the stochastic-depth decisions while a parent detector trains.</summary>
    private readonly Random _trainingRandom = LayerInitializationSeedScope.NextRandom();

    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var stages = Trunk.Forward(Batched(input), IsTrainingMode, _trainingRandom);
        return EfficientNetTrunkConfiguration.PyramidStages.Select(i => stages[i]).ToList();
    }

    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <summary>Loads a timm <c>tf_efficientnet</c> state dict (its head and classifier entries are not read).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Trunk.LoadTorchWeights(read);

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write((int)_variant);
        writer.Write(Layers.Count);
        foreach (var layer in Layers) BackboneSerialization.WriteLayerParameters(writer, (LayerBase<T>)layer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        var variant = (EfficientNetVariant)reader.ReadInt32();
        int layerCount = reader.ReadInt32();
        if (variant != _variant)
            throw new InvalidOperationException($"EfficientNet variant mismatch: expected {_variant}, got {variant}.");
        if (layerCount != Layers.Count)
            throw new InvalidOperationException($"EfficientNet layer count mismatch: expected {Layers.Count}, got {layerCount}.");
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
        _trunk = new EfficientNetTrunk<T>(Engine, layers, _trunkConfiguration);
        _trunk.InitializeLikeReference(LayerInitializationSeedScope.NextRandom());
        Layers.AddRange(layers);
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
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
            $"{GetType().Name}: detection backbones train as part of a parent detector. Train the parent detection model instead.");

    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones.");
}
