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
public class ResNet<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private readonly ConvolutionalLayer<T> _conv1;
    private readonly List<ResNetStage<T>> _stages;
    private readonly ResNetVariant _variant;
    private readonly int _inChannels;
    /// <summary>
    /// Activation between stages. Defaults to ReLU per the He et al. 2016 paper;
    /// callers can override with any <see cref="IActivationFunction{T}"/>.
    /// </summary>
    private readonly IActivationFunction<T> _activation;

    public bool IsFrozen { get; private set; }
    public string Name => $"ResNet-{GetLayerCount(_variant)}";
    public IReadOnlyList<int> OutputChannels { get; }
    public IReadOnlyList<int> Strides => new[] { 4, 8, 16, 32 };

    /// <summary>
    /// Creates a new ResNet backbone.
    /// </summary>
    /// <param name="variant">ResNet variant (18, 34, 50, 101, or 152).</param>
    /// <param name="inChannels">Number of input channels (default 3 for RGB).</param>
    /// <param name="activation">
    /// Activation applied between stages and at the stem. <c>null</c> resolves to the
    /// He et al. 2016 paper default <see cref="ReLUActivation{T}"/>.
    /// </param>
    public ResNet(
        ResNetVariant variant = ResNetVariant.ResNet50,
        int inChannels = 3,
        IActivationFunction<T>? activation = null)
        : base(NeuralNetworkArchitecture<T>.CreateDynamicSpatial(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.ImageClassification,
                channels: inChannels,
                outputSize: 1),
              new MeanSquaredErrorLoss<T>())
    {
        _variant = variant;
        _inChannels = inChannels;
        _activation = activation ?? new ReLUActivation<T>();
        _stages = new List<ResNetStage<T>>();

        bool useBottleneck = variant >= ResNetVariant.ResNet50;
        int expansion = useBottleneck ? 4 : 1;
        int[] baseChannels = { 64, 128, 256, 512 };
        OutputChannels = baseChannels.Select(c => c * expansion).ToArray();

        // Stem 7×7 conv stride=2 — input depth resolves lazily.
        _conv1 = new ConvolutionalLayer<T>(outputDepth: 64, kernelSize: 7, stride: 2, padding: 3);

        int[] blockCounts = GetBlockCounts(variant);
        int currentChannels = 64;
        for (int i = 0; i < 4; i++)
        {
            int outChannels = baseChannels[i];
            int stride = i == 0 ? 1 : 2;
            var stage = new ResNetStage<T>(currentChannels, outChannels, blockCounts[i], stride, useBottleneck, _activation);
            _stages.Add(stage);
            currentChannels = outChannels * expansion;
        }
    }

    private static int GetLayerCount(ResNetVariant variant) => variant switch
    {
        ResNetVariant.ResNet18 => 18,
        ResNetVariant.ResNet34 => 34,
        ResNetVariant.ResNet50 => 50,
        ResNetVariant.ResNet101 => 101,
        ResNetVariant.ResNet152 => 152,
        _ => 50
    };

    private static int[] GetBlockCounts(ResNetVariant variant) => variant switch
    {
        ResNetVariant.ResNet18 => new[] { 2, 2, 2, 2 },
        ResNetVariant.ResNet34 => new[] { 3, 4, 6, 3 },
        ResNetVariant.ResNet50 => new[] { 3, 4, 6, 3 },
        ResNetVariant.ResNet101 => new[] { 3, 4, 23, 3 },
        ResNetVariant.ResNet152 => new[] { 3, 8, 36, 3 },
        _ => new[] { 3, 4, 6, 3 }
    };

    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        // Backbones accept a single [C,H,W] image or a batched [N,C,H,W] tensor.
        // MaxPool2D and the residual stages operate on rank-4 NCHW (MaxPool2D
        // reads Shape[3]), so promote a rank-3 single image to [1,C,H,W]
        // (matching SwinTransformer.EnsureBatchedNchw).
        if (input.Shape.Length == 3)
            input = input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        else if (input.Shape.Length != 4)
            throw new ArgumentException(
                $"ResNet expects a [C,H,W] or [N,C,H,W] image tensor, but got rank-{input.Shape.Length} " +
                $"[{string.Join(",", input.Shape.ToArray())}].", nameof(input));

        var features = new List<Tensor<T>>();
        var x = _conv1.Forward(input);
        x = _activation.Activate(x);
        x = BackboneOps<T>.MaxPool2D(x, kernelSize: 3, stride: 2, padding: 1);
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].Forward(x);
            features.Add(x); // C2..C5
        }
        return features;
    }

    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <summary>
    /// Sum across the stem conv plus every residual stage. Inherited
    /// <c>NeuralNetworkBase&lt;T&gt;.GetParameterCount()</c> already delegates to
    /// this virtual property, so the <see cref="IDetectionBackbone{T}"/>
    /// contract is satisfied without re-declaring the method here.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            long count = _conv1.ParameterCount;
            for (int i = 0; i < _stages.Count; i++)
                count += _stages[i].GetParameterCount();
            return count;
        }
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write((int)_variant);
        writer.Write(_stages.Count);
        BackboneSerialization.WriteLayerParameters(writer, _conv1);
        foreach (var stage in _stages) stage.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        var variant = (ResNetVariant)reader.ReadInt32();
        int stageCount = reader.ReadInt32();
        if (variant != _variant)
            throw new InvalidOperationException($"ResNet variant mismatch: expected {_variant}, got {variant}.");
        if (stageCount != _stages.Count)
            throw new InvalidOperationException($"ResNet stage count mismatch: expected {_stages.Count}, got {stageCount}.");
        BackboneSerialization.ReadLayerParameters(reader, _conv1);
        foreach (var stage in _stages) stage.ReadParameters(reader);
    }

    public virtual void Freeze() => IsFrozen = true;
    public virtual void Unfreeze() => IsFrozen = false;
    public (int Height, int Width) GetExpectedInputSize() => (640, 640);

    public override Tensor<T> Predict(Tensor<T> input)
    {
        var features = ExtractFeatures(input);
        if (features.Count == 0)
            throw new InvalidOperationException(
                $"{GetType().Name}.ExtractFeatures returned no feature maps. A backbone must produce at least one feature map.");
        return features[features.Count - 1];
    }

    protected override void InitializeLayers()
    {
        // Backbones own their per-stage layers directly; the inherited Layers list stays empty.
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) => WriteParameters(writer);
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) => ReadParameters(reader);

    /// <inheritdoc />
    /// <remarks>
    /// Constructs a fresh ResNet with the same variant and input-channel configuration.
    /// MemberwiseClone would alias internal layers and tensors, so deserialization into the
    /// returned instance would mutate the original.
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new ResNet<T>(_variant, _inChannels, _activation);

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

    public override Vector<T> GetParameters() =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not expose a flat parameter vector. " +
            "Use WriteParameters(BinaryWriter) / ReadParameters(BinaryReader) to round-trip weights.");

    public override void SetParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not accept a flat parameter vector. Use ReadParameters(BinaryReader).");

    public override void UpdateParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not accept a flat parameter update vector. " +
            "Update happens inside the parent detector's optimizer step.");

    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones. " +
            "Use ReadParameters(BinaryReader) on a fresh instance.");

    /// <inheritdoc />
    /// <remarks>
    /// Round-trips the parameter binary stream through a fresh
    /// <see cref="CreateNewInstance"/> so internal Conv / BN layers and their
    /// tensor buffers are independent copies — <c>MemberwiseClone()</c> would
    /// alias every reference type and a subsequent train step on the copy
    /// would mutate the original's weights.
    /// </remarks>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = (ResNet<T>)CreateNewInstance();
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            WriteParameters(writer);
        }
        ms.Position = 0;
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            copy.ReadParameters(reader);
        }
        return copy;
    }
}

/// <summary>
/// A stage in ResNet containing multiple residual blocks.
/// </summary>
internal class ResNetStage<T>
{
    private readonly List<ResidualBlock<T>> _blocks;

    public ResNetStage(int inChannels, int outChannels, int numBlocks, int stride, bool useBottleneck, IActivationFunction<T> activation)
    {
        _blocks = new List<ResidualBlock<T>>();
        int expansion = useBottleneck ? 4 : 1;

        _blocks.Add(new ResidualBlock<T>(
            inChannels, outChannels, stride, useBottleneck,
            downsample: inChannels != outChannels * expansion || stride != 1,
            activation: activation));

        for (int i = 1; i < numBlocks; i++)
        {
            _blocks.Add(new ResidualBlock<T>(
                outChannels * expansion, outChannels, 1, useBottleneck, downsample: false, activation: activation));
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = input;
        foreach (var block in _blocks) x = block.Forward(x);
        return x;
    }

    public long GetParameterCount()
    {
        long count = 0;
        foreach (var block in _blocks) count += block.GetParameterCount();
        return count;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_blocks.Count);
        foreach (var block in _blocks) block.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int blockCount = reader.ReadInt32();
        if (blockCount != _blocks.Count)
            throw new InvalidOperationException($"ResNetStage block count mismatch: expected {_blocks.Count}, got {blockCount}.");
        foreach (var block in _blocks) block.ReadParameters(reader);
    }
}

/// <summary>
/// Individual residual block in ResNet.
/// </summary>
internal class ResidualBlock<T>
{
    private readonly ConvolutionalLayer<T> _conv1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly ConvolutionalLayer<T>? _conv3;
    private readonly ConvolutionalLayer<T>? _downsample;
    private readonly bool _useBottleneck;
    private readonly IActivationFunction<T> _activation;

    public ResidualBlock(int inChannels, int outChannels, int stride, bool useBottleneck, bool downsample, IActivationFunction<T> activation)
    {
        _useBottleneck = useBottleneck;
        _activation = activation;
        int expansion = useBottleneck ? 4 : 1;

        if (useBottleneck)
        {
            _conv1 = new ConvolutionalLayer<T>(outChannels, kernelSize: 1, stride: 1, padding: 0);
            _conv2 = new ConvolutionalLayer<T>(outChannels, kernelSize: 3, stride: stride, padding: 1);
            _conv3 = new ConvolutionalLayer<T>(outChannels * expansion, kernelSize: 1, stride: 1, padding: 0);
        }
        else
        {
            _conv1 = new ConvolutionalLayer<T>(outChannels, kernelSize: 3, stride: stride, padding: 1);
            _conv2 = new ConvolutionalLayer<T>(outChannels * expansion, kernelSize: 3, stride: 1, padding: 1);
        }

        if (downsample)
            _downsample = new ConvolutionalLayer<T>(outChannels * expansion, kernelSize: 1, stride: stride, padding: 0);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var identity = _downsample is not null ? _downsample.Forward(input) : input;
        var x = _conv1.Forward(input);
        x = _activation.Activate(x);
        x = _conv2.Forward(x);
        x = _activation.Activate(x);
        if (_useBottleneck && _conv3 is not null) x = _conv3.Forward(x);
        x = BackboneOps<T>.AddResidual(x, identity);
        return _activation.Activate(x);
    }

    public long GetParameterCount()
    {
        long count = _conv1.ParameterCount + _conv2.ParameterCount;
        if (_conv3 is not null) count += _conv3.ParameterCount;
        if (_downsample is not null) count += _downsample.ParameterCount;
        return count;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_useBottleneck);
        writer.Write(_downsample is not null);
        BackboneSerialization.WriteLayerParameters(writer, _conv1);
        BackboneSerialization.WriteLayerParameters(writer, _conv2);
        if (_useBottleneck && _conv3 is not null) BackboneSerialization.WriteLayerParameters(writer, _conv3);
        if (_downsample is not null) BackboneSerialization.WriteLayerParameters(writer, _downsample);
    }

    public void ReadParameters(BinaryReader reader)
    {
        bool useBottleneck = reader.ReadBoolean();
        bool hasDownsample = reader.ReadBoolean();
        if (useBottleneck != _useBottleneck)
            throw new InvalidOperationException("ResidualBlock bottleneck mismatch.");
        if (hasDownsample != (_downsample is not null))
            throw new InvalidOperationException("ResidualBlock downsample mismatch.");
        BackboneSerialization.ReadLayerParameters(reader, _conv1);
        BackboneSerialization.ReadLayerParameters(reader, _conv2);
        if (_useBottleneck && _conv3 is not null) BackboneSerialization.ReadLayerParameters(reader, _conv3);
        if (_downsample is not null) BackboneSerialization.ReadLayerParameters(reader, _downsample);
    }
}
