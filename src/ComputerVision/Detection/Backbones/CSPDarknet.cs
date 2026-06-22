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

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// CSP-Darknet backbone network used in YOLO family models (v5, v7, v8).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CSP-Darknet is a specialized feature extraction network
/// designed for real-time object detection. It uses Cross-Stage Partial connections
/// to reduce computation while maintaining accuracy.</para>
///
/// <para>Reference: Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection"</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("YOLOv4: Optimal Speed and Accuracy of Object Detection",
    "https://arxiv.org/abs/2004.10934",
    Year = 2020,
    Authors = "Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao")]
public class CSPDarknet<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private readonly List<CSPBlock<T>> _stages;
    private readonly ConvolutionalLayer<T> _stem;
    private readonly int _depth;
    private readonly double _depthOriginal;
    private readonly double _widthMultiplier;
    private readonly int _inChannels;
    private readonly int[] _stageChannels;
    /// <summary>
    /// Activation applied throughout the network. Defaults to SiLU (the YOLOv4 paper's
    /// choice); callers can pass any <see cref="IActivationFunction{T}"/> to override.
    /// </summary>
    private readonly IActivationFunction<T> _activation;

    public bool IsFrozen { get; private set; }
    public string Name => $"CSPDarknet-{_widthMultiplier:0.0}x";
    public IReadOnlyList<int> OutputChannels { get; }
    public IReadOnlyList<int> Strides => new[] { 8, 16, 32 };

    /// <summary>
    /// Creates a new CSP-Darknet backbone.
    /// </summary>
    /// <param name="depth">Depth multiplier for number of blocks (default 1.0 = medium).</param>
    /// <param name="widthMultiplier">Width multiplier for channel counts (default 1.0 = medium).</param>
    /// <param name="inChannels">Number of input channels (default 3 for RGB).</param>
    /// <param name="activation">
    /// Activation function applied throughout the network. <c>null</c> resolves to
    /// the YOLOv4 paper default <see cref="SiLUActivation{T}"/>.
    /// </param>
    public CSPDarknet(
        double depth = 1.0,
        double widthMultiplier = 1.0,
        int inChannels = 3,
        IActivationFunction<T>? activation = null)
        : base(NeuralNetworkArchitecture<T>.CreateDynamicSpatial(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.ImageClassification,
                channels: inChannels,
                outputSize: 1),
              new MeanSquaredErrorLoss<T>())
    {
        _depthOriginal = depth;
        _depth = Math.Max(1, (int)Math.Round(depth));
        _widthMultiplier = widthMultiplier;
        _inChannels = inChannels;
        _activation = activation ?? new SiLUActivation<T>();
        _stages = new List<CSPBlock<T>>();

        int[] baseChannels = { 64, 128, 256, 512 };
        _stageChannels = baseChannels.Select(c => (int)(c * widthMultiplier)).ToArray();
        OutputChannels = new[] { _stageChannels[1], _stageChannels[2], _stageChannels[3] };

        _stem = new ConvolutionalLayer<T>(outputDepth: _stageChannels[0] / 2, kernelSize: 3, stride: 2, padding: 1);

        int currentChannels = _stageChannels[0] / 2;
        for (int i = 0; i < 4; i++)
        {
            int outChannels = _stageChannels[i];
            int numBlocks = GetBlockCount(i, _depth);
            var stage = new CSPBlock<T>(currentChannels, outChannels, numBlocks, stride: 2, activation: _activation);
            _stages.Add(stage);
            currentChannels = outChannels;
        }
    }

    private int GetBlockCount(int stage, int depth)
    {
        int[] baseCounts = { 1, 2, 8, 8 };
        return Math.Max(1, (int)Math.Round(baseCounts[stage] * depth * 0.33));
    }

    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>();
        var x = _stem.Forward(input);
        x = _activation.Activate(x);
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].Forward(x);
            if (i >= 1) features.Add(x);
        }
        return features;
    }

    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <summary>
    /// Sum across stem + every CSP stage. Inherited
    /// <c>NeuralNetworkBase&lt;T&gt;.GetParameterCount()</c> delegates to this
    /// virtual property, satisfying the <see cref="IDetectionBackbone{T}"/> contract.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            long count = _stem.ParameterCount;
            for (int i = 0; i < _stages.Count; i++) count += _stages[i].GetParameterCount();
            return count;
        }
    }

    public void WriteParameters(BinaryWriter writer)
    {
        BackboneSerialization.WriteLayerParameters(writer, _stem);
        writer.Write(_stages.Count);
        foreach (var stage in _stages) stage.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        BackboneSerialization.ReadLayerParameters(reader, _stem);
        int stageCount = reader.ReadInt32();
        if (stageCount != _stages.Count)
            throw new InvalidOperationException($"Expected {_stages.Count} stages but found {stageCount}.");
        foreach (var stage in _stages) stage.ReadParameters(reader);
    }

    public virtual void Freeze() => IsFrozen = true;
    public virtual void Unfreeze() => IsFrozen = false;
    public (int Height, int Width) GetExpectedInputSize() => (640, 640);

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var features = ExtractFeatures(input);
        if (features.Count == 0)
            throw new InvalidOperationException(
                $"{GetType().Name}.ExtractFeatures returned no feature maps.");
        return features[features.Count - 1];
    }

    protected override void InitializeLayers() { }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) => WriteParameters(writer);
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) => ReadParameters(reader);

    /// <inheritdoc />
    /// <remarks>
    /// Constructs a fresh CSPDarknet with the same depth, width multiplier, and
    /// input-channel configuration. All internal layers are freshly allocated.
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new CSPDarknet<T>(_depthOriginal, _widthMultiplier, _inChannels, _activation);

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
            $"{GetType().Name}: detection backbones train as part of a parent detector.");

    public override Vector<T> GetParameters() =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not expose a flat parameter vector. Use WriteParameters/ReadParameters.");

    public override void SetParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not accept a flat parameter vector. Use ReadParameters.");

    public override void UpdateParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not accept a flat parameter update vector.");

    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones.");

    /// <inheritdoc />
    /// <remarks>
    /// Round-trips the parameter binary stream through a fresh
    /// <see cref="CreateNewInstance"/> so internal Conv / BN layers and their
    /// tensor buffers are independent copies — see ResNet.DeepCopy.
    /// </remarks>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = (CSPDarknet<T>)CreateNewInstance();
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

    // SiLU activation moved to BackboneOps<T>.ApplySiLU — was duplicated 3 times in this file.
}

/// <summary>
/// Cross-Stage Partial block used in CSP-Darknet.
/// </summary>
internal class CSPBlock<T>
{
    private readonly ConvolutionalLayer<T> _downsample;
    private readonly ConvolutionalLayer<T> _cv1;
    private readonly ConvolutionalLayer<T> _cv2;
    private readonly ConvolutionalLayer<T> _cv3;
    private readonly List<CSPBottleneckBlock<T>> _bottlenecks;
    private readonly IActivationFunction<T> _activation;

    public CSPBlock(int inChannels, int outChannels, int numBlocks, int stride, IActivationFunction<T> activation)
    {
        _activation = activation;
        int hiddenChannels = outChannels / 2;

        _downsample = new ConvolutionalLayer<T>(outChannels, kernelSize: 3, stride: stride, padding: 1);
        _cv1 = new ConvolutionalLayer<T>(hiddenChannels, kernelSize: 1, stride: 1, padding: 0);
        _cv2 = new ConvolutionalLayer<T>(hiddenChannels, kernelSize: 1, stride: 1, padding: 0);

        _bottlenecks = new List<CSPBottleneckBlock<T>>();
        for (int i = 0; i < numBlocks; i++)
            _bottlenecks.Add(new CSPBottleneckBlock<T>(hiddenChannels, activation: activation));

        _cv3 = new ConvolutionalLayer<T>(outChannels, kernelSize: 1, stride: 1, padding: 0);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = _downsample.Forward(input);
        x = _activation.Activate(x);
        var y1 = _cv1.Forward(x);
        y1 = _activation.Activate(y1);
        var y2 = _cv2.Forward(x);
        y2 = _activation.Activate(y2);
        foreach (var b in _bottlenecks) y2 = b.Forward(y2);
        var concat = AiDotNetEngine.Current.TensorConcatenate(new[] { y1, y2 }, axis: 1);
        var output = _cv3.Forward(concat);
        return _activation.Activate(output);
    }

    public long GetParameterCount()
    {
        long count = _downsample.ParameterCount + _cv1.ParameterCount + _cv2.ParameterCount + _cv3.ParameterCount;
        foreach (var b in _bottlenecks) count += b.GetParameterCount();
        return count;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        BackboneSerialization.WriteLayerParameters(writer, _downsample);
        BackboneSerialization.WriteLayerParameters(writer, _cv1);
        BackboneSerialization.WriteLayerParameters(writer, _cv2);
        BackboneSerialization.WriteLayerParameters(writer, _cv3);
        writer.Write(_bottlenecks.Count);
        foreach (var b in _bottlenecks) b.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        BackboneSerialization.ReadLayerParameters(reader, _downsample);
        BackboneSerialization.ReadLayerParameters(reader, _cv1);
        BackboneSerialization.ReadLayerParameters(reader, _cv2);
        BackboneSerialization.ReadLayerParameters(reader, _cv3);
        int bottleneckCount = reader.ReadInt32();
        if (bottleneckCount != _bottlenecks.Count)
            throw new InvalidOperationException($"Expected {_bottlenecks.Count} bottlenecks but found {bottleneckCount}.");
        foreach (var b in _bottlenecks) b.ReadParameters(reader);
    }

    // SiLU activation moved to BackboneOps<T>.ApplySiLU — was duplicated 3 times in this file.
}

/// <summary>
/// Bottleneck block with residual connection used inside CSP blocks.
/// Renamed from <c>BottleneckBlock</c> to <c>CSPBottleneckBlock</c> to avoid clashing
/// with the layer-level <c>BottleneckBlock&lt;T&gt;</c> in <c>NeuralNetworks.Layers</c>.
/// </summary>
internal class CSPBottleneckBlock<T>
{
    private readonly ConvolutionalLayer<T> _cv1;
    private readonly ConvolutionalLayer<T> _cv2;
    private readonly bool _add;
    private readonly IActivationFunction<T> _activation;

    public CSPBottleneckBlock(int channels, IActivationFunction<T> activation, bool add = true)
    {
        _add = add;
        _activation = activation;
        _cv1 = new ConvolutionalLayer<T>(channels, kernelSize: 3, stride: 1, padding: 1);
        _cv2 = new ConvolutionalLayer<T>(channels, kernelSize: 3, stride: 1, padding: 1);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var y = _cv1.Forward(input);
        y = _activation.Activate(y);
        y = _cv2.Forward(y);
        y = _activation.Activate(y);
        if (_add)
            y = BackboneOps<T>.AddResidual(y, input);
        return y;
    }

    public long GetParameterCount() => _cv1.ParameterCount + _cv2.ParameterCount;

    public void WriteParameters(BinaryWriter writer)
    {
        BackboneSerialization.WriteLayerParameters(writer, _cv1);
        BackboneSerialization.WriteLayerParameters(writer, _cv2);
    }

    public void ReadParameters(BinaryReader reader)
    {
        BackboneSerialization.ReadLayerParameters(reader, _cv1);
        BackboneSerialization.ReadLayerParameters(reader, _cv2);
    }

    // SiLU activation moved to BackboneOps<T>.ApplySiLU — was duplicated 3 times in this file.
}
