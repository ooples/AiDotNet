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

        // Publish the stem and stages now. InitializeLayers() is otherwise driven lazily, and
        // nothing on this model's paths triggers it before a forward -- so a freshly constructed
        // backbone reported an empty Layers list, and every base walk over it concluded the network
        // had no learnable parameters at all. Safe here: this is the end of the derived constructor,
        // so _stem and _stages are assigned.
        EnsureArchitectureInitialized();
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
    /// Returns the per-layer activations produced by a forward pass, keyed by a
    /// human-readable name. CSPDarknet organizes its layers as a stem convolution
    /// plus CSP stages (the <see cref="IDetectionBackbone{T}"/> feature pyramid)
    /// rather than the flat base <c>Layers</c> collection, so the base
    /// <see cref="NeuralNetworkBase{T}.GetNamedLayerActivations"/> — which iterates
    /// <c>Layers</c> — would return an empty map. Mirror <see cref="ExtractFeatures"/>'s
    /// forward path exactly and expose the activated stem output plus each CSP
    /// stage's output, so interpretability/activation consumers get the network's
    /// real intermediate features instead of nothing.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        // Promote a single [C,H,W] image to [1,C,H,W] exactly as the forward path expects.
        if (input.Shape.Length == 3)
            input = input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        else if (input.Shape.Length != 4)
            throw new ArgumentException(
                $"CSPDarknet expects a [C,H,W] or [N,C,H,W] image tensor, but got rank-{input.Shape.Length} " +
                $"[{string.Join(",", input.Shape.ToArray())}].", nameof(input));

        // Keys are prefixed with a zero-padded forward-depth index so a consumer that sorts by key
        // (AiModelResult treats the lexicographically-highest key as the final/deepest activation)
        // reads the deepest CSP stage, not the stem. Plain "Stem"/"Stage{i}" sorted "Stem" last.
        var activations = new Dictionary<string, Tensor<T>>();
        var x = _stem.Forward(input);
        x = _activation.Activate(x);
        activations["Layer_00_Stem"] = x.Clone();
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].Forward(x);
            activations[$"Layer_{i + 1:D2}_Stage{i + 1}"] = x.Clone();
        }
        return activations;
    }

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

    /// <summary>
    /// Publishes the stem and CSP stages as <c>Layers</c>, in the order <see cref="ExtractFeatures"/>
    /// runs them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This was empty and <see cref="CSPBlock{T}"/> was a bare holder class, so every weight in the
    /// model was unreachable from the base walks. The visible symptom was one failure --
    /// <c>Parameters_ShouldBeNonEmpty</c> -- but the real damage was silent: five further tests
    /// (MoreData_ShouldNotDegrade, OptimizerStep_ParamL2_DoesNotExplode,
    /// LossStrictlyDecreasesOnMemorizationTask, Gradients_MatchFiniteDifference,
    /// Clone_AfterTraining_ShouldPreserveLearnedWeights) each finished in under a millisecond by
    /// short-circuiting on a zero-length parameter vector. They passed without testing anything.
    /// </para>
    /// <para>
    /// One entry per CSP stage, NOT the stages' inner convolutions flattened: a CSP block splits
    /// into two branches and concatenates, so a flat list would describe a sequential chain this
    /// model does not run -- the wrong-topology trap behind the SlimSAM / Mask2Former cluster. Each
    /// block owns its own branching inside its Forward, so the list stays a truthful chain.
    /// </para>
    /// <para>
    /// Safe to populate here: this runs from <c>EnsureArchitectureInitialized()</c>, not from the
    /// base constructor, so <c>_stem</c> and <c>_stages</c> are already assigned.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        Layers.Add(_stem);
        Layers.AddRange(_stages);
    }

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
/// <remarks>
/// A real <see cref="LayerBase{T}"/> rather than a bare holder class, so CSPDarknet can publish its
/// stages as <c>Layers</c>. As a plain class its weights were unreachable from every base walk:
/// the model reported no learnable parameters at all, and five training invariants "passed" in
/// under a millisecond by short-circuiting on a zero-length parameter vector.
/// </remarks>
// No [LayerProperty]: TestScaffoldGenerator would emit a standalone layer test, and neither
// ctor is expressible as literal TestConstructorArgs (both require an IActivationFunction).
// TrainableParameterGenerator keys on LayerBase inheritance, not the attribute, so the
// generated EnsureSubLayersRegistered() for this CSP stage is emitted regardless.
internal partial class CSPBlock<T> : LayerBase<T>
{
    private readonly ConvolutionalLayer<T> _downsample;
    private readonly ConvolutionalLayer<T> _cv1;
    private readonly ConvolutionalLayer<T> _cv2;
    private readonly ConvolutionalLayer<T> _cv3;
    private readonly List<CSPBottleneckBlock<T>> _bottlenecks;
    private readonly IActivationFunction<T> _activation;

    /// <summary>Construction state: the 'inChannels' the layer was built with.</summary>
    private readonly int _inChannels;

    /// <summary>Construction state: the 'outChannels' the layer was built with.</summary>
    private readonly int _outChannels;

    /// <summary>Construction state: the 'numBlocks' the layer was built with.</summary>
    private readonly int _numBlocks;

    /// <summary>Construction state: the 'stride' the layer was built with.</summary>
    private readonly int _stride;

    public CSPBlock(int inChannels, int outChannels, int numBlocks, int stride, IActivationFunction<T> activation)
        : base(new[] { inChannels, -1, -1 }, new[] { outChannels, -1, -1 },
               (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _stride = stride;
        _numBlocks = numBlocks;
        _outChannels = outChannels;
        _inChannels = inChannels;
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

    /// <summary>Channel width is the block's own; spatial extent follows the stride-3x3 downsample.</summary>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Convolutional;

    public override bool SupportsTraining => true;

    private IEnumerable<LayerBase<T>> InnerLayers()
    {
        yield return _downsample;
        yield return _cv1;
        yield return _cv2;
        foreach (var b in _bottlenecks) yield return b;
        yield return _cv3;
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Hands the children to GetSubLayers() via the generated EnsureSubLayersRegistered().
        EnsureInitializedFromInput(input);

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

    public override void UpdateParameters(T learningRate)
    {
        foreach (var l in InnerLayers()) l.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        Vector<T> all = Vector<T>.Empty();
        foreach (var l in InnerLayers()) all = Vector<T>.Concatenate(all, l.GetParameters());
        return all;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var l in InnerLayers())
        {
            int len = (int)l.ParameterCount;
            var slice = new Vector<T>(parameters.AsSpan().Slice(offset, len).ToArray());
            l.SetParameters(slice);
            offset += len;
        }
        if (offset != parameters.Length)
            throw new ArgumentException($"Expected {offset} parameters for CSPBlock, but got {parameters.Length}.");
    }

    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var l in InnerLayers()) l.SetTrainingMode(isTraining);
    }

    public override void ResetState()
    {
        foreach (var l in InnerLayers()) l.ResetState();
    }

    public override long ParameterCount => GetParameterCount();

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
// No [LayerProperty]: TestScaffoldGenerator would emit a standalone layer test, and neither
// ctor is expressible as literal TestConstructorArgs (both require an IActivationFunction).
// TrainableParameterGenerator keys on LayerBase inheritance, not the attribute, so the
// generated EnsureSubLayersRegistered() for this bottleneck is emitted regardless.
internal partial class CSPBottleneckBlock<T> : LayerBase<T>
{
    private readonly ConvolutionalLayer<T> _cv1;
    private readonly ConvolutionalLayer<T> _cv2;
    private readonly bool _add;
    private readonly IActivationFunction<T> _activation;

    /// <summary>Construction state: the 'channels' the layer was built with.</summary>
    private readonly int _channels;

    public CSPBottleneckBlock(int channels, IActivationFunction<T> activation, bool add = true)
        : base(new[] { channels, -1, -1 }, new[] { channels, -1, -1 },
               (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _channels = channels;
        _add = add;
        _activation = activation;
        _cv1 = new ConvolutionalLayer<T>(channels, kernelSize: 3, stride: 1, padding: 1);
        _cv2 = new ConvolutionalLayer<T>(channels, kernelSize: 3, stride: 1, padding: 1);
    }

    /// <summary>Stride-1 "same"-padded convs plus a residual add: shape in, shape out.</summary>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Identity;

    public override bool SupportsTraining => true;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Hands _cv1/_cv2 to GetSubLayers() via the generated EnsureSubLayersRegistered().
        EnsureInitializedFromInput(input);

        var y = _cv1.Forward(input);
        y = _activation.Activate(y);
        y = _cv2.Forward(y);
        y = _activation.Activate(y);
        if (_add)
            y = BackboneOps<T>.AddResidual(y, input);
        return y;
    }

    public override void UpdateParameters(T learningRate)
    {
        _cv1.UpdateParameters(learningRate);
        _cv2.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
        => Vector<T>.Concatenate(_cv1.GetParameters(), _cv2.GetParameters());

    public override void SetParameters(Vector<T> parameters)
    {
        int len1 = (int)_cv1.ParameterCount;
        int len2 = (int)_cv2.ParameterCount;
        if (parameters.Length != len1 + len2)
            throw new ArgumentException($"Expected {len1 + len2} parameters for CSPBottleneckBlock, but got {parameters.Length}.");
        _cv1.SetParameters(new Vector<T>(parameters.AsSpan().Slice(0, len1).ToArray()));
        _cv2.SetParameters(new Vector<T>(parameters.AsSpan().Slice(len1, len2).ToArray()));
    }

    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _cv1.SetTrainingMode(isTraining);
        _cv2.SetTrainingMode(isTraining);
    }

    public override void ResetState()
    {
        _cv1.ResetState();
        _cv2.ResetState();
    }

    public override long ParameterCount => GetParameterCount();

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
