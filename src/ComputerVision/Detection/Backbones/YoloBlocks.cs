using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using System.Linq;

namespace AiDotNet.ComputerVision.Detection.Backbones;

// Ultralytics YOLO building blocks (ultralytics/nn/modules/block.py and conv.py), shared by the per-paper
// YOLO backbones and necks. Rank-4 [batch, channels, H, W] only: every block concatenates on axis 1.

/// <summary>YOLO's Conv: convolution, batch normalization, then SiLU (or no activation when <c>act</c> is false).</summary>
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 2, 8, 8", TestConstructorArgs = "4, 3, 2")]
[AutoParameters]
public partial class YoloConv<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _groups;
    private readonly bool _act;
    private readonly ConvolutionalLayer<T> _conv;
    private readonly BatchNormalizationLayer<T> _norm;
    private readonly IActivationFunction<T> _silu = new SiLUActivation<T>();

    /// <summary>Creates a Conv with "same" padding for odd kernels.</summary>
    public YoloConv([LayerState] int outChannels, [LayerState] int kernelSize = 1, [LayerState] int stride = 1,
        [LayerState] int groups = 1, [LayerState] bool act = true)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _groups = groups;
        _act = act;
        // Ultralytics Conv is nn.Conv2d(..., bias=False) followed by BatchNorm2d, whose shift makes a conv bias
        // redundant. Stated explicitly: BiasMode.Auto asks the following layer, and inside this block the BN is not
        // in any layer list the conv can see, so Auto kept a bias the reference does not have (1 extra per channel).
        _conv = new ConvolutionalLayer<T>(outChannels, kernelSize, stride, kernelSize / 2, (IActivationFunction<T>?)null,
            initializationStrategy: null, nonlinearityForInit: null, groups: groups, biasMode: BiasMode.Never);
        // The reference YOLO initialize_weights (ultralytics and WongKinYiu/yolov9 utils/torch_utils.py)
        // sets every BatchNorm2d to eps=1e-3, momentum=0.03 (PyTorch's weight on the new batch; here the
        // weight on the running value, so 0.97).
        _norm = new BatchNormalizationLayer<T>(epsilon: 1e-3, momentum: 0.97);
    }

    internal IEnumerable<LayerBase<T>> Children() { yield return _conv; yield return _norm; }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank != 4 ? null : new[]
    {
        new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
        new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels)),
        new OutputAxisContract(TensorAxis.Height, AxisRelation.Window(TensorAxis.Height, _kernelSize, _stride, _kernelSize / 2)),
        new OutputAxisContract(TensorAxis.Width, AxisRelation.Window(TensorAxis.Width, _kernelSize, _stride, _kernelSize / 2)),
    };


    /// <inheritdoc/>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Convolutional;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        var y = _norm.Forward(_conv.Forward(input));
        return _act ? _silu.Activate(y) : y;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}

/// <summary>YOLO's Bottleneck: two 3x3 Convs, with the input added back when <c>shortcut</c> and widths match.</summary>
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 4, TestInputShape = "1, 4, 8, 8", TestConstructorArgs = "4")]
[AutoParameters]
public partial class YoloBottleneck<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly double _expansion;
    private readonly YoloConv<T> _cv1;
    private readonly YoloConv<T> _cv2;
    private readonly bool _shortcut;

    /// <summary>Creates a bottleneck of <paramref name="channels"/> in and out.</summary>
    public YoloBottleneck([LayerState] int channels, [LayerState] bool shortcut = true, [LayerState] double expansion = 1.0)
        : base(new[] { channels, -1, -1 }, new[] { channels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _channels = channels;
        _expansion = expansion;
        _cv1 = new YoloConv<T>(Math.Max(1, (int)(channels * expansion)), 3);
        _cv2 = new YoloConv<T>(channels, 3);
        _shortcut = shortcut;
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank != 4 ? null : new[]
    {
        new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
        new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_channels)),
        new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
        new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width)),
    };

    /// <inheritdoc/>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Identity;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        // The residual adds the input to a _channels-wide output, so the widths must agree. Reference YOLO only
        // builds a shortcut when c1 == c2; here the declared width is the contract, so a mismatch is a caller error.
        int inputChannels = input.Shape.Length == 4 ? input.Shape[1] : input.Shape[0];
        if (_shortcut && inputChannels != _channels)
            throw new ArgumentException(
                $"YoloBottleneck with a shortcut expects {_channels} input channels but received {inputChannels}.",
                nameof(input));
        var y = _cv2.Forward(_cv1.Forward(input));
        return _shortcut ? BackboneOps<T>.AddResidual(y, input) : y;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { _cv1.UpdateParameters(learningRate); _cv2.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); _cv1.SetTrainingMode(isTraining); _cv2.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { _cv1.ResetState(); _cv2.ResetState(); }
}

/// <summary>
/// YOLOv8's C2f (Jocher et al. 2023): split into two halves, run n bottlenecks on the second, and concatenate
/// both halves with EVERY bottleneck's output before a 1x1 fusion Conv.
/// </summary>
/// <remarks>
/// The reference cv1 is one 1x1 Conv to 2c channels whose output is chunked in two; two 1x1 Convs to c each
/// compute exactly the same function (the weight rows partition), without a channel slice.
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 4, 8, 8", TestConstructorArgs = "8, 1, true")]
[AutoParameters]
public partial class C2fBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly int _depth;
    private readonly bool _shortcut;
    private readonly YoloConv<T> _cv1a;
    private readonly YoloConv<T> _cv1b;
    private readonly List<YoloBottleneck<T>> _blocks;
    private readonly YoloConv<T> _cv2;

    /// <summary>Creates a C2f of <paramref name="outChannels"/> with <paramref name="depth"/> bottlenecks.</summary>
    public C2fBlock([LayerState] int outChannels, [LayerState] int depth, [LayerState] bool shortcut)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels;
        _depth = depth;
        _shortcut = shortcut;
        int hidden = Math.Max(1, outChannels / 2);
        _cv1a = new YoloConv<T>(hidden);
        _cv1b = new YoloConv<T>(hidden);
        _blocks = Enumerable.Range(0, Math.Max(1, depth)).Select(_ => new YoloBottleneck<T>(hidden, shortcut)).ToList();
        _cv2 = new YoloConv<T>(outChannels);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _cv1a; yield return _cv1b;
        foreach (var b in _blocks) yield return b;
        yield return _cv2;
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank != 4 ? null : new[]
    {
        new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
        new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels)),
        new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
        new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width)),
    };

    /// <inheritdoc/>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Convolutional;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        var parts = new List<Tensor<T>> { _cv1a.Forward(input), _cv1b.Forward(input) };
        foreach (var block in _blocks) parts.Add(block.Forward(parts[^1]));
        return _cv2.Forward(AiDotNetEngine.Current.TensorConcatenate(parts.ToArray(), axis: 1));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}

/// <summary>
/// Spatial Pyramid Pooling - Fast: a 1x1 Conv to half width, three chained 5x5 stride-1 max pools, and a 1x1
/// Conv over the four concatenated maps (equivalent to SPP's 5/9/13 pools, at a third of the cost).
/// </summary>
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 4, 8, 8", TestConstructorArgs = "4, 8, 5")]
[AutoParameters]
public partial class SPPFLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly YoloConv<T> _cv1;
    private readonly YoloConv<T> _cv2;
    private readonly int _poolSize;

    /// <summary>Creates an SPPF from <paramref name="inChannels"/> to <paramref name="outChannels"/>.</summary>
    public SPPFLayer([LayerState] int inChannels, [LayerState] int outChannels, [LayerState] int poolSize = 5)
        : base(new[] { inChannels, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _cv1 = new YoloConv<T>(Math.Max(1, inChannels / 2));
        _cv2 = new YoloConv<T>(outChannels);
        _poolSize = poolSize;
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank != 4 ? null : new[]
    {
        new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
        new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels)),
        new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
        new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width)),
    };


    /// <inheritdoc/>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Convolutional;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        var x = _cv1.Forward(input);
        var y1 = CvTensorOps<T>.MaxPoolPadded(x, _poolSize, 1, _poolSize / 2);
        var y2 = CvTensorOps<T>.MaxPoolPadded(y1, _poolSize, 1, _poolSize / 2);
        var y3 = CvTensorOps<T>.MaxPoolPadded(y2, _poolSize, 1, _poolSize / 2);
        return _cv2.Forward(AiDotNetEngine.Current.TensorConcatenate(new[] { x, y1, y2, y3 }, axis: 1));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { _cv1.UpdateParameters(learningRate); _cv2.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); _cv1.SetTrainingMode(isTraining); _cv2.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { _cv1.ResetState(); _cv2.ResetState(); }
}
