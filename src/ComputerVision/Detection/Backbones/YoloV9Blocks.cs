using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using System.Linq;

namespace AiDotNet.ComputerVision.Detection.Backbones;

// YOLOv9 (GELAN) building blocks, following ultralytics/nn/modules/block.py. Rank-4 [batch, channels, H, W].

/// <summary>YOLOv9's RepConvN at training time: a 3x3 Conv and a 1x1 Conv (each with batch norm, no activation) summed, then SiLU. Re-parameterization into one 3x3 conv is an inference-time optimization that computes the same function.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class RepConvNBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly YoloConv<T> _conv3;
    private readonly YoloConv<T> _conv1;
    private readonly IActivationFunction<T> _silu = new SiLUActivation<T>();

    /// <summary>Creates the block.</summary>
    public RepConvNBlock([LayerState] int outChannels)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels;
        _conv3 = new YoloConv<T>(outChannels, 3, act: false);
        _conv1 = new YoloConv<T>(outChannels, 1, act: false);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _conv3; yield return _conv1;
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
        return _silu.Activate(AiDotNetEngine.Current.TensorAdd(_conv3.Forward(input), _conv1.Forward(input)));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>YOLOv9's RepNBottleneck: RepConvN then a 3x3 Conv, with the input added back when <c>shortcut</c> is set.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, true")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class RepBottleneckBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly bool _shortcut;
    private readonly RepConvNBlock<T> _cv1;
    private readonly YoloConv<T> _cv2;

    /// <summary>Creates the block.</summary>
    public RepBottleneckBlock([LayerState] int channels, [LayerState] bool shortcut = true)
        : base(new[] { -1, -1, -1 }, new[] { channels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _channels = channels; _shortcut = shortcut;
        _cv1 = new RepConvNBlock<T>(channels);
        _cv2 = new YoloConv<T>(channels, 3);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _cv1; yield return _cv2;
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
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Convolutional;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        var y = _cv2.Forward(_cv1.Forward(input));
        return _shortcut ? BackboneOps<T>.AddResidual(y, input) : y;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>YOLOv9's RepNCSP: a CSP (C3) block of <c>depth</c> RepN bottlenecks over half the width, concatenated with a 1x1 bypass.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, 1")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class RepCSPBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly int _depth;
    private readonly bool _shortcut;
    private readonly YoloConv<T> _cv1;
    private readonly YoloConv<T> _cv2;
    private readonly YoloConv<T> _cv3;
    private readonly List<RepBottleneckBlock<T>> _blocks;

    /// <summary>Creates the block.</summary>
    public RepCSPBlock([LayerState] int outChannels, [LayerState] int depth = 1, [LayerState] bool shortcut = true)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels; _depth = depth; _shortcut = shortcut;
        int hidden = Math.Max(1, outChannels / 2);
        _cv1 = new YoloConv<T>(hidden);
        _cv2 = new YoloConv<T>(hidden);
        _cv3 = new YoloConv<T>(outChannels);
        _blocks = Enumerable.Range(0, Math.Max(1, depth)).Select(_ => new RepBottleneckBlock<T>(hidden, shortcut)).ToList();
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _cv1; yield return _cv2; yield return _cv3;
        foreach (var b in _blocks) yield return b;
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
        var y = _cv1.Forward(input);
        foreach (var b in _blocks) y = b.Forward(y);
        return _cv3.Forward(AiDotNetEngine.Current.TensorConcatenate(new[] { y, _cv2.Forward(input) }, axis: 1));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>YOLOv9's RepNCSPELAN4 (GELAN): a 1x1 Conv split in two, two RepNCSP + 3x3 Conv stages chained on the second half, and a 1x1 Conv over all four maps. The reference splits one c3-wide 1x1 Conv; two c3/2-wide Convs over the same channel partition compute the same function.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, 8, 4, 1")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class RepNCSPELAN4Block<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly int _hiddenChannels;
    private readonly int _stageChannels;
    private readonly int _depth;
    private readonly YoloConv<T> _cv1a;
    private readonly YoloConv<T> _cv1b;
    private readonly RepCSPBlock<T> _csp2;
    private readonly YoloConv<T> _conv2;
    private readonly RepCSPBlock<T> _csp3;
    private readonly YoloConv<T> _conv3;
    private readonly YoloConv<T> _cv4;

    /// <summary>Creates the block.</summary>
    public RepNCSPELAN4Block([LayerState] int outChannels, [LayerState] int hiddenChannels, [LayerState] int stageChannels, [LayerState] int depth = 1)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels; _hiddenChannels = hiddenChannels; _stageChannels = stageChannels; _depth = depth;
        int half = Math.Max(1, hiddenChannels / 2);
        _cv1a = new YoloConv<T>(half);
        _cv1b = new YoloConv<T>(half);
        _csp2 = new RepCSPBlock<T>(stageChannels, depth);
        _conv2 = new YoloConv<T>(stageChannels, 3);
        _csp3 = new RepCSPBlock<T>(stageChannels, depth);
        _conv3 = new YoloConv<T>(stageChannels, 3);
        _cv4 = new YoloConv<T>(outChannels);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _cv1a; yield return _cv1b; yield return _csp2; yield return _conv2;
        yield return _csp3; yield return _conv3; yield return _cv4;
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
        var a = _cv1a.Forward(input);
        var b = _cv1b.Forward(input);
        var y2 = _conv2.Forward(_csp2.Forward(b));
        var y3 = _conv3.Forward(_csp3.Forward(y2));
        return _cv4.Forward(AiDotNetEngine.Current.TensorConcatenate(new[] { a, b, y2, y3 }, axis: 1));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>YOLOv9's ELAN1 (GELAN-t/s stage 2): RepNCSPELAN4's layout with plain 3x3 Convs in place of the RepNCSP stages.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, 8, 4")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ELAN1Block<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly int _hiddenChannels;
    private readonly int _stageChannels;
    private readonly YoloConv<T> _cv1a;
    private readonly YoloConv<T> _cv1b;
    private readonly YoloConv<T> _conv2;
    private readonly YoloConv<T> _conv3;
    private readonly YoloConv<T> _cv4;

    /// <summary>Creates the block.</summary>
    public ELAN1Block([LayerState] int outChannels, [LayerState] int hiddenChannels, [LayerState] int stageChannels)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels; _hiddenChannels = hiddenChannels; _stageChannels = stageChannels;
        int half = Math.Max(1, hiddenChannels / 2);
        _cv1a = new YoloConv<T>(half);
        _cv1b = new YoloConv<T>(half);
        _conv2 = new YoloConv<T>(stageChannels, 3);
        _conv3 = new YoloConv<T>(stageChannels, 3);
        _cv4 = new YoloConv<T>(outChannels);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _cv1a; yield return _cv1b; yield return _conv2; yield return _conv3; yield return _cv4;
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
        var a = _cv1a.Forward(input);
        var b = _cv1b.Forward(input);
        var y2 = _conv2.Forward(b);
        var y3 = _conv3.Forward(y2);
        return _cv4.Forward(AiDotNetEngine.Current.TensorConcatenate(new[] { a, b, y2, y3 }, axis: 1));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>YOLOv9's AConv downsample: a 2x2 stride-1 average pool, then a 3x3 stride-2 Conv.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.DownSampling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class AConvBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly YoloConv<T> _cv1;

    /// <summary>Creates the block.</summary>
    public AConvBlock([LayerState] int outChannels)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels;
        _cv1 = new YoloConv<T>(outChannels, 3, 2);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _cv1;
    }

    /// <inheritdoc/>
    /// <remarks>Downsampling through a pooling step and a strided convolution; no single-window relation describes it.</remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;
    /// <inheritdoc/>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Convolutional;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        return _cv1.Forward(AiDotNetEngine.Current.AvgPool2D(input, new[] { 2, 2 }, new[] { 1, 1 }));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>YOLOv9's ADown downsample: a 2x2 stride-1 average pool, then the channels split in two: one half through a 3x3 stride-2 Conv, the other through a 3x3 stride-2 max pool and a 1x1 Conv.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.DownSampling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ADownBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly YoloConv<T> _cv1;
    private readonly YoloConv<T> _cv2;

    /// <summary>Creates the block.</summary>
    public ADownBlock([LayerState] int outChannels)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels;
        int half = Math.Max(1, outChannels / 2);
        _cv1 = new YoloConv<T>(half, 3, 2);
        _cv2 = new YoloConv<T>(half, 1);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _cv1; yield return _cv2;
    }

    /// <inheritdoc/>
    /// <remarks>Downsampling through a pooling step and a strided convolution; no single-window relation describes it.</remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;
    /// <inheritdoc/>
    protected internal override ShapeRelationKind OutputShapeRelation => ShapeRelationKind.Convolutional;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        var engine = AiDotNetEngine.Current;
        var x = engine.AvgPool2D(input, new[] { 2, 2 }, new[] { 1, 1 });
        int b = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3], half = c / 2;
        var x1 = engine.TensorSlice(x, new[] { 0, 0, 0, 0 }, new[] { b, half, h, w });
        var x2 = engine.TensorSlice(x, new[] { 0, half, 0, 0 }, new[] { b, c - half, h, w });
        return engine.TensorConcatenate(new[] { _cv1.Forward(x1), _cv2.Forward(CvTensorOps<T>.MaxPoolPadded(x2, 3, 2, 1)) }, axis: 1);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>YOLOv9's SPPELAN: a 1x1 Conv, three chained 5x5 stride-1 max pools, and a 1x1 Conv over the four concatenated maps.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, 4")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SPPELANBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly int _hiddenChannels;
    private readonly YoloConv<T> _cv1;
    private readonly YoloConv<T> _cv5;

    /// <summary>Creates the block.</summary>
    public SPPELANBlock([LayerState] int outChannels, [LayerState] int hiddenChannels)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels; _hiddenChannels = hiddenChannels;
        _cv1 = new YoloConv<T>(hiddenChannels);
        _cv5 = new YoloConv<T>(outChannels);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _cv1; yield return _cv5;
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
        var y1 = CvTensorOps<T>.MaxPoolPadded(x, 5, 1, 2);
        var y2 = CvTensorOps<T>.MaxPoolPadded(y1, 5, 1, 2);
        var y3 = CvTensorOps<T>.MaxPoolPadded(y2, 5, 1, 2);
        return _cv5.Forward(AiDotNetEngine.Current.TensorConcatenate(new[] { x, y1, y2, y3 }, axis: 1));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
