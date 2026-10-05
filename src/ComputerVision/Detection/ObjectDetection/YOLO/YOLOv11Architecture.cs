using System.IO;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using System.Linq;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;
/// <summary>YOLO11's C3k: a CSP block (C3) of <c>depth</c> 3x3 bottlenecks over half the width, concatenated with a 1x1 bypass.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, 2, true")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class C3kBlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly int _depth;
    private readonly bool _shortcut;
    private readonly YoloConv<T> _cv1;
    private readonly YoloConv<T> _cv2;
    private readonly YoloConv<T> _cv3;
    private readonly List<YoloBottleneck<T>> _blocks;

    /// <summary>Creates the block.</summary>
    public C3kBlock([LayerState] int outChannels, [LayerState] int depth = 2, [LayerState] bool shortcut = true)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels; _depth = depth; _shortcut = shortcut;
        int hidden = Math.Max(1, outChannels / 2);
        _cv1 = new YoloConv<T>(hidden);
        _cv2 = new YoloConv<T>(hidden);
        _cv3 = new YoloConv<T>(outChannels);
        _blocks = Enumerable.Range(0, Math.Max(1, depth)).Select(_ => new YoloBottleneck<T>(hidden, shortcut)).ToList();
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
/// <summary>YOLO11's C3k2: the C2f layout (split, every inner block's output concatenated), with C3k inner blocks when <c>c3k</c> is set and plain bottlenecks otherwise.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, 1, false, 0.5")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class C3k2Block<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outChannels;
    private readonly int _depth;
    private readonly bool _c3k;
    private readonly double _expansion;
    private readonly bool _shortcut;
    private readonly YoloConv<T> _cv1a;
    private readonly YoloConv<T> _cv1b;
    private readonly List<LayerBase<T>> _blocks;
    private readonly YoloConv<T> _cv2;

    /// <summary>Creates the block.</summary>
    public C3k2Block([LayerState] int outChannels, [LayerState] int depth, [LayerState] bool c3k, [LayerState] double expansion = 0.5, [LayerState] bool shortcut = true)
        : base(new[] { -1, -1, -1 }, new[] { outChannels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _outChannels = outChannels; _depth = depth; _c3k = c3k; _expansion = expansion; _shortcut = shortcut;
        int hidden = Math.Max(1, (int)(outChannels * expansion));
        _cv1a = new YoloConv<T>(hidden);
        _cv1b = new YoloConv<T>(hidden);
        _blocks = Enumerable.Range(0, Math.Max(1, depth))
            // Ultralytics C3k2 builds its plain inner blocks as Bottleneck(c, c, shortcut, g), i.e. Bottleneck's
            // default e = 0.5; only C2f and C3k pass e = 1.0. YoloBottleneck defaults to 1.0 (the C2f case).
            .Select(_ => c3k ? (LayerBase<T>)new C3kBlock<T>(hidden, 2, shortcut) : new YoloBottleneck<T>(hidden, shortcut, expansion: 0.5))
            .ToList();
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
/// <summary>YOLO11's PSABlock: multi-head self-attention with a depthwise positional encoding on the values, then a 2x FFN, each added back to its input.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, 1")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PSABlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly int _heads;
    private readonly int _keyDim;
    private readonly YoloConv<T> _q;
    private readonly YoloConv<T> _k;
    private readonly YoloConv<T> _v;
    private readonly YoloConv<T> _pe;
    private readonly YoloConv<T> _proj;
    private readonly YoloConv<T> _ffn1;
    private readonly YoloConv<T> _ffn2;

    /// <summary>Creates the block.</summary>
    public PSABlock([LayerState] int channels, [LayerState] int heads)
        : base(new[] { -1, -1, -1 }, new[] { channels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _channels = channels; _heads = Math.Max(1, heads);
        int headDim = Math.Max(1, channels / _heads);
        _keyDim = Math.Max(1, (int)(headDim * 0.5)); // attn_ratio 0.5
        // The reference qkv is one 1x1 Conv whose output is split per head; batch norm is per channel, so
        // three Convs over the same channel partition compute the same function.
        _q = new YoloConv<T>(_heads * _keyDim, act: false);
        _k = new YoloConv<T>(_heads * _keyDim, act: false);
        _v = new YoloConv<T>(channels, act: false);
        _pe = new YoloConv<T>(channels, 3, groups: channels, act: false);
        _proj = new YoloConv<T>(channels, act: false);
        _ffn1 = new YoloConv<T>(channels * 2);
        _ffn2 = new YoloConv<T>(channels, act: false);
    }

    internal IEnumerable<LayerBase<T>> Children()
    {
        yield return _q; yield return _k; yield return _v; yield return _pe; yield return _proj;
        yield return _ffn1; yield return _ffn2;
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
        var engine = AiDotNetEngine.Current;
        int batch = input.Shape[0], height = input.Shape[2], width = input.Shape[3], n = height * width;
        int headDim = _channels / _heads;
        // [B, heads*d, H, W] -> [B, heads, N, d], the layout Engine.ScaledDotProductAttention takes.
        var q = engine.TensorPermute(engine.Reshape(_q.Forward(input), new[] { batch, _heads, _keyDim, n }), new[] { 0, 1, 3, 2 });
        var k = engine.TensorPermute(engine.Reshape(_k.Forward(input), new[] { batch, _heads, _keyDim, n }), new[] { 0, 1, 3, 2 });
        var vMap = _v.Forward(input);
        var v = engine.TensorPermute(engine.Reshape(vMap, new[] { batch, _heads, headDim, n }), new[] { 0, 1, 3, 2 });
        // The fused attention op MultiHeadAttentionLayer uses. A hand-built BatchMatMul -> Softmax chain
        // left q and k with exactly zero gradient although a finite-difference probe showed they move the
        // output, so the tape was dropping them.
        var context = engine.ScaledDotProductAttention(q, k, v, mask: null, scale: 1.0 / Math.Sqrt(_keyDim), out _);
        var attended = engine.Reshape(engine.TensorPermute(context, new[] { 0, 1, 3, 2 }), new[] { batch, _channels, height, width });
        var x = BackboneOps<T>.AddResidual(_proj.Forward(engine.TensorAdd(attended, _pe.Forward(vMap))), input);
        return BackboneOps<T>.AddResidual(_ffn2.Forward(_ffn1.Forward(x)), x);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>YOLO11's C2PSA: split in two, run <c>depth</c> PSABlocks on one half, and fuse both halves with a 1x1 Conv.</summary>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 8, 8, 8", TestConstructorArgs = "8, 1")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class C2PSABlock<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly int _depth;
    private readonly YoloConv<T> _cv1a;
    private readonly YoloConv<T> _cv1b;
    private readonly List<PSABlock<T>> _blocks;
    private readonly YoloConv<T> _cv2;

    /// <summary>Creates the block.</summary>
    public C2PSABlock([LayerState] int channels, [LayerState] int depth)
        : base(new[] { -1, -1, -1 }, new[] { channels, -1, -1 }, (IActivationFunction<T>)new IdentityActivation<T>())
    {
        _channels = channels; _depth = depth;
        int hidden = Math.Max(1, channels / 2);
        _cv1a = new YoloConv<T>(hidden);
        _cv1b = new YoloConv<T>(hidden);
        _blocks = Enumerable.Range(0, Math.Max(1, depth)).Select(_ => new PSABlock<T>(hidden, Math.Max(1, hidden / 64))).ToList();
        _cv2 = new YoloConv<T>(channels);
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
        var b = _cv1b.Forward(input);
        foreach (var block in _blocks) b = block.Forward(b);
        return _cv2.Forward(AiDotNetEngine.Current.TensorConcatenate(new[] { _cv1a.Forward(input), b }, axis: 1));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { foreach (var l in Children()) l.UpdateParameters(learningRate); }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining) { base.SetTrainingMode(isTraining); foreach (var l in Children()) l.SetTrainingMode(isTraining); }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var l in Children()) l.ResetState(); }
}
/// <summary>
/// YOLO11's backbone (ultralytics yolo11.yaml): Conv/C3k2 stages, SPPF, then C2PSA attention on P5.
/// </summary>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Ultralytics YOLO11", "https://docs.ultralytics.com/models/yolo11/", Year = 2024,
    Authors = "Glenn Jocher, Jing Qiu")]
[ArchitectureFromPaper("https://docs.ultralytics.com/models/yolo11/",
    "YOLO11 is published as code; this is the backbone section of its yolo11.yaml.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class YOLOv11Backbone<T> : YoloStagedBackboneBase<T>
{
    /// <summary>Creates the backbone.</summary>
    /// <param name="options">Model size and input channels; defaults to Nano over three channels.</param>
    public YOLOv11Backbone(YoloBackboneOptions? options = null)
        : base($"YOLOv11Backbone-{YoloBackboneOptions.OrDefault(options).Size}", YoloBackboneOptions.OrDefault(options).InChannels)
    {
        var resolved = YoloBackboneOptions.OrDefault(options);
        resolved.Validate();
        var s = YoloScale.ForV11(resolved.Size);
        bool c3k = s.ForcesC3k;
        int c64 = s.Channels(64), c128 = s.Channels(128), c256 = s.Channels(256), c512 = s.Channels(512), c1024 = s.Channels(1024);
        int n = s.Repeats(2);
        AddStage(new YoloConv<T>(c64, 3, 2));                        // 0  P1/2
        AddStage(new YoloConv<T>(c128, 3, 2));                       // 1  P2/4
        AddStage(new C3k2Block<T>(c256, n, c3k, 0.25));              // 2
        AddStage(new YoloConv<T>(c256, 3, 2));                       // 3  P3/8
        AddStage(new C3k2Block<T>(c512, n, c3k, 0.25));              // 4  -> P3
        AddStage(new YoloConv<T>(c512, 3, 2));                       // 5  P4/16
        AddStage(new C3k2Block<T>(c512, n, true));                   // 6  -> P4
        AddStage(new YoloConv<T>(c1024, 3, 2));                      // 7  P5/32
        AddStage(new C3k2Block<T>(c1024, n, true));                  // 8
        AddStage(new SPPFLayer<T>(c1024, c1024, 5));                 // 9
        AddStage(new C2PSABlock<T>(c1024, n));                       // 10 -> P5
        CompleteStages(new[] { 4, 6, 10 }, new[] { c512, c512, c1024 });
    }
}

/// <summary>YOLO11's PAN-FPN neck: YOLOv8's topology with C3k2 in place of C2f.</summary>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Ultralytics YOLO11", "https://docs.ultralytics.com/models/yolo11/", Year = 2024,
    Authors = "Glenn Jocher, Jing Qiu")]
[ArchitectureFromPaper("https://docs.ultralytics.com/models/yolo11/",
    "YOLO11 is published as code; this is the head (PAN-FPN) section of its yolo11.yaml.")]
public partial class YOLOv11Neck<T> : YoloPanNeckBase<T>
{
    /// <summary>Creates the neck for a model size.</summary>
    public YOLOv11Neck(ModelSize size)
        : base("YOLO11-PAN", BuildBlocks(size))
    {
    }

    private static YoloPanBlocks<T> BuildBlocks(ModelSize size)
    {
        var s = YoloScale.ForV11(size);
        bool c3k = s.ForcesC3k;
        int c256 = s.Channels(256), c512 = s.Channels(512), c1024 = s.Channels(1024);
        int n = s.Repeats(2);
        return new YoloPanBlocks<T>(
            topDown4: new C3k2Block<T>(c512, n, c3k),
            topDown3: new C3k2Block<T>(c256, n, c3k),
            down3: new YoloConv<T>(c256, 3, 2),
            bottomUp4: new C3k2Block<T>(c512, n, c3k),
            down4: new YoloConv<T>(c512, 3, 2),
            bottomUp5: new C3k2Block<T>(c1024, n, true),
            levelChannels: new[] { c256, c512, c1024 });
    }
}