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

/// <summary>How a YOLOv9 size builds its auxiliary (PGI) branch.</summary>
internal enum YoloV9AuxiliaryKind
{
    /// <summary>t, s and e: a second top-down path (SPPELAN, then two RepNCSPELAN4) over the primary backbone.</summary>
    TopDown,

    /// <summary>m and c: a reversible auxiliary backbone fed by CBLinear projections of the primary one.</summary>
    ReversibleBackbone,
}

/// <summary>
/// One YOLOv9 size, transcribed from its yaml: ultralytics yolov9{t,s,m,c,e}.yaml for the architecture and
/// WongKinYiu/yolov9 models/detect/yolov9-*.yaml for the PGI auxiliary branch. Block arguments are
/// RepNCSPELAN4's [c2, c3, c4, n], ELAN1's [c2, c3, c4] and SPPELAN's [c2, c3].
/// </summary>
internal sealed record YoloV9Config(
    int Stem0, int Stem1, int[] Stage2, bool Stage2IsElan1, bool UsesADown, int[] Downs, int[][] Stages,
    int[] Spp, int[] Td4, int[] Td3, int[] HeadDowns, int[] Bu4, int[] Bu5,
    YoloV9AuxiliaryKind Auxiliary, bool DualBackbone)
{
    internal static YoloV9Config For(ModelSize size) => size switch
    {
        ModelSize.Nano => new(16, 32, new[] { 32, 32, 16 }, true, false, new[] { 64, 96, 128 },
            new[] { new[] { 64, 64, 32, 3 }, new[] { 96, 96, 48, 3 }, new[] { 128, 128, 64, 3 } },
            new[] { 128, 64 }, new[] { 96, 96, 48, 3 }, new[] { 64, 64, 32, 3 }, new[] { 48, 64 },
            new[] { 96, 96, 48, 3 }, new[] { 128, 128, 64, 3 }, YoloV9AuxiliaryKind.TopDown, false),
        ModelSize.Small => new(32, 64, new[] { 64, 64, 32 }, true, false, new[] { 128, 192, 256 },
            new[] { new[] { 128, 128, 64, 3 }, new[] { 192, 192, 96, 3 }, new[] { 256, 256, 128, 3 } },
            new[] { 256, 128 }, new[] { 192, 192, 96, 3 }, new[] { 128, 128, 64, 3 }, new[] { 96, 128 },
            new[] { 192, 192, 96, 3 }, new[] { 256, 256, 128, 3 }, YoloV9AuxiliaryKind.TopDown, false),
        ModelSize.Medium => new(32, 64, new[] { 128, 128, 64, 1 }, false, false, new[] { 240, 360, 480 },
            new[] { new[] { 240, 240, 120, 1 }, new[] { 360, 360, 180, 1 }, new[] { 480, 480, 240, 1 } },
            new[] { 480, 240 }, new[] { 360, 360, 180, 1 }, new[] { 240, 240, 120, 1 }, new[] { 180, 240 },
            new[] { 360, 360, 180, 1 }, new[] { 480, 480, 240, 1 }, YoloV9AuxiliaryKind.ReversibleBackbone, false),
        ModelSize.XLarge => new(64, 128, new[] { 256, 128, 64, 2 }, false, true, new[] { 256, 512, 1024 },
            new[] { new[] { 512, 256, 128, 2 }, new[] { 1024, 512, 256, 2 }, new[] { 1024, 512, 256, 2 } },
            new[] { 512, 256 }, new[] { 512, 512, 256, 2 }, new[] { 256, 256, 128, 2 }, new[] { 256, 512 },
            new[] { 512, 512, 256, 2 }, new[] { 512, 1024, 512, 2 }, YoloV9AuxiliaryKind.TopDown, true),
        _ => new(64, 128, new[] { 256, 128, 64, 1 }, false, true, new[] { 256, 512, 512 },
            new[] { new[] { 512, 256, 128, 1 }, new[] { 512, 512, 256, 1 }, new[] { 512, 512, 256, 1 } },
            new[] { 512, 256 }, new[] { 512, 512, 256, 1 }, new[] { 256, 256, 128, 1 }, new[] { 256, 512 },
            new[] { 512, 512, 256, 1 }, new[] { 512, 512, 256, 1 }, YoloV9AuxiliaryKind.ReversibleBackbone, false),
    };

    internal static LayerBase<T> Elan<T>(int[] a) => new RepNCSPELAN4Block<T>(a[0], a[1], a[2], a.Length > 3 ? a[3] : 1);

    internal LayerBase<T> Down<T>(int channels) => UsesADown ? new ADownBlock<T>(channels) : new AConvBlock<T>(channels);

    /// <summary>The nine stages stem0, stem1, stage2, down3, stage4 (P3), down5, stage6 (P4), down7, stage8 (P5).</summary>
    internal List<LayerBase<T>> BuildStages<T>() => new()
    {
        new YoloConv<T>(Stem0, 3, 2),
        new YoloConv<T>(Stem1, 3, 2),
        Stage2IsElan1 ? new ELAN1Block<T>(Stage2[0], Stage2[1], Stage2[2]) : Elan<T>(Stage2),
        Down<T>(Downs[0]),
        Elan<T>(Stages[0]),
        Down<T>(Downs[1]),
        Elan<T>(Stages[1]),
        Down<T>(Downs[2]),
        Elan<T>(Stages[2]),
    };

    /// <summary>Output channels of each of the nine stages.</summary>
    internal int[] StageWidths => new[] { Stem0, Stem1, Stage2[0], Downs[0], Stages[0][0], Downs[1], Stages[1][0], Downs[2], Stages[2][0] };
}

/// <summary>
/// A second backbone whose stages are fused with CBLinear projections of the primary backbone (YOLOv9's
/// CBNet-style composite: the auxiliary reversible branch of m/c, and GELAN-e's main path).
/// </summary>
/// <remarks>
/// Source i projects to fusion levels 0..i with one 1x1 convolution per level (the reference's single
/// CBLinear conv split per level computes the same function). At level j the branch adds every source
/// i &gt;= j's level-j projection, nearest-resized to the branch's current map, to its own features (CBFuse).
/// </remarks>
internal sealed class YoloV9CompositeBackbone<T>
{
    private readonly int[] _sourceTaps;
    private readonly int[] _fuseAfter;

    internal List<LayerBase<T>> Stages { get; }

    /// <summary>Projections[i][j]: source i's projection to fusion level j (j &lt;= i).</summary>
    internal List<List<ConvolutionalLayer<T>>> Projections { get; }

    internal YoloV9CompositeBackbone(YoloV9Config config, int[] sourceTaps, int[] fuseAfter)
    {
        _sourceTaps = sourceTaps;
        _fuseAfter = fuseAfter;
        Stages = config.BuildStages<T>();
        var widths = config.StageWidths;
        Projections = sourceTaps.Select((_, i) => Enumerable.Range(0, i + 1)
            .Select(j => new ConvolutionalLayer<T>(widths[fuseAfter[j]], 1, 1, 0, (Interfaces.IActivationFunction<T>?)null))
            .ToList()).ToList();
    }

    internal IEnumerable<LayerBase<T>> Layers() => Stages.Concat(Projections.SelectMany(p => p));

    /// <summary>Runs the branch from the image; returns every stage's output.</summary>
    internal List<Tensor<T>> Forward(Tensor<T> input, IReadOnlyList<Tensor<T>> primaryStages)
    {
        var engine = AiDotNetEngine.Current;
        var projected = _sourceTaps.Select((tap, i) => Projections[i].Select(p => p.Forward(primaryStages[tap])).ToList()).ToList();
        var outputs = new List<Tensor<T>>(Stages.Count);
        var x = input;
        for (int s = 0; s < Stages.Count; s++)
        {
            x = Stages[s].Forward(x);
            int level = Array.IndexOf(_fuseAfter, s);
            if (level >= 0)
            {
                for (int i = level; i < projected.Count; i++)
                    x = engine.TensorAdd(x, CvTensorOps<T>.ResizeNearest(projected[i][level], x.Shape[2], x.Shape[3]));
            }
            outputs.Add(x);
        }
        return outputs;
    }
}

/// <summary>
/// YOLOv9's backbone (Wang et al. 2024): GELAN stages from the size's yaml, plus the PGI auxiliary branch that
/// supervises it during training. GELAN-e's main path is itself a composite (dual) backbone.
/// </summary>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information",
    "https://arxiv.org/abs/2402.13616", Year = 2024, Authors = "Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class YOLOv9Backbone<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private static readonly int[] Taps = { 4, 6, 8 };
    private readonly YoloV9Config _config;
    private readonly List<LayerBase<T>> _primary;
    private readonly YoloV9CompositeBackbone<T>? _dual;
    private readonly YoloV9CompositeBackbone<T>? _auxiliaryBackbone;
    private readonly SPPELANBlock<T>? _auxSpp;
    private readonly LayerBase<T>? _auxTd4;
    private readonly LayerBase<T>? _auxTd3;
    private bool _auxiliaryResolved;

    /// <summary>Whether the backbone is frozen.</summary>
    public bool IsFrozen { get; private set; }

    /// <summary>The backbone name.</summary>
    public string Name { get; }

    /// <summary>Channels of P3, P4 and P5 fed to the neck.</summary>
    public IReadOnlyList<int> OutputChannels { get; }

    /// <summary>Channels of the auxiliary A3, A4 and A5 maps.</summary>
    public IReadOnlyList<int> AuxiliaryChannels { get; }

    /// <summary>Strides of P3, P4 and P5.</summary>
    public IReadOnlyList<int> Strides => new[] { 8, 16, 32 };

    /// <summary>Creates the backbone for a model size (Nano = t, Small = s, Medium = m, Large = c, XLarge = e).</summary>
    public YOLOv9Backbone(ModelSize size = ModelSize.Nano, int inChannels = 3)
        : base(DetectionBackboneArchitecture<T>.Create(inChannels), new MeanSquaredErrorLoss<T>())
    {
        _config = YoloV9Config.For(size);
        Name = $"YOLOv9Backbone-{size}";
        _primary = _config.BuildStages<T>();
        var widths = _config.StageWidths;
        OutputChannels = Taps.Select(t => widths[t]).ToArray();
        if (_config.DualBackbone)
            _dual = new YoloV9CompositeBackbone<T>(_config, new[] { 0, 2, 4, 6, 8 }, new[] { 0, 1, 3, 5, 7 });

        if (_config.Auxiliary == YoloV9AuxiliaryKind.ReversibleBackbone)
        {
            _auxiliaryBackbone = new YoloV9CompositeBackbone<T>(_config, Taps, new[] { 3, 5, 7 });
            AuxiliaryChannels = Taps.Select(t => widths[t]).ToArray();
        }
        else
        {
            _auxSpp = new SPPELANBlock<T>(_config.Spp[0], _config.Spp[1]);
            _auxTd4 = YoloV9Config.Elan<T>(_config.Td4);
            _auxTd3 = YoloV9Config.Elan<T>(_config.Td3);
            AuxiliaryChannels = new[] { _config.Td3[0], _config.Td4[0], _config.Spp[0] };
        }
        EnsureArchitectureInitialized();
        SetTrainingMode(false);
    }

    private IEnumerable<LayerBase<T>> AllModules()
    {
        foreach (var s in _primary) yield return s;
        if (_dual is not null) foreach (var l in _dual.Layers()) yield return l;
        if (_auxiliaryBackbone is not null) foreach (var l in _auxiliaryBackbone.Layers()) yield return l;
        if (_auxSpp is not null) yield return _auxSpp;
        if (_auxTd4 is not null) yield return _auxTd4;
        if (_auxTd3 is not null) yield return _auxTd3;
    }

    private (List<Tensor<T>> Main, List<Tensor<T>> Primary) RunMain(Tensor<T> input)
    {
        var primary = new List<Tensor<T>>(_primary.Count);
        var x = input;
        foreach (var stage in _primary) { x = stage.Forward(x); primary.Add(x); }
        var mainStages = _dual is null ? primary : _dual.Forward(input, primary);
        return (Taps.Select(t => mainStages[t]).ToList(), primary);
    }

    private List<Tensor<T>> RunAuxiliary(Tensor<T> input, List<Tensor<T>> primary)
    {
        if (_auxiliaryBackbone is not null)
        {
            var aux = _auxiliaryBackbone.Forward(input, primary);
            return Taps.Select(t => aux[t]).ToList();
        }

        if (_auxSpp is null || _auxTd4 is null || _auxTd3 is null)
            throw new InvalidOperationException("YOLOv9's top-down auxiliary branch was not built.");
        var engine = AiDotNetEngine.Current;
        var p3 = primary[Taps[0]];
        var p4 = primary[Taps[1]];
        var a5 = _auxSpp.Forward(primary[Taps[2]]);
        var a4 = _auxTd4.Forward(engine.TensorConcatenate(new[] { CvTensorOps<T>.ResizeNearest(a5, p4.Shape[2], p4.Shape[3]), p4 }, axis: 1));
        var a3 = _auxTd3.Forward(engine.TensorConcatenate(new[] { CvTensorOps<T>.ResizeNearest(a4, p3.Shape[2], p3.Shape[3]), p3 }, axis: 1));
        return new List<Tensor<T>> { a3, a4, a5 };
    }

    /// <inheritdoc/>
    /// <remarks>The inference path: the auxiliary branch does not run, apart from one sizing pass on first use.</remarks>
    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var (main, primary) = RunMain(input);
        if (!_auxiliaryResolved)
        {
            // The auxiliary branch trains only, but its lazily sized layers must exist whenever the
            // parameters are enumerated, saved or cloned.
            _ = RunAuxiliary(input, primary);
            _auxiliaryResolved = true;
        }
        return main;
    }

    /// <summary>The training path: the main P3/P4/P5 maps and the auxiliary A3/A4/A5 maps from one pass.</summary>
    internal (List<Tensor<T>> Main, List<Tensor<T>> Auxiliary) ExtractWithAuxiliary(Tensor<T> input)
    {
        var (main, primary) = RunMain(input);
        _auxiliaryResolved = true;
        return (main, RunAuxiliary(input, primary));
    }

    /// <inheritdoc/>
    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <inheritdoc/>
    public void WriteParameters(BinaryWriter writer) { foreach (var m in AllModules()) BackboneSerialization.WriteLayerParameters(writer, m); }

    /// <inheritdoc/>
    public void ReadParameters(BinaryReader reader) { foreach (var m in AllModules()) BackboneSerialization.ReadLayerParameters(reader, m); }

    /// <summary>Freezes the backbone.</summary>
    public virtual void Freeze() => IsFrozen = true;

    /// <summary>Unfreezes the backbone.</summary>
    public virtual void Unfreeze() => IsFrozen = false;

    /// <summary>YOLOv9's canonical training resolution.</summary>
    public (int Height, int Width) GetExpectedInputSize() => (640, 640);

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) => ExtractFeatures(input)[^1];

    /// <inheritdoc/>
    protected override void InitializeLayers() => Layers.AddRange(AllModules());

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = Name,
        AdditionalInfo = new Dictionary<string, object> { ["OutputChannels"] = OutputChannels, ["AuxiliaryChannels"] = AuxiliaryChannels }
    };

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) =>
        throw new NotSupportedException($"{GetType().Name}: detection backbones train as part of a parent detector.");

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException($"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones.");
}

/// <summary>
/// YOLOv9's head (PAN-FPN): SPPELAN on P5, top-down upsample + concat + RepNCSPELAN4, then bottom-up
/// AConv/ADown + concat + RepNCSPELAN4, with every size's own widths.
/// </summary>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information",
    "https://arxiv.org/abs/2402.13616", Year = 2024, Authors = "Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao")]
public partial class YOLOv9Neck<T> : NeckBase<T>
{
    private readonly SPPELANBlock<T> _spp;
    private readonly LayerBase<T> _topDown4;
    private readonly LayerBase<T> _topDown3;
    private readonly LayerBase<T> _down3;
    private readonly LayerBase<T> _bottomUp4;
    private readonly LayerBase<T> _down4;
    private readonly LayerBase<T> _bottomUp5;
    private readonly int[] _levelChannels;

    /// <summary>Creates the head for a model size.</summary>
    public YOLOv9Neck(ModelSize size)
    {
        var c = YoloV9Config.For(size);
        _spp = new SPPELANBlock<T>(c.Spp[0], c.Spp[1]);
        _topDown4 = YoloV9Config.Elan<T>(c.Td4);
        _topDown3 = YoloV9Config.Elan<T>(c.Td3);
        _down3 = c.Down<T>(c.HeadDowns[0]);
        _bottomUp4 = YoloV9Config.Elan<T>(c.Bu4);
        _down4 = c.Down<T>(c.HeadDowns[1]);
        _bottomUp5 = YoloV9Config.Elan<T>(c.Bu5);
        _levelChannels = new[] { c.Td3[0], c.Bu4[0], c.Bu5[0] };
        SetTrainingMode(false);
    }

    private IEnumerable<LayerBase<T>> Blocks()
    {
        yield return _spp; yield return _topDown4; yield return _topDown3; yield return _down3;
        yield return _bottomUp4; yield return _down4; yield return _bottomUp5;
    }

    /// <inheritdoc/>
    public override string Name => "YOLOv9-PAN";

    /// <inheritdoc/>
    public override int OutputChannels => _levelChannels[^1];

    /// <inheritdoc/>
    public override IReadOnlyList<int> LevelChannels => _levelChannels;

    /// <inheritdoc/>
    public override int NumLevels => 3;

    /// <inheritdoc/>
    public override List<Tensor<T>> Forward(List<Tensor<T>> features)
    {
        if (features is null || features.Count != 3)
            throw new ArgumentException("YOLOv9's head takes exactly P3, P4 and P5.", nameof(features));
        var engine = AiDotNetEngine.Current;
        var (p3, p4) = (features[0], features[1]);
        var p5 = _spp.Forward(features[2]);
        var n4 = _topDown4.Forward(engine.TensorConcatenate(new[] { UpTo(p5, p4), p4 }, axis: 1));
        var out3 = _topDown3.Forward(engine.TensorConcatenate(new[] { UpTo(n4, p3), p3 }, axis: 1));
        var out4 = _bottomUp4.Forward(engine.TensorConcatenate(new[] { _down3.Forward(out3), n4 }, axis: 1));
        var out5 = _bottomUp5.Forward(engine.TensorConcatenate(new[] { _down4.Forward(out4), p5 }, axis: 1));
        return new List<Tensor<T>> { out3, out4, out5 };
    }

    private static Tensor<T> UpTo(Tensor<T> x, Tensor<T> target)
        => CvTensorOps<T>.ResizeNearest(x, target.Shape[2], target.Shape[3]);

    /// <inheritdoc/>
    public override void SetTrainingMode(bool training)
    {
        base.SetTrainingMode(training);
        foreach (var block in Blocks()) block.SetTrainingMode(training);
    }

    /// <inheritdoc/>
    public override long GetParameterCount() => Blocks().Sum(b => (long)b.ParameterCount);

    /// <inheritdoc/>
    public override void WriteParameters(BinaryWriter writer) { foreach (var b in Blocks()) BackboneSerialization.WriteLayerParameters(writer, b); }

    /// <inheritdoc/>
    public override void ReadParameters(BinaryReader reader) { foreach (var b in Blocks()) BackboneSerialization.ReadLayerParameters(reader, b); }
}
