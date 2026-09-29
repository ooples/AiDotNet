using System.IO;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using System.Linq;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// Ultralytics YOLO compound scaling (yolov8.yaml, yolo11.yaml): depth and width multipliers and the
/// channel cap applied before the width multiplier.
/// </summary>
internal readonly record struct YoloScale(double Depth, double Width, int MaxChannels, bool ForcesC3k = false)
{
    internal static YoloScale ForV8(ModelSize size) => size switch
    {
        ModelSize.Nano => new(0.33, 0.25, 1024),
        ModelSize.Small => new(0.33, 0.50, 1024),
        ModelSize.Medium => new(0.67, 0.75, 768),
        ModelSize.Large => new(1.00, 1.00, 512),
        ModelSize.XLarge => new(1.00, 1.25, 512),
        _ => throw new ArgumentOutOfRangeException(nameof(size), size, "YOLOv8 publishes no scale for this model size."),
    };

    // yolo11.yaml scales; m, l and x use C3k inner blocks in every C3k2 (ultralytics parse_model).
    internal static YoloScale ForV11(ModelSize size) => size switch
    {
        ModelSize.Nano => new(0.50, 0.25, 1024),
        ModelSize.Small => new(0.50, 0.50, 1024),
        ModelSize.Medium => new(0.50, 1.00, 512, true),
        ModelSize.Large => new(1.00, 1.00, 512, true),
        ModelSize.XLarge => new(1.00, 1.50, 512, true),
        _ => throw new ArgumentOutOfRangeException(nameof(size), size, "YOLO11 publishes no scale for this model size."),
    };

    /// <summary>Output channels of a layer declared with <paramref name="channels"/> at full width.</summary>
    internal int Channels(int channels) => Math.Max(1, (int)Math.Ceiling(Math.Min(channels, MaxChannels) * Width / 8.0) * 8);

    /// <summary>Repeats of a block declared with <paramref name="repeats"/> at full depth.</summary>
    internal int Repeats(int repeats) => Math.Max(1, (int)Math.Round(repeats * Depth));
}

/// <summary>
/// YOLOv8's backbone: Conv stem, four stride-2 Conv + C2f stages, then SPPF (Jocher et al. 2023).
/// </summary>
/// <remarks>
/// P3, P4 and P5 (strides 8, 16, 32) feed the neck. Every Conv is convolution + batch norm + SiLU, and the
/// backbone's C2f bottlenecks use their residual shortcut. The previous YOLOv8 ran a YOLOv4/v5-style CSP
/// backbone without batch norm and without SPPF.
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Ultralytics YOLOv8", "https://github.com/ultralytics/ultralytics", Year = 2023,
    Authors = "Glenn Jocher, Ayush Chaurasia, Jing Qiu")]
[ArchitectureFromPaper("https://github.com/ultralytics/ultralytics",
    "YOLOv8 is published as code; this is the backbone section of its yolov8.yaml.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class YOLOv8Backbone<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private readonly List<LayerBase<T>> _stages = new();
    private readonly int[] _taps;

    /// <summary>Whether the backbone is frozen.</summary>
    public bool IsFrozen { get; private set; }

    /// <summary>The backbone name.</summary>
    public string Name { get; }

    /// <summary>Channels of P3, P4 and P5.</summary>
    public IReadOnlyList<int> OutputChannels { get; }

    /// <summary>Strides of P3, P4 and P5.</summary>
    public IReadOnlyList<int> Strides => new[] { 8, 16, 32 };

    /// <summary>Creates the backbone at a model size.</summary>
    public YOLOv8Backbone(ModelSize size = ModelSize.Nano, int inChannels = 3)
        : base(DetectionBackboneArchitecture<T>.Create(inChannels), new MeanSquaredErrorLoss<T>())
    {
        var s = YoloScale.ForV8(size);
        Name = $"YOLOv8Backbone-{size}";
        int c64 = s.Channels(64), c128 = s.Channels(128), c256 = s.Channels(256), c512 = s.Channels(512), c1024 = s.Channels(1024);
        _stages.Add(new YoloConv<T>(c64, 3, 2));                       // 0  P1/2
        _stages.Add(new YoloConv<T>(c128, 3, 2));                      // 1  P2/4
        _stages.Add(new C2fBlock<T>(c128, s.Repeats(3), true));        // 2
        _stages.Add(new YoloConv<T>(c256, 3, 2));                      // 3  P3/8
        _stages.Add(new C2fBlock<T>(c256, s.Repeats(6), true));        // 4  -> P3
        _stages.Add(new YoloConv<T>(c512, 3, 2));                      // 5  P4/16
        _stages.Add(new C2fBlock<T>(c512, s.Repeats(6), true));        // 6  -> P4
        _stages.Add(new YoloConv<T>(c1024, 3, 2));                     // 7  P5/32
        _stages.Add(new C2fBlock<T>(c1024, s.Repeats(3), true));       // 8
        _stages.Add(new SPPFLayer<T>(c1024, c1024, 5));                // 9  -> P5
        _taps = new[] { 4, 6, 9 };
        OutputChannels = new[] { c256, c512, c1024 };
        EnsureArchitectureInitialized();
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>(3);
        var x = input;
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].Forward(x);
            if (Array.IndexOf(_taps, i) >= 0) features.Add(x);
        }
        return features;
    }

    /// <inheritdoc/>
    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <inheritdoc/>
    public void WriteParameters(BinaryWriter writer)
    {
        foreach (var stage in _stages) BackboneSerialization.WriteLayerParameters(writer, stage);
    }

    /// <inheritdoc/>
    public void ReadParameters(BinaryReader reader)
    {
        foreach (var stage in _stages) BackboneSerialization.ReadLayerParameters(reader, stage);
    }

    /// <summary>Freezes the backbone.</summary>
    public virtual void Freeze() => IsFrozen = true;

    /// <summary>Unfreezes the backbone.</summary>
    public virtual void Unfreeze() => IsFrozen = false;

    /// <summary>YOLOv8's canonical training resolution.</summary>
    public (int Height, int Width) GetExpectedInputSize() => (640, 640);

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) => ExtractFeatures(input)[^1];

    /// <inheritdoc/>
    protected override void InitializeLayers() => Layers.AddRange(_stages);

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = Name,
        AdditionalInfo = new Dictionary<string, object> { ["OutputChannels"] = OutputChannels, ["Strides"] = Strides }
    };

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) =>
        throw new NotSupportedException($"{GetType().Name}: detection backbones train as part of a parent detector.");

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException($"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones.");
}

/// <summary>
/// YOLOv8's PAN-FPN neck: top-down nearest upsampling with concatenation and C2f (no shortcut), then
/// bottom-up stride-2 Conv with concatenation and C2f. Outputs P3, P4 and P5 at their own widths.
/// </summary>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Ultralytics YOLOv8", "https://github.com/ultralytics/ultralytics", Year = 2023,
    Authors = "Glenn Jocher, Ayush Chaurasia, Jing Qiu")]
[ArchitectureFromPaper("https://github.com/ultralytics/ultralytics",
    "YOLOv8 is published as code; this is the head (PAN-FPN) section of its yolov8.yaml.")]
public partial class YOLOv8Neck<T> : NeckBase<T>
{
    private readonly C2fBlock<T> _topDown4;
    private readonly C2fBlock<T> _topDown3;
    private readonly YoloConv<T> _down3;
    private readonly C2fBlock<T> _bottomUp4;
    private readonly YoloConv<T> _down4;
    private readonly C2fBlock<T> _bottomUp5;
    private readonly int[] _levelChannels;

    /// <summary>Creates the neck for a model size.</summary>
    public YOLOv8Neck(ModelSize size)
    {
        var s = YoloScale.ForV8(size);
        int c256 = s.Channels(256), c512 = s.Channels(512), c1024 = s.Channels(1024);
        int n = s.Repeats(3);
        _topDown4 = new C2fBlock<T>(c512, n, false);
        _topDown3 = new C2fBlock<T>(c256, n, false);
        _down3 = new YoloConv<T>(c256, 3, 2);
        _bottomUp4 = new C2fBlock<T>(c512, n, false);
        _down4 = new YoloConv<T>(c512, 3, 2);
        _bottomUp5 = new C2fBlock<T>(c1024, n, false);
        _levelChannels = new[] { c256, c512, c1024 };
        SetTrainingMode(false);
    }

    private IEnumerable<LayerBase<T>> Blocks()
    {
        yield return _topDown4; yield return _topDown3; yield return _down3;
        yield return _bottomUp4; yield return _down4; yield return _bottomUp5;
    }

    /// <inheritdoc/>
    public override string Name => "YOLOv8-PAN";

    /// <inheritdoc/>
    /// <remarks>The widest level; the per-level widths are <see cref="LevelChannels"/>.</remarks>
    public override int OutputChannels => _levelChannels[^1];

    /// <inheritdoc/>
    public override IReadOnlyList<int> LevelChannels => _levelChannels;

    /// <inheritdoc/>
    public override int NumLevels => 3;

    /// <inheritdoc/>
    public override List<Tensor<T>> Forward(List<Tensor<T>> features)
    {
        if (features is null || features.Count != 3)
            throw new ArgumentException("YOLOv8's neck takes exactly P3, P4 and P5.", nameof(features));
        var (p3, p4, p5) = (features[0], features[1], features[2]);
        var engine = AiDotNetEngine.Current;

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
    public override void WriteParameters(BinaryWriter writer)
    {
        foreach (var block in Blocks()) BackboneSerialization.WriteLayerParameters(writer, block);
    }

    /// <inheritdoc/>
    public override void ReadParameters(BinaryReader reader)
    {
        foreach (var block in Blocks()) BackboneSerialization.ReadLayerParameters(reader, block);
    }
}