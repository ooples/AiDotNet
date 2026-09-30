using System.IO;
using System.Linq;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// The staged YOLO backbone shared by YOLOv8 and YOLO11: a chain of stages with P3, P4 and P5 tapped at fixed stage
/// indices. Each version supplies only its stage list; extraction, serialization, freezing and metadata live here once,
/// so a fix to any of them reaches both.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class YoloStagedBackboneBase<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private readonly List<LayerBase<T>> _stages = new();
    private int[] _taps = Array.Empty<int>();
    private IReadOnlyList<int> _outputChannels = Array.Empty<int>();

    /// <summary>Creates the network shell; the derived constructor then adds its stages.</summary>
    /// <param name="name">The backbone name.</param>
    /// <param name="inChannels">The number of input image channels.</param>
    /// <remarks>
    /// Stages are added from the derived constructor BODY, after this constructor has armed the layer-initialization
    /// seed scope, so they draw their seeds from it (#2201). Building them before the base call would not.
    /// </remarks>
    protected YoloStagedBackboneBase(string name, int inChannels)
        : base(DetectionBackboneArchitecture<T>.Create(inChannels), new MeanSquaredErrorLoss<T>())
    {
        Name = name;
    }

    /// <summary>Whether the backbone is frozen.</summary>
    public bool IsFrozen { get; private set; }

    /// <summary>The backbone name.</summary>
    public string Name { get; }

    /// <summary>Channels of P3, P4 and P5.</summary>
    public IReadOnlyList<int> OutputChannels => _outputChannels;

    /// <summary>Strides of P3, P4 and P5.</summary>
    public IReadOnlyList<int> Strides => new[] { 8, 16, 32 };

    /// <summary>Appends the next stage.</summary>
    /// <param name="stage">The stage, in execution order.</param>
    protected void AddStage(LayerBase<T> stage) => _stages.Add(stage ?? throw new ArgumentNullException(nameof(stage)));

    /// <summary>Declares the tapped stages and their widths, then finishes construction.</summary>
    /// <param name="taps">Stage indices whose outputs are P3, P4 and P5.</param>
    /// <param name="outputChannels">The channel count of each tapped stage.</param>
    protected void CompleteStages(int[] taps, int[] outputChannels)
    {
        if (taps is null || taps.Length != 3)
            throw new ArgumentException("A YOLO backbone taps exactly P3, P4 and P5.", nameof(taps));
        if (outputChannels is null || outputChannels.Length != taps.Length)
            throw new ArgumentException("Give one channel count per tapped stage.", nameof(outputChannels));
        foreach (int tap in taps)
            if (tap < 0 || tap >= _stages.Count)
                throw new ArgumentOutOfRangeException(nameof(taps), tap, $"There are {_stages.Count} stages.");
        _taps = taps;
        _outputChannels = outputChannels;
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

    /// <summary>The canonical Ultralytics training resolution.</summary>
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
/// The six blocks of a YOLO PAN-FPN neck in execution order, and the width of each output level.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class YoloPanBlocks<T>
{
    /// <summary>Creates the block set.</summary>
    public YoloPanBlocks(LayerBase<T> topDown4, LayerBase<T> topDown3, YoloConv<T> down3,
        LayerBase<T> bottomUp4, YoloConv<T> down4, LayerBase<T> bottomUp5, int[] levelChannels)
    {
        TopDown4 = topDown4 ?? throw new ArgumentNullException(nameof(topDown4));
        TopDown3 = topDown3 ?? throw new ArgumentNullException(nameof(topDown3));
        Down3 = down3 ?? throw new ArgumentNullException(nameof(down3));
        BottomUp4 = bottomUp4 ?? throw new ArgumentNullException(nameof(bottomUp4));
        Down4 = down4 ?? throw new ArgumentNullException(nameof(down4));
        BottomUp5 = bottomUp5 ?? throw new ArgumentNullException(nameof(bottomUp5));
        if (levelChannels is null || levelChannels.Length != 3)
            throw new ArgumentException("A YOLO neck emits exactly P3, P4 and P5.", nameof(levelChannels));
        LevelChannels = levelChannels;
    }

    internal LayerBase<T> TopDown4 { get; }
    internal LayerBase<T> TopDown3 { get; }
    internal YoloConv<T> Down3 { get; }
    internal LayerBase<T> BottomUp4 { get; }
    internal YoloConv<T> Down4 { get; }
    internal LayerBase<T> BottomUp5 { get; }
    internal int[] LevelChannels { get; }
}

/// <summary>
/// The PAN-FPN neck shared by YOLOv8 and YOLO11: top-down nearest upsampling with concatenation and a fusion block,
/// then bottom-up stride-2 Conv with concatenation and a fusion block. The versions differ only in the fusion block
/// (C2f versus C3k2), which each supplies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Partial, with the blocks in readonly fields of THIS type, because the model-parameter generator discovers a neck's
/// weights from the declared fields of a partial type; blocks held anywhere else reach no parameter vector.
/// </remarks>
public abstract partial class YoloPanNeckBase<T> : NeckBase<T>
{
    private readonly string _name;
    private readonly LayerBase<T> _topDown4;
    private readonly LayerBase<T> _topDown3;
    private readonly YoloConv<T> _down3;
    private readonly LayerBase<T> _bottomUp4;
    private readonly YoloConv<T> _down4;
    private readonly LayerBase<T> _bottomUp5;
    private readonly int[] _levelChannels;

    /// <summary>Creates the neck from its blocks.</summary>
    /// <param name="name">The neck name.</param>
    /// <param name="blocks">The six blocks and the level widths.</param>
    protected YoloPanNeckBase(string name, YoloPanBlocks<T> blocks)
    {
        if (blocks is null) throw new ArgumentNullException(nameof(blocks));
        _name = name;
        _topDown4 = blocks.TopDown4;
        _topDown3 = blocks.TopDown3;
        _down3 = blocks.Down3;
        _bottomUp4 = blocks.BottomUp4;
        _down4 = blocks.Down4;
        _bottomUp5 = blocks.BottomUp5;
        _levelChannels = blocks.LevelChannels;
        SetTrainingMode(false);
    }

    private IEnumerable<LayerBase<T>> Blocks()
    {
        yield return _topDown4; yield return _topDown3; yield return _down3;
        yield return _bottomUp4; yield return _down4; yield return _bottomUp5;
    }

    /// <inheritdoc/>
    public override string Name => _name;

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
            throw new ArgumentException($"{_name} takes exactly P3, P4 and P5.", nameof(features));
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
        // Null only during the base constructor, which calls this before the fields are assigned.
        if (_topDown4 is null) return;
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
