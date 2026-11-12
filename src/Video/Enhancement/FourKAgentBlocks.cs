using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// One super-resolution expert in 4KAgent's toolbox: a native paper network built into the agent's layer list, with its
/// paper's first training stage.
/// </summary>
internal abstract class FourKAgentExpert<T>
{
    private readonly int _first;
    private int _count;

    /// <summary>Starts the expert's span at the end of <paramref name="layers"/>; the subclass builds, then calls
    /// <see cref="Seal"/>.</summary>
    protected FourKAgentExpert(List<ILayer<T>> layers) => _first = layers.Count;

    protected void Seal(List<ILayer<T>> layers) => _count = layers.Count - _first;

    /// <summary>The expert's name, also the component its optimizer is declared under.</summary>
    internal abstract string Name { get; }

    /// <summary>Whether the expert is fidelity-oriented (pixel-loss trained) rather than perception-oriented.</summary>
    internal abstract bool Fidelity { get; }

    /// <summary>Its paper's Adam learning rate.</summary>
    internal abstract double LearningRate { get; }

    /// <summary>The rate multiplier of its optical-flow network (1 when the paper trains it at the base rate).</summary>
    internal virtual double FlowRateMultiplier => 1.0;

    /// <summary>Training steps for which <see cref="FrozenEarly"/> stays fixed.</summary>
    internal virtual int FreezeIterations => 0;

    /// <summary>The cosine schedule's iterations and floor, or null for a constant rate.</summary>
    internal virtual (int Iterations, double Minimum)? Cosine => null;

    /// <summary>Every parameter of the expert.</summary>
    internal IReadOnlyList<Tensor<T>> Parameters(IReadOnlyList<ILayer<T>> layers)
        => TapeTrainingStep<T>.CollectParameters(layers.Skip(_first).Take(_count));

    /// <summary>The optical-flow parameters, trained at <see cref="FlowRateMultiplier"/>.</summary>
    internal abstract IReadOnlyList<Tensor<T>> Flow(IReadOnlyList<ILayer<T>> layers);

    /// <summary>Parameters held fixed for the first <see cref="FreezeIterations"/> steps.</summary>
    internal virtual IReadOnlyList<Tensor<T>> FrozenEarly(IReadOnlyList<ILayer<T>> layers) => Flow(layers);

    /// <summary>Parameters the paper never trains.</summary>
    internal virtual IReadOnlyList<Tensor<T>> AlwaysFrozen(IReadOnlyList<ILayer<T>> layers) => Array.Empty<Tensor<T>>();

    /// <summary>The parameters trained at step <paramref name="step"/> (1-based).</summary>
    internal IReadOnlyList<Tensor<T>> Trainable(IReadOnlyList<ILayer<T>> layers, int step)
    {
        var frozen = new HashSet<Tensor<T>>(AlwaysFrozen(layers));
        if (step <= FreezeIterations) frozen.UnionWith(FrozenEarly(layers));
        return Parameters(layers).Where(p => !frozen.Contains(p)).ToList();
    }

    /// <summary>Clips [N, T, 3, H, W] in [0, 1] to [N, T, 3, 4H, 4W].</summary>
    internal abstract Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips);

    /// <summary>The first training stage's loss on clips and 4x targets in [0, 1].</summary>
    internal abstract Tensor<T> Loss(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips, Tensor<T> target);

    // Frames [..., 3, H, W] as a batch of images [B, 3, H, W].
    protected static Tensor<T> AsImages(IEngine engine, Tensor<T> frames)
    {
        int rank = frames.Rank, h = frames.Shape[rank - 2], w = frames.Shape[rank - 1];
        return engine.Reshape(frames, new[] { frames.Length / (3 * h * w), 3, h, w });
    }

    protected static Tensor<T> Weighted(IEngine engine, Tensor<T> loss, double weight)
        => engine.TensorMultiplyScalar(loss, MathHelper.GetNumericOperations<T>().FromDouble(weight));
}

/// <summary>
/// BasicVSR (Chan et al. 2021; configs/basicvsr/basicvsr_2xb4_reds4.py): Charbonnier loss, Adam (0.9, 0.99) at 2e-4 on
/// a cosine schedule, SPyNet at an eighth of the rate and frozen for the first 5000 iterations.
/// </summary>
internal sealed class FourKAgentBasicVsr<T> : FourKAgentExpert<T>
{
    private readonly BasicVSROptions _options;
    private readonly CharbonnierLoss<T> _loss;
    private readonly BasicVsrBackbone<T> _backbone;

    internal FourKAgentBasicVsr(List<ILayer<T>> layers, BasicVSROptions options) : base(layers)
    {
        _options = options;
        _loss = new CharbonnierLoss<T>(options.CharbonnierEpsilon);
        _backbone = new BasicVsrBackbone<T>(layers, options.MidChannels, options.NumResBlocks);
        Seal(layers);
    }

    internal override string Name => "basicvsr";
    internal override bool Fidelity => true;
    internal override double LearningRate => _options.LearningRate;
    internal override double FlowRateMultiplier => _options.SpyNetLearningRateMultiplier;
    internal override int FreezeIterations => _options.FlowFreezeIterations;
    internal override (int Iterations, double Minimum)? Cosine => (_options.CosineIterations, _options.MinLearningRate);
    internal override IReadOnlyList<Tensor<T>> Flow(IReadOnlyList<ILayer<T>> layers) => _backbone.FlowParameters(layers);
    internal override Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips) => _backbone.Run(engine, layers, clips);

    internal override Tensor<T> Loss(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips, Tensor<T> target)
        => _loss.ComputeTapeLoss(Run(engine, layers, clips), target);
}

/// <summary>
/// IconVSR (Chan et al. 2021; configs/iconvsr/iconvsr_2xb4_reds4.py): BasicVSR's recipe, with the EDVR keyframe
/// extractor frozen alongside SPyNet for the first 5000 iterations.
/// </summary>
internal sealed class FourKAgentIconVsr<T> : FourKAgentExpert<T>
{
    private readonly IconVSROptions _options;
    private readonly CharbonnierLoss<T> _loss;
    private readonly IconVsrBackbone<T> _backbone;

    internal FourKAgentIconVsr(List<ILayer<T>> layers, IconVSROptions options) : base(layers)
    {
        _options = options;
        _loss = new CharbonnierLoss<T>(options.CharbonnierEpsilon);
        _backbone = new IconVsrBackbone<T>(layers, options.MidChannels, options.NumResBlocks, options.KeyframeStride,
            options.RefillPadding, options.EdvrChannels, options.DeformableGroups, options.EdvrExtractionBlocks);
        Seal(layers);
    }

    internal override string Name => "iconvsr";
    internal override bool Fidelity => true;
    internal override double LearningRate => _options.LearningRate;
    internal override double FlowRateMultiplier => _options.SpyNetLearningRateMultiplier;
    internal override int FreezeIterations => _options.FlowFreezeIterations;
    internal override (int Iterations, double Minimum)? Cosine => (_options.CosineIterations, _options.MinLearningRate);
    internal override IReadOnlyList<Tensor<T>> Flow(IReadOnlyList<ILayer<T>> layers) => _backbone.FlowParameters(layers);

    internal override IReadOnlyList<Tensor<T>> FrozenEarly(IReadOnlyList<ILayer<T>> layers)
        => _backbone.FlowParameters(layers).Concat(_backbone.EdvrParameters(layers)).ToList();

    internal override Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips) => _backbone.Run(engine, layers, clips);

    internal override Tensor<T> Loss(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips, Tensor<T> target)
        => _loss.ComputeTapeLoss(Run(engine, layers, clips), target);
}

/// <summary>
/// RealBasicVSR (Chan et al. 2022; configs/real_basicvsr/realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py):
/// L1 on the output and L1 on the cleaned frames against the 4x area-downsampled (unsharp-masked) target, Adam (0.9,
/// 0.99) at 1e-4, SPyNet never trained. At inference the cleaning stops at the release's 1.5-level residue.
/// </summary>
internal sealed class FourKAgentRealBasicVsr<T> : FourKAgentExpert<T>
{
    // RealBasicVSRNet's training threshold: dynamic_refine_thres = 255 levels.
    private const double TrainingCleaningThreshold = 1.0;
    private readonly RealBasicVSROptions _options;
    private readonly MeanAbsoluteErrorLoss<T> _loss = new();
    private readonly RealBasicVsrCore<T> _core;

    internal FourKAgentRealBasicVsr(List<ILayer<T>> layers, RealBasicVSROptions options) : base(layers)
    {
        _options = options;
        _core = new RealBasicVsrCore<T>(layers, options.MidChannels, options.NumPropagationBlocks, options.NumCleaningBlocks);
        Seal(layers);
    }

    internal override string Name => "realbasicvsr";
    internal override bool Fidelity => false;
    internal override double LearningRate => _options.LearningRate;
    internal override IReadOnlyList<Tensor<T>> Flow(IReadOnlyList<ILayer<T>> layers) => _core.FlowParameters(layers);
    internal override IReadOnlyList<Tensor<T>> FrozenEarly(IReadOnlyList<ILayer<T>> layers) => Array.Empty<Tensor<T>>();
    internal override IReadOnlyList<Tensor<T>> AlwaysFrozen(IReadOnlyList<ILayer<T>> layers) => _core.FlowParameters(layers);

    internal override Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips)
        => _core.Run(engine, layers, clips, _options.InferenceCleaningThreshold / 255.0).Output;

    internal override Tensor<T> Loss(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips, Tensor<T> target)
    {
        var sharpened = _options.SharpenTarget
            ? RealBasicVsrCore<T>.UnsharpMask(target, _options.UnsharpWeight, _options.UnsharpKernelSize, _options.UnsharpThreshold)
            : target;
        var (output, cleaned) = _core.Run(engine, layers, clips, TrainingCleaningThreshold);
        var pixel = Weighted(engine, _loss.ComputeTapeLoss(output, sharpened), _options.PixelLossWeight);
        var cleaning = Weighted(engine, _loss.ComputeTapeLoss(cleaned, RealBasicVsrCore<T>.AreaDownsample4(engine, sharpened)),
            _options.CleaningLossWeight);
        return engine.TensorAdd(pixel, cleaning);
    }
}

/// <summary>
/// RealViformer (Zhang and Yao 2024): the reconstruction stage, Charbonnier plus 0.001 SSIM, Adam (0.9, 0.99) at 1e-4,
/// SPyNet frozen for the first 5000 iterations.
/// </summary>
internal sealed class FourKAgentRealViformer<T> : FourKAgentExpert<T>
{
    private readonly RealViformerOptions _options;
    private readonly CharbonnierLoss<T> _loss;
    private readonly RealViformerBackbone<T> _backbone;

    internal FourKAgentRealViformer(List<ILayer<T>> layers, RealViformerOptions options) : base(layers)
    {
        _options = options;
        _loss = new CharbonnierLoss<T>(options.CharbonnierEpsilon);
        _backbone = new RealViformerBackbone<T>(layers, options.NumFeatures, options.NumBlocks, options.Heads,
            options.FfnExpansionFactor, options.MergeHeads, options.SqueezeFactor, options.ChannelMask);
        Seal(layers);
    }

    internal override string Name => "realviformer";
    internal override bool Fidelity => false;
    internal override double LearningRate => _options.LearningRate;
    internal override int FreezeIterations => _options.FlowFreezeIterations;
    internal override IReadOnlyList<Tensor<T>> Flow(IReadOnlyList<ILayer<T>> layers) => _backbone.FlowParameters(layers);
    internal override Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips) => _backbone.Run(engine, layers, clips);

    internal override Tensor<T> Loss(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> clips, Tensor<T> target)
    {
        var output = Run(engine, layers, clips);
        var loss = Weighted(engine, _loss.ComputeTapeLoss(output, target), _options.CharbonnierWeight);
        return _options.SsimWeight > 0
            ? engine.TensorAdd(loss, Weighted(engine, RealViformerBlocks.SsimLoss(engine, AsImages(engine, output), AsImages(engine, target)),
                _options.SsimWeight))
            : loss;
    }
}
