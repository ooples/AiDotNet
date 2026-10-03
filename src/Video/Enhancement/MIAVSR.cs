using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// MIA-VSR: masked inter and intra-frame attention for efficient video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MIA-VSR (Zhou et al., CVPR 2024) is a recurrent video super-resolution transformer:
/// - Bidirectional second-order propagation (BasicVSR++'s grid): four branches, backward, forward,
///   backward, forward, each refining every frame from the two frames it already enhanced
/// - Inter-and-intra-frame attention: queries from the current frame attend, inside 8x8 windows,
///   to keys and values from the current frame and the two previous enhanced frames, aligned by
///   SPyNet patch alignment (PSRT)
/// - Adaptive masked processing: each block predicts which positions changed since the previous
///   frame and reuses last frame's result for the rest, trained with a Gumbel-softmax mask and a
///   sparsity loss, and run sparsely at inference
/// - Pixel-shuffle reconstruction plus a bilinear residual
/// </para>
/// <para>
/// <b>For Beginners:</b> MIA-VSR makes video super-resolution faster by being selective.
/// Instead of comparing every pixel with every other pixel in neighboring frames (which is
/// very slow), it uses "masks" to focus only on the most important parts. It has two types:
/// inter-frame masks find the best matching regions across time, and intra-frame masks
/// enhance spatial details within each frame.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new MIAVSR&lt;float&gt;(arch, "miavsr.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "MIA-VSR: Masked Inter and Intra-Frame Attention for Video
/// Super-Resolution" (Zhou et al., CVPR 2024)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Video Super-Resolution Transformer with Masked Inter&Intra-Frame Attention",
    "https://arxiv.org/abs/2401.06312",
    Year = 2024,
    Authors = "Xingyu Zhou, Leheng Zhang, Xiaorui Zhao, Keze Wang, Leida Li, Shuhang Gu")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, ReferenceBatchSize = 24,
                Phase = TrainingPhase.PreTraining,
                Source = "Zhou et al. 2024, Sec. 4.1: Adam at a batch size of 24, following the "
                        + "settings of BasicVSR++ for 600K iterations with an initial learning rate of "
                        + "2e-4.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4,
                Phase = TrainingPhase.FineTuning,
                Source = "Zhou et al. 2024, Sec. 4.1: a further 300K iterations on REDS from the "
                        + "well-trained model, at an initial learning rate of 1e-4.")]
public partial class MIAVSR<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    // The reconstruction loss of the paper, sqrt(||I_hat - I||^2 + eps^2) with eps = 1e-3 (Sec. 3.3).
    private const double CharbonnierEpsilon = 1e-3;

    private readonly MIAVSROptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // The paper network; null in ONNX mode or when the caller supplied its own layers.
    private MiaVsrNetwork<T>? _network;

    // The pretrained, frozen flow estimator for patch alignment; null aligns with zero motion. The caller
    // owns its weights: they are not this model's parameters, so they are never counted, optimized or
    // restored here.
    [AiDotNet.Attributes.ExternalState]
    private readonly SpyNetLayer<T>? _flowEstimator;

    // Gumbel noise for the training-time masks, seeded from the options when a seed is given.
    private readonly Random _random;

    // λ·L_mask from the latest training forward, consumed once by the objective.
    private Tensor<T>? _pendingMaskLoss;

    #endregion

    #region Constructors

    /// <summary>Creates a MIA-VSR model in ONNX inference mode.</summary>
    public MIAVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, MIAVSROptions? options = null)
        : base(architecture, new CharbonnierLoss<T>(CharbonnierEpsilon))
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new MIAVSROptions();
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a MIA-VSR model in native training mode.</summary>
    /// <param name="architecture">The network architecture; InputDepth is the frame channel count.</param>
    /// <param name="options">The model options; the defaults are the paper's.</param>
    /// <param name="optimizer">The training optimizer; the paper's Adam when null.</param>
    /// <param name="flowEstimator">A pretrained SPyNet for PSRT patch alignment, used frozen. Without one,
    /// neighbouring frames are attended in place (zero motion) rather than aligned by an untrained estimate.</param>
    public MIAVSR(NeuralNetworkArchitecture<T> architecture, MIAVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        SpyNetLayer<T>? flowEstimator = null)
        : base(architecture, new CharbonnierLoss<T>(CharbonnierEpsilon))
    {
        _options = options ?? new MIAVSROptions();
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
        _useNativeMode = true;
        _flowEstimator = flowEstimator;
        // The rate this model publishes on its own options. Built bare, the optimizer
        // would use its own default instead and LearningRate would be configuration that
        // nothing reads — the defect that diverged MusicFlamingo's training.
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
        new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        { InitialLearningRate = _options.LearningRate });
        ScaleFactor = _options.ScaleFactor;
        InitializeLayers();
    }

    #endregion

    #region Video Super-Resolution

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : ForwardNative(preprocessed, training: false);
        return PostprocessOutput(output);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        // The paper topology: shallow features, SPyNet patch alignment, four propagation branches of
        // inter-and-intra-frame attention blocks, and the pixel-shuffle head. Layers publishes exactly
        // the instances the forward runs, so training, serialization and clone walk the same weights.
        int channels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        _network = new MiaVsrNetwork<T>(_options, channels, _flowEstimator);
        Layers.AddRange(_network.Layers);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return RunOnnxInference(input);
        return ForwardNative(input, training: false);
    }

    /// <inheritdoc />
    /// <remarks>
    /// The training forward samples Gumbel keep-masks and records their mean, which
    /// <see cref="ConsumeAuxiliaryTapeLoss"/> hands to the objective as λ·L_mask.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => ForwardNative(input, training: true);

    /// <inheritdoc />
    /// <remarks>The mask-sparsity term of the paper's objective, L = L_sr + λ·L_mask (Sec. 3.3).</remarks>
    protected override Tensor<T>? ConsumeAuxiliaryTapeLoss()
    {
        var term = _pendingMaskLoss;
        _pendingMaskLoss = null;
        return term;
    }

    /// <inheritdoc />
    /// <remarks>
    /// MIA-VSR's graph is dynamic: it recurs over frames, samples fresh Gumbel noise every step and
    /// keeps or drops positions by a data-dependent mask. A traced and replayed graph would freeze one
    /// step's masks and noise, so training runs on the eager tape.
    /// </remarks>
    protected override bool SupportsFusedCompiledTraining => false;

    private bool _lazyShapesProbed;
    private bool _lazyShapesProbing;

    /// <inheritdoc />
    /// <remarks>
    /// The base walk feeds the architecture's shape through Layers as one sequential chain, which this
    /// network is not: frames are split, flow is estimated per pair, and tokens are windowed. Resolve
    /// through the real forward on a two-frame clip at the smallest size the flow estimator accepts.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (IsOnnxMode) return;
        if (_network is null)
        {
            // A caller-supplied stack is an ordinary sequential chain, which the base walk resolves.
            base.ResolveLazyLayerShapes();
            return;
        }

        if (_lazyShapesProbed || _lazyShapesProbing) return;
        _lazyShapesProbing = true;

        int channels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        int side = Math.Max(_options.WindowSize, _network.MinimumFlowSize);
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            _ = ForwardNative(new Tensor<T>(new[] { 1, 2, channels, side, side }), training: false);
            // Only a probe that completed has resolved the shapes; a failed one is retried next time.
            _lazyShapesProbed = true;
        }
        finally
        {
            _lazyShapesProbing = false;
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <summary>The paper network, for tests that inspect its inference modes; null in ONNX mode.</summary>
    internal MiaVsrNetwork<T>? Network => _network;

    private Tensor<T> ForwardNative(Tensor<T> input, bool training)
    {
        if (_network is null)
        {
            // A caller-supplied architecture is an ordinary layer chain.
            var output = input;
            foreach (var layer in Layers) output = layer.Forward(output);
            return output;
        }

        _network.BindTo(Layers);
        var result = _network.Forward(input, training, _random, out var maskLoss);
        _pendingMaskLoss = training && maskLoss is not null && _options.MaskLossWeight > 0
            ? Engine.TensorMultiplyScalar(maskLoss, NumOps.FromDouble(_options.MaskLossWeight))
            : null;
        return result;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MIAVSR-Native" : "MIAVSR-ONNX",
            Description = $"MIA-VSR {_options.Variant} masked inter/intra-frame attention VSR (Zhou et al., CVPR 2024)",
            Complexity = _options.NumPropagationBranches * _options.BlocksPerBranch
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumPropagationBranches"] = _options.NumPropagationBranches.ToString();
        m.AdditionalInfo["BlocksPerBranch"] = _options.BlocksPerBranch.ToString();
        m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["MaskLossWeight"] = _options.MaskLossWeight.ToString(System.Globalization.CultureInfo.InvariantCulture);
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MIAVSR<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
