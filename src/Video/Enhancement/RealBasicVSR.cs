using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// RealBasicVSR: real-world video super-resolution with an image-cleaning module.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is the authors' RealBasicVSRNet (open-mmlab/mmagic, models/editors/real_basicvsr): an image-cleaning
/// module of 20 residual blocks predicts a residue that is added to every low-quality frame, repeated up to three times
/// while the mean residue is large, and the cleaned frames go through BasicVSR with 20 blocks per propagation branch,
/// so noise and compression artifacts are removed before the recurrence can spread them.
/// </para>
/// <para>
/// This class trains the release's first, PSNR-oriented stage (realbasicvsr_wogan): an L1 loss on the output and an L1
/// "cleaning" loss of the cleaned frames against the 4x area-downsampled target, both against the unsharp-masked target,
/// with Adam (0.9, 0.99) at 1e-4 and SPyNet frozen; inference uses a 0.999 moving average of the weights.
/// <see cref="RealBasicVSRSharp{T}"/> is the second, adversarial stage. The release degrades the clean clips with random
/// blur, noise, resizing and compression to make training pairs; here the caller supplies the pairs. SPyNet is loaded
/// pretrained in the release and starts from PyTorch's default initialization here.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real videos from phones or old cameras have noise, blur, and
/// compression artifacts. RealBasicVSR first "cleans" each frame to remove these issues,
/// then upscales the cleaned frames. This prevents noise from spreading through the
/// video during the upscaling process.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 15, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new RealBasicVSR&lt;float&gt;(arch, new RealBasicVSROptions());
/// model.Train(degradedClip, cleanClip);         // [15, 3, 64, 64] -> [15, 3, 256, 256], pixel values 0-255
/// var upscaled = model.Upscale(degradedClip);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Investigating Tradeoffs in Real-World Video Super-Resolution",
    "https://arxiv.org/abs/2111.12704",
    Year = 2022,
    Authors = "Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta2 = 0.99,
                Source = "configs/real_basicvsr/realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py: Adam lr 1e-4, "
                        + "betas (0.9, 0.99), no learning-rate schedule, 300k iterations.")]
public partial class RealBasicVSR<T> : VideoSuperResolutionBase<T>
{
    // RealBasicVSRNet's training threshold: dynamic_refine_thres = 255 levels.
    private const double TrainingCleaningThreshold = 1.0;

    private readonly RealBasicVSROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private RealBasicVsrCore<T>? _core;
    private ParameterAverage<T>? _average;

    /// <summary>Creates a RealBasicVSR model in ONNX inference mode.</summary>
    public RealBasicVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, RealBasicVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new RealBasicVSROptions();
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a RealBasicVSR model in native training mode.</summary>
    public RealBasicVSR(NeuralNetworkArchitecture<T> architecture, RealBasicVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new RealBasicVSROptions();
        Options = _options;
        if (_options.MidChannels < 1) throw new ArgumentOutOfRangeException(nameof(options), "MidChannels must be positive.");
        if (_options.NumPropagationBlocks < 0 || _options.NumCleaningBlocks < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Block counts cannot be negative.");
        _useNativeMode = true;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        InitializeLayers();
        if (_options.EmaMomentum is double momentum) _average = new ParameterAverage<T>(momentum);
        _optimizer = optimizer;
    }

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    #region Layers and network

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.AddRange(Architecture.Layers);
            return;
        }
        _core = new RealBasicVsrCore<T>(Layers, _options.MidChannels, _options.NumPropagationBlocks, _options.NumCleaningBlocks);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (_usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        Core.Run(Engine, Layers, new Tensor<T>(new[] { 1, 2, 3, 32, 32 }), TrainingCleaningThreshold);
    }

    private RealBasicVsrCore<T> Core => _core ?? throw new InvalidOperationException("RealBasicVSR's native layers are not initialized.");

    private (Tensor<T> Output, Tensor<T> Cleaned) RunClips(Tensor<T> frames, double threshold)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        var (output, cleaned) = Core.Run(Engine, Layers, clips, threshold);
        return batched
            ? (output, cleaned)
            : (Engine.Reshape(output, output._shape.Skip(1).ToArray()), Engine.Reshape(cleaned, cleaned._shape.Skip(1).ToArray()));
    }

    private IReadOnlyList<Tensor<T>> AllParameters() => TapeTrainingStep<T>.CollectParameters(Layers);

    /// <summary>Test hook: the cleaned frames and the output for frames scaled to [0, 1], with the trained weights.</summary>
    internal (Tensor<T> Output, Tensor<T> Cleaned) RunNormalizedForTest(Tensor<T> frames, double threshold) => RunClips(frames, threshold);

    /// <summary>Test hook: the cleaning passes the last forward applied.</summary>
    internal int LastCleaningPasses => _core?.LastCleaningPasses ?? 0;

    /// <summary>The SPyNet parameters (never trained).</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters => _core is null ? Array.Empty<Tensor<T>>() : _core.FlowParameters(Layers);

    #endregion

    #region Forward and training

    /// <inheritdoc />
    /// <remarks>The release evaluates its moving average of the weights; cleaning stops below the inference threshold.</remarks>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        double threshold = _options.InferenceCleaningThreshold / 255.0;
        return _average is { HasAverage: true }
            ? _average.WithAverage(AllParameters(), () => RunClips(input, threshold).Output)
            : RunClips(input, threshold).Output;
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunClips(PreprocessFrames(input), TrainingCleaningThreshold).Output;

    // The cleaning passes stop on a host read of the residue, which a compiled plan would bake at trace time.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildPaperOptimizer()
        => _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.99,
                Epsilon = 1e-8,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
            }));

    // RealBasicVSRNet: self.basicvsr.spynet.requires_grad_(False).
    private IReadOnlyList<Tensor<T>> TrainableParameters()
    {
        var flow = new HashSet<Tensor<T>>(FlowParameters);
        return AllParameters().Where(p => !flow.Contains(p)).ToList();
    }

    /// <summary>The target scored by the losses: unsharp-masked when <see cref="RealBasicVSROptions.SharpenTarget"/> is set.</summary>
    internal Tensor<T> LossTarget(Tensor<T> normalizedTarget)
        => _options.SharpenTarget
            ? RealBasicVsrCore<T>.UnsharpMask(normalizedTarget, _options.UnsharpWeight, _options.UnsharpKernelSize, _options.UnsharpThreshold)
            : normalizedTarget;

    /// <inheritdoc />
    /// <remarks>L1 of the output and L1 of the cleaned frames against the 4x area-downsampled target.</remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        var optimizer = BuildPaperOptimizer();
        var target = LossTarget(NormalizeFrames(expectedOutput));
        var cleanTarget = RealBasicVsrCore<T>.AreaDownsample4(Engine, target);
        return new[]
        {
            new TrainingObjective<T>(new[] { input, target, cleanTarget }, data =>
            {
                var (output, cleaned) = RunClips(PreprocessFrames(data[0]), TrainingCleaningThreshold);
                var pixel = Engine.TensorMultiplyScalar(LossFunction.ComputeTapeLoss(output, data[1]), NumOps.FromDouble(_options.PixelLossWeight));
                var cleaning = Engine.TensorMultiplyScalar(LossFunction.ComputeTapeLoss(cleaned, data[2]), NumOps.FromDouble(_options.CleaningLossWeight));
                return Engine.TensorAdd(pixel, cleaning);
            })
            {
                Optimizer = optimizer,
                Parameters = TrainableParameters,
            }
        };
    }

    /// <inheritdoc />
    protected override void OnTrainingStepEnded(bool succeeded)
    {
        base.OnTrainingStepEnded(succeeded);
        if (succeeded && !_usingCustomLayers) _average?.Update(AllParameters());
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? BuildPaperOptimizer() : base.GetOrCreateBaseOptimizer();

    #endregion

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "RealBasicVSR-Native" : "RealBasicVSR-ONNX",
            Description = "RealBasicVSR real-world VSR with image cleaning (Chan et al., CVPR 2022)",
            Complexity = _options.NumPropagationBlocks
        };
        m.AdditionalInfo["MidChannels"] = _options.MidChannels.ToString();
        m.AdditionalInfo["NumPropagationBlocks"] = _options.NumPropagationBlocks.ToString();
        m.AdditionalInfo["NumCleaningBlocks"] = _options.NumCleaningBlocks.ToString();
        m.AdditionalInfo["ScaleFactor"] = "4";
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RealBasicVSR<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}
