using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// RealViformer: a recurrent transformer for real-world video super-resolution built on channel attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is the authors' RealViformer for x4 real-world video super-resolution (Zhang and Yao, "RealViformer:
/// Investigating Attention for Real-World Video Super-Resolution", ECCV 2024; Yuehan717/RealViformer,
/// archs/realviformer_arch.py and the configuration in inference_realviformer.py). A unidirectional recurrent loop warps
/// the hidden state to each frame with SPyNet flow, merges it with the frame's shallow features through channel cross-
/// attention (the CAF), and reconstructs the result with a three-level Restormer encoder-decoder ([2, 3, 4] blocks with
/// [48, 96, 192] channels and one refinement block) whose transposed attention is squeezed by 4 and re-weighted by a
/// learned channel mask (the ICA). A 1x1 convolution makes the next hidden state, and x4 pixel-shuffle upsampling adds
/// to the bilinearly upsampled frame.
/// </para>
/// <para>
/// Training follows Sec. 4.4: 15-frame clips, SPyNet frozen for the first 5,000 iterations, then 300,000 iterations of
/// Charbonnier plus SSIM loss and another 130,000 iterations adding perceptual and GAN losses, weighted 1, 0.001, 1 and
/// 0.005, with RealBasicVSR's VGG19 perceptual loss, U-Net discriminator and vanilla GAN loss. The optimizer settings are
/// in the unreleased supplementary material, so the rates are RealBasicVSR's, whose losses the paper says it follows:
/// Adam (0.9, 0.99) at 1e-4, then 5e-5 for the generator and 1e-4 for the discriminator. The degradations (blur, noise,
/// JPEG and video compression) are data preparation and are not part of the model. The release's SPyNet is loaded
/// pretrained and starts from PyTorch's default initialization here; its GELU is the erf form and the library's the tanh
/// approximation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real videos are noisy and compressed, and attention that compares positions tends to copy those
/// artifacts from frame to frame. RealViformer instead compares feature channels, which the authors found far less
/// sensitive to artifacts, and learns to turn down channels that look unreliable.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 15, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new RealViformer&lt;float&gt;(arch, new RealViformerOptions());
/// model.Train(degradedClip, cleanClip);        // [15, 3, 64, 64] -> [15, 3, 256, 256], pixel values 0-255
/// var upscaled = model.Upscale(degradedClip);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RealViformer: Investigating Attention for Real-World Video Super-Resolution",
    "https://arxiv.org/abs/2407.13987",
    Year = 2024,
    Authors = "Yuehan Zhang, Angela Yao")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta2 = 0.99, Component = "reconstruction",
                Source = "Zhang and Yao 2024, Sec. 4.4: 300k iterations of Charbonnier + SSIM; the optimizer is in the "
                        + "unreleased supplementary, so the rate is RealBasicVSR's first stage (Adam 1e-4, betas (0.9, 0.99)).")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-5, Beta2 = 0.99, Component = "generator",
                Source = "Zhang and Yao 2024, Sec. 4.4: 130k further iterations with perceptual and GAN losses following "
                        + "RealBasicVSR, whose second stage trains the generator with Adam 5e-5, betas (0.9, 0.99).")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta2 = 0.99, Component = "discriminator",
                Source = "RealBasicVSR's discriminator, which the paper follows: Adam 1e-4, betas (0.9, 0.99).")]
public partial class RealViformer<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly RealViformerOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _adversarialOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;
    private RealViformerBackbone<T>? _backbone;
    private SpectralNormUNetDiscriminator<T>? _discriminator;
    private Vgg19PerceptualBlock<T>? _vgg;
    private ParameterAverage<T>? _average;
    // The generator's output from the step's generator objective, detached, which the discriminator step scores.
    private Tensor<T>? _lastFake;

    /// <summary>Creates a RealViformer model in ONNX inference mode.</summary>
    public RealViformer(NeuralNetworkArchitecture<T> architecture, string modelPath, RealViformerOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new RealViformerOptions();
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a RealViformer model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the paper's defaults when null.</param>
    /// <param name="optimizer">The first-stage optimizer; the paper's Adam when null. The adversarial stage always uses
    /// Adam at <see cref="RealViformerOptions.AdversarialLearningRate"/> and the discriminator Adam at
    /// <see cref="RealViformerOptions.DiscriminatorLearningRate"/>.</param>
    public RealViformer(NeuralNetworkArchitecture<T> architecture, RealViformerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>((options ?? new RealViformerOptions()).CharbonnierEpsilon))
    {
        _options = options ?? new RealViformerOptions();
        Options = _options;
        if (_options.NumFeatures < 4 || _options.NumFeatures % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumFeatures must be an even number of at least 4.");
        if (_options.NumBlocks is not { Length: 4 } || _options.NumBlocks.Any(b => b < 0))
            throw new ArgumentOutOfRangeException(nameof(options), "NumBlocks needs four non-negative counts (three levels and refinement).");
        if (_options.Heads is not { Length: 3 } || _options.Heads.Any(h => h < 1))
            throw new ArgumentOutOfRangeException(nameof(options), "Heads needs three positive counts.");
        if (_options.MergeHeads < 1 || _options.SqueezeFactor < 1 || _options.DiscriminatorChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MergeHeads, SqueezeFactor and DiscriminatorChannels must be positive.");
        _useNativeMode = true;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        InitializeLayers();
        if (_options.EmaMomentum is double momentum) _average = new ParameterAverage<T>(momentum);
        _optimizer = optimizer;
    }

    /// <inheritdoc/>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        state.Declare(nameof(RealViformer<T>) + ".EmaWeights", () => _average?.ToVector(), restored => _average?.Restore(restored));
        // The step count selects the training stage, so a clone continues in the stage the original was in.
        state.DeclareDoubleVector(nameof(RealViformer<T>) + ".TrainingSteps", () => new Vector<double>(new double[] { _trainingSteps }),
            restored => _trainingSteps = restored is { Length: > 0 } ? (int)restored[0] : 0);
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
        int dim = _options.NumFeatures;
        int squeeze = _options.SqueezeFactor;
        for (int level = 0; level < 3; level++)
        {
            int width = (level == 0 ? 1 : level == 1 ? 2 : 4) * dim;
            if (width % squeeze != 0 || width / squeeze % _options.Heads[level] != 0)
                throw new InvalidOperationException($"Level {level + 1}'s {width} channels must squeeze by {squeeze} into {_options.Heads[level]} heads.");
        }
        if (2 * dim % squeeze != 0 || 2 * dim / squeeze % _options.Heads[0] != 0 || dim % _options.MergeHeads != 0)
            throw new InvalidOperationException("The decoder's channels must squeeze into the first level's heads, and NumFeatures into MergeHeads.");
        _backbone = new RealViformerBackbone<T>(Layers, dim, _options.NumBlocks, _options.Heads, _options.FfnExpansionFactor,
            _options.MergeHeads, squeeze, _options.ChannelMask);
        var random = _options.Seed is int seed ? RandomHelper.CreateSeededRandom(seed) : RandomHelper.CreateSecureRandom();
        _discriminator = new SpectralNormUNetDiscriminator<T>(Layers, _options.DiscriminatorChannels, random);
        // A zero perceptual weight leaves the 20M-parameter VGG19 out entirely rather than carrying it unused.
        if (_options.PerceptualWeight > 0) _vgg = new Vgg19PerceptualBlock<T>(Layers);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        var output = Backbone.Run(Engine, Layers, new Tensor<T>(new[] { 1, 2, 3, 8, 8 }));
        var frames = Engine.Reshape(output, new[] { 2, 3, 32, 32 });
        Discriminator.Run(Engine, Layers, frames, training: false);
        _vgg?.Loss(Engine, Layers, frames, frames);
        MarkLayerShapesResolved();
    }

    private RealViformerBackbone<T> Backbone
        => _backbone ?? throw new InvalidOperationException("RealViformer's native layers are not initialized.");

    private SpectralNormUNetDiscriminator<T> Discriminator
        => _discriminator ?? throw new InvalidOperationException("RealViformer's native layers are not initialized.");

    private Tensor<T> RunClips(Tensor<T> frames)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        var output = Backbone.Run(Engine, Layers, clips);
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    // Frames [..., 3, H, W] as a batch of images [B, 3, H, W].
    private Tensor<T> AsImages(Tensor<T> frames)
    {
        int rank = frames.Rank;
        int h = frames.Shape[rank - 2], w = frames.Shape[rank - 1];
        return Engine.Reshape(frames, new[] { frames.Length / (3 * h * w), 3, h, w });
    }

    private IReadOnlyList<Tensor<T>> GeneratorParameters()
        => _discriminator is null
            ? TapeTrainingStep<T>.CollectParameters(Layers)
            : TapeTrainingStep<T>.CollectParameters(Layers.Take(_discriminator.First));

    /// <summary>The discriminator's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> DiscriminatorParameters
        => _discriminator is null ? Array.Empty<Tensor<T>>() : _discriminator.Parameters(Layers);

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters => _backbone is null ? Array.Empty<Tensor<T>>() : _backbone.FlowParameters(Layers);

    /// <summary>Test hook: the network on frames already scaled to [0, 1].</summary>
    internal Tensor<T> UpscaleNormalizedForTest(Tensor<T> frames) => RunClips(frames);

    /// <summary>Whether the next training step is in the adversarial stage.</summary>
    internal bool InAdversarialStage => _trainingSteps >= _options.ReconstructionIterations;

    #endregion

    #region Forward and training

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        return _average is { HasAverage: true }
            ? _average.WithAverage(GeneratorParameters(), () => RunClips(input))
            : RunClips(input);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunClips(PreprocessFrames(input));

    // Spectral normalization's power iteration runs on the host, which a compiled plan would bake at trace time.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private AdamOptimizer<T, Tensor<T>, Tensor<T>> ReleaseAdam(double learningRate)
        => new(this, new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = learningRate,
            Beta1 = 0.9,
            Beta2 = 0.99,
            Epsilon = 1e-8,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            UseAMSGrad = false,
        });

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildPaperOptimizer()
        => _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdam(_options.LearningRate), "reconstruction");

    // The second stage starts a fresh Adam at its own rate, as RealBasicVSR's second stage does.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildAdversarialOptimizer()
        => _adversarialOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdam(_options.AdversarialLearningRate), "generator");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildDiscriminatorOptimizer()
        => _discriminatorOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdam(_options.DiscriminatorLearningRate), "discriminator");

    // SPyNet is fixed for the first FlowFreezeIterations and tuned with the rest afterwards.
    private IReadOnlyList<Tensor<T>> TrainableGeneratorParameters()
    {
        var all = GeneratorParameters();
        if (_trainingSteps > _options.FlowFreezeIterations) return all;
        var flow = new HashSet<Tensor<T>>(FlowParameters);
        return all.Where(p => !flow.Contains(p)).ToList();
    }

    // mmagic GANLoss('vanilla'): BCE with logits against an all-real or all-fake label, averaged over pixels.
    private Tensor<T> VanillaGanLoss(Tensor<T> logits, bool targetIsReal)
    {
        var softplus = Engine.Softplus(targetIsReal ? Engine.TensorNegate(logits) : logits);
        return Engine.ReduceMean(softplus, Enumerable.Range(0, softplus.Rank).ToArray(), false);
    }

    private Tensor<T> Weighted(Tensor<T> loss, double weight) => Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(weight));

    // Charbonnier + weighted SSIM, the first stage's loss and the base of the second's.
    private Tensor<T> ReconstructionLoss(Tensor<T> output, Tensor<T> target)
    {
        var loss = Weighted(LossFunction.ComputeTapeLoss(output, target), _options.CharbonnierWeight);
        return _options.SsimWeight > 0
            ? Engine.TensorAdd(loss, Weighted(RealViformerBlocks.SsimLoss(Engine, AsImages(output), AsImages(target)), _options.SsimWeight))
            : loss;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Until <see cref="RealViformerOptions.ReconstructionIterations"/> the generator steps on Charbonnier + SSIM.
    /// Afterwards each step is RealBasicVSR's: the generator on Charbonnier, SSIM, perceptual and adversarial losses
    /// with the discriminator held fixed, then the discriminator on the real clip and the generator's detached output.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        bool adversarial = InAdversarialStage;
        _trainingSteps++;
        var target = NormalizeFrames(expectedOutput);
        _lastFake = null;
        if (!adversarial)
        {
            return new[]
            {
                new TrainingObjective<T>(new[] { input, target }, data => ReconstructionLoss(RunClips(PreprocessFrames(data[0])), data[1]))
                {
                    Optimizer = BuildPaperOptimizer(),
                    Parameters = TrainableGeneratorParameters,
                },
            };
        }

        return new[]
        {
            new TrainingObjective<T>(new[] { input, target }, data =>
            {
                var output = RunClips(PreprocessFrames(data[0]));
                _lastFake = Engine.StopGradient(output);
                var images = AsImages(output);
                var loss = ReconstructionLoss(output, data[1]);
                if (_vgg is not null)
                    loss = Engine.TensorAdd(loss, Weighted(_vgg.Loss(Engine, Layers, images, AsImages(data[1])), _options.PerceptualWeight));
                return Engine.TensorAdd(loss, Weighted(VanillaGanLoss(Discriminator.Run(Engine, Layers, images, training: true), true), _options.GANWeight));
            })
            {
                Optimizer = BuildAdversarialOptimizer(),
                Parameters = TrainableGeneratorParameters,
            },
            new TrainingObjective<T>(
                () => new[] { target, _lastFake ?? throw new InvalidOperationException("The generator step did not run.") },
                data =>
                {
                    var real = VanillaGanLoss(Discriminator.Run(Engine, Layers, AsImages(data[0]), training: true), true);
                    var fake = VanillaGanLoss(Discriminator.Run(Engine, Layers, AsImages(data[1]), training: true), false);
                    return Engine.TensorAdd(real, fake);
                })
            {
                Optimizer = BuildDiscriminatorOptimizer(),
                Parameters = () => DiscriminatorParameters,
            },
        };
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The current stage's generator loss on frames scaled to [0, 1] (Charbonnier + SSIM; perceptual and
    /// adversarial terms join in the second stage).</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var output = RunClips(PreprocessFrames(input));
        var normalized = NormalizeFrames(target);
        var loss = ReconstructionLoss(output, normalized);
        if (InAdversarialStage)
        {
            var images = AsImages(output);
            if (_vgg is not null)
                loss = Engine.TensorAdd(loss, Weighted(_vgg.Loss(Engine, Layers, images, AsImages(normalized)), _options.PerceptualWeight));
            loss = Engine.TensorAdd(loss, Weighted(VanillaGanLoss(Discriminator.Run(Engine, Layers, images, training: false), true), _options.GANWeight));
        }
        return loss[0];
    }

    /// <inheritdoc />
    protected override void OnTrainingStepEnded(bool succeeded)
    {
        base.OnTrainingStepEnded(succeeded);
        _lastFake = null;
        if (succeeded && !_usingCustomLayers) _average?.Update(GeneratorParameters());
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

    /// <summary>Frames are pixels on the 0-255 scale; the network divides them by 255.</summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.Value;

    /// <summary>Upscaled frames come back on the same 0-255 scale.</summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "RealViformer-Native" : "RealViformer-ONNX",
            Description = "RealViformer channel-attention recurrent real-world VSR (Zhang and Yao, ECCV 2024)",
            Complexity = _options.NumBlocks.Sum()
        };
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumBlocks"] = string.Join(",", _options.NumBlocks);
        m.AdditionalInfo["Heads"] = string.Join(",", _options.Heads);
        m.AdditionalInfo["SqueezeFactor"] = _options.SqueezeFactor.ToString();
        m.AdditionalInfo["ScaleFactor"] = "4";
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RealViformer<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
