using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// DualX-VSR: a dual axial spatial x temporal transformer for real-world video super-resolution without motion
/// compensation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is Cao et al.'s DualX-VSR (arXiv 2506.04830, 2025), a ViViT-based model with no optical flow, alignment or
/// recurrence. Each frame passes a 3x3 convolution to 64 channels, a two-block temporal transformer mixes every pixel's
/// frames, and 2x2 patches become 1,280-wide tokens. Six vertical-temporal blocks attend over all height x time positions
/// of each token column, then six horizontal-temporal blocks over all width x time positions of each row, with rotary
/// positions (RoPE, base 10,000). The tokens are decoded back to pixels, a second two-block temporal transformer runs, and
/// per-frame convolution and pixel shuffle upsample by four.
/// </para>
/// <para>
/// The paper releases no code, so what it leaves open follows the ViViT implementation it builds on (vit-pytorch):
/// pre-norm residual blocks with LayerNorm, bias-free q/k/v projections to 12 heads of 64 channels, GELU MLPs, a final
/// norm per transformer and the LayerNorm-Linear-LayerNorm patch embedding. With these choices the default network has
/// 127.8M parameters against the paper's 127.95M. RoPE over a height (or width) x time sequence gives each axis half of a
/// head's channels; the upsampler is BasicVSR's (pixel shuffle x2 stages with LeakyReLU, conv_hr, conv_last).
/// </para>
/// <para>
/// Training follows Sec. 5.1 and App. C.1: AdamW at 5e-5 on Charbonnier loss for 500,000 pretraining and 150,000
/// fine-tuning iterations (bicubic, then RealBasicVSR's first-order degradations: data preparation, not the model), then
/// 25,000 iterations with Charbonnier, perceptual and GAN losses weighted 0.01, 1 and 0.005 following BasicVSR, at 1e-5 for
/// the generator and RealBasicVSR's spectral-norm U-Net discriminator, with its VGG19 perceptual loss and vanilla GAN
/// loss. AdamW's betas and decay are unstated, so they are PyTorch's defaults.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most video upscalers first work out how things moved between frames. DualX-VSR skips that: its
/// attention looks along a column across all frames, then along a row across all frames, so every position can draw on
/// matching content anywhere in the clip, like solving a crossword by reading down and across.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 16, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new DualXVSR&lt;float&gt;(arch, new DualXVSROptions());
/// model.Train(degradedClip, cleanClip);        // [16, 3, 64, 64] -> [16, 3, 256, 256], pixel values 0-255
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
[ResearchPaper("DualX-VSR: Dual Axial Spatial x Temporal Transformer for Real-World Video Super-Resolution without Motion Compensation",
    "https://arxiv.org/abs/2506.04830",
    Year = 2025,
    Authors = "Shuo Cao, Yihao Liu, Xiaohui Li, Yuanting Gao, Yu Zhou, Chao Dong")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, WeightDecay = 0.01, Component = "reconstruction",
                Source = "Cao et al. 2025, Sec. 5.1: AdamW at 5e-5 for 500k pretraining and 150k fine-tuning iterations on "
                        + "Charbonnier loss; betas and decay are unstated, so PyTorch's AdamW defaults.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, WeightDecay = 0.01, Component = "generator",
                Source = "Cao et al. 2025, Sec. 5.1: 25k GAN iterations at 1e-5 for the generator.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, WeightDecay = 0.01, Component = "discriminator",
                Source = "Cao et al. 2025, Sec. 5.1: 25k GAN iterations at 1e-5 for the discriminator.")]
public partial class DualXVSR<T> : VideoSuperResolutionBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly DualXVSROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _adversarialOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;
    private DualXBackbone<T>? _backbone;
    private SpectralNormUNetDiscriminator<T>? _discriminator;
    private Vgg19PerceptualBlock<T>? _vgg;
    // The generator's output from the step's generator objective, detached, which the discriminator step scores.
    private Tensor<T>? _lastFake;

    /// <summary>Creates a DualX-VSR model in ONNX inference mode.</summary>
    public DualXVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, DualXVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options is null ? new DualXVSROptions() : new DualXVSROptions(options);
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a DualX-VSR model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the paper's defaults when null.</param>
    /// <param name="optimizer">The Charbonnier stages' optimizer; the paper's AdamW when null. The GAN stage always uses
    /// AdamW at <see cref="DualXVSROptions.AdversarialLearningRate"/> and the discriminator AdamW at
    /// <see cref="DualXVSROptions.DiscriminatorLearningRate"/>.</param>
    public DualXVSR(NeuralNetworkArchitecture<T> architecture, DualXVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>((options ?? new DualXVSROptions()).CharbonnierEpsilon))
    {
        _options = options is null ? new DualXVSROptions() : new DualXVSROptions(options);
        Options = _options;
        if (_options.NumFeatures < 1 || _options.PatchSize < 1 || _options.EmbedDim < 1 || _options.NumHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "NumFeatures, PatchSize, EmbedDim and NumHeads must be positive.");
        if (_options.HeadDim < 4 || _options.HeadDim % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "HeadDim must be a positive multiple of 4 (RoPE rotates channel pairs on two axes).");
        if (_options.ScaleFactor < 2 || (_options.ScaleFactor & (_options.ScaleFactor - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ScaleFactor must be a power of two of at least 2.");
        if (_options.EmbeddingTemporalDepth < 0 || _options.VerticalDepth < 0 || _options.HorizontalDepth < 0
            || _options.ReconstructionTemporalDepth < 0 || _options.DiscriminatorChannels < 1 || _options.MlpRatio <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Depths must be non-negative and the MLP ratio and discriminator width positive.");
        _useNativeMode = true;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        InitializeLayers();
        _optimizer = optimizer;
    }

    /// <inheritdoc/>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        // The step count selects the training stage, so a clone continues in the stage the original was in.
        state.DeclareDoubleVector(nameof(DualXVSR<T>) + ".TrainingSteps", () => new Vector<double>(new double[] { _trainingSteps }),
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
        var o = _options;
        _backbone = new DualXBackbone<T>(Layers, o.NumFeatures, o.PatchSize, o.EmbedDim, o.NumHeads, o.HeadDim, o.MlpRatio,
            o.EmbeddingTemporalDepth, o.VerticalDepth, o.HorizontalDepth, o.ReconstructionTemporalDepth, o.ScaleFactor, o.RopeBase);
        var random = o.Seed is int seed ? RandomHelper.CreateSeededRandom(seed) : RandomHelper.CreateSecureRandom();
        _discriminator = new SpectralNormUNetDiscriminator<T>(Layers, o.DiscriminatorChannels, random);
        // A zero perceptual weight leaves the 20M-parameter VGG19 out entirely rather than carrying it unused.
        if (o.PerceptualWeight > 0) _vgg = new Vgg19PerceptualBlock<T>(Layers);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (LayerShapesResolved || _usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        int side = 4 * _options.PatchSize;
        var output = Backbone.Forward(Engine, Layers, new Tensor<T>(new[] { 1, 2, 3, side, side }));
        var frames = Engine.Reshape(output, output._shape.Skip(1).ToArray());
        Discriminator.Run(Engine, Layers, frames, training: false);
        _vgg?.Loss(Engine, Layers, frames, frames);
        MarkLayerShapesResolved();
    }

    private DualXBackbone<T> Backbone
        => _backbone ?? throw new InvalidOperationException("DualX-VSR's native layers are not initialized.");

    private SpectralNormUNetDiscriminator<T> Discriminator
        => _discriminator ?? throw new InvalidOperationException("DualX-VSR's native layers are not initialized.");

    private Tensor<T> RunClips(Tensor<T> frames)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        var output = Backbone.Forward(Engine, Layers, clips);
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

    /// <summary>Test hook: the network on frames already scaled to [0, 1].</summary>
    internal Tensor<T> UpscaleNormalizedForTest(Tensor<T> frames) => RunClips(frames);

    /// <summary>Whether the next training step is in the GAN stage.</summary>
    internal bool InAdversarialStage => _trainingSteps >= _options.ReconstructionIterations;

    #endregion

    #region Forward and training

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input) => _usingCustomLayers ? base.Forward(input) : RunClips(input);

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunClips(PreprocessFrames(input));

    // Spectral normalization's power iteration runs on the host, which a compiled plan would bake at trace time.
    /// <inheritdoc />
    protected override bool SupportsFusedCompiledTraining => false;

    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> PaperAdamW(double learningRate)
        => new(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = learningRate,
            WeightDecay = _options.WeightDecay,
            UseAdaptiveLearningRate = false,
        });

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildPaperOptimizer()
        => _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(_options.LearningRate), "reconstruction");

    // The GAN stage starts a fresh AdamW at its own rate.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildAdversarialOptimizer()
        => _adversarialOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(_options.AdversarialLearningRate), "generator");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildDiscriminatorOptimizer()
        => _discriminatorOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, PaperAdamW(_options.DiscriminatorLearningRate), "discriminator");

    // mmagic GANLoss('vanilla'): BCE with logits against an all-real or all-fake label, averaged over pixels.
    private Tensor<T> VanillaGanLoss(Tensor<T> logits, bool targetIsReal)
    {
        var softplus = Engine.Softplus(targetIsReal ? Engine.TensorNegate(logits) : logits);
        return Engine.ReduceMean(softplus, Enumerable.Range(0, softplus.Rank).ToArray(), false);
    }

    private Tensor<T> Weighted(Tensor<T> loss, double weight) => Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(weight));

    // The GAN stage's generator loss: weighted Charbonnier, perceptual and adversarial terms.
    private Tensor<T> AdversarialLoss(Tensor<T> output, Tensor<T> target, bool training)
    {
        var images = AsImages(output);
        var loss = Weighted(LossFunction.ComputeTapeLoss(output, target), _options.AdversarialPixelWeight);
        if (_vgg is not null)
            loss = Engine.TensorAdd(loss, Weighted(_vgg.Loss(Engine, Layers, images, AsImages(target)), _options.PerceptualWeight));
        return Engine.TensorAdd(loss, Weighted(VanillaGanLoss(Discriminator.Run(Engine, Layers, images, training), true), _options.GANWeight));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Until <see cref="DualXVSROptions.ReconstructionIterations"/> the generator steps on Charbonnier loss. Afterwards
    /// each step trains the generator on Charbonnier, perceptual and adversarial losses with the discriminator held
    /// fixed, then the discriminator on the real clip and the generator's detached output.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        // Train on one NumFrames-frame clip of a longer sequence, as the paper's data pipeline does.
        (input, expectedOutput) = TrainingClip(input, expectedOutput);
        bool adversarial = InAdversarialStage;
        _trainingSteps++;
        var target = NormalizeFrames(expectedOutput);
        _lastFake = null;
        if (!adversarial)
        {
            return new[]
            {
                new TrainingObjective<T>(new[] { input, target }, data => LossFunction.ComputeTapeLoss(RunClips(PreprocessFrames(data[0])), data[1]))
                {
                    Optimizer = BuildPaperOptimizer(),
                    Parameters = GeneratorParameters,
                },
            };
        }

        return new[]
        {
            new TrainingObjective<T>(new[] { input, target }, data =>
            {
                var output = RunClips(PreprocessFrames(data[0]));
                _lastFake = Engine.StopGradient(output);
                return AdversarialLoss(output, data[1], training: true);
            })
            {
                Optimizer = BuildAdversarialOptimizer(),
                Parameters = GeneratorParameters,
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
    /// <remarks>The current stage's generator loss on frames scaled to [0, 1] (Charbonnier; perceptual and adversarial
    /// terms join in the GAN stage).</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        using var _ = new NoGradScope<T>();
        if (_usingCustomLayers || !_useNativeMode) return LossFunction.ComputeTapeLoss(Predict(input), target)[0];
        var output = RunClips(PreprocessFrames(input));
        var normalized = NormalizeFrames(target);
        return (InAdversarialStage ? AdversarialLoss(output, normalized, training: false) : LossFunction.ComputeTapeLoss(output, normalized))[0];
    }

    /// <inheritdoc />
    protected override void OnTrainingStepEnded(bool succeeded)
    {
        base.OnTrainingStepEnded(succeeded);
        _lastFake = null;
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
            Name = _useNativeMode ? "DualXVSR-Native" : "DualXVSR-ONNX",
            Description = "DualX-VSR dual axial spatial x temporal transformer VSR without motion compensation (Cao et al. 2025)",
            Complexity = _options.VerticalDepth + _options.HorizontalDepth
        };
        m.AdditionalInfo["EmbedDim"] = _options.EmbedDim.ToString();
        m.AdditionalInfo["Depths"] = $"{_options.EmbeddingTemporalDepth},{_options.VerticalDepth},{_options.HorizontalDepth},{_options.ReconstructionTemporalDepth}";
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["PatchSize"] = _options.PatchSize.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DualXVSR<T>));
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
