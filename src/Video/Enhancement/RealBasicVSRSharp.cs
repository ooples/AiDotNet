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
/// RealBasicVSR-Sharp: the adversarial training stage of RealBasicVSR, trading PSNR for perceptual quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The generator is the authors' RealBasicVSRNet, the same network <see cref="RealBasicVSR{T}"/> trains in the first
/// stage: an image-cleaning module applied up to three times, then BasicVSR. This class trains the release's second
/// stage (configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py): the generator minimizes the L1
/// output loss, a VGG19 perceptual loss (conv1_2 .. conv5_4 before ReLU, weights 0.1, 0.1, 1, 1, 1), the L1 cleaning
/// loss and a vanilla GAN loss weighted 0.05 from a spectrally normalized U-Net discriminator, with Adam (0.9, 0.99) at
/// 5e-5; the discriminator then steps on the real clip and the same, detached output with Adam at 1e-4. The output and
/// perceptual losses score the unsharp-masked target, the discriminator the unsharpened one. SPyNet stays frozen and
/// inference uses a 0.999 moving average of the generator.
/// </para>
/// <para>
/// The release starts this stage from the first stage's weights and an ImageNet-pretrained VGG19 and SPyNet; here they
/// start from PyTorch's default initialization (load trained weights through the parameter API), and the caller
/// supplies degraded/clean pairs where the release synthesizes the degradations.
/// </para>
/// <para>
/// <b>For Beginners:</b> There are two ways to measure video quality: mathematical accuracy
/// (PSNR) and visual quality (how it looks to your eyes). The base RealBasicVSR maximizes
/// PSNR, while this "Sharp" variant maximizes visual quality. The Sharp version produces
/// images with crisper textures and more natural-looking details.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 15, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new RealBasicVSRSharp&lt;float&gt;(arch, new RealBasicVSRSharpOptions());
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
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-5, Beta2 = 0.99, Component = "generator",
                Source = "configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py: generator Adam lr 5e-5, "
                        + "betas (0.9, 0.99), no learning-rate schedule, 150k iterations.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta2 = 0.99, Component = "discriminator",
                Source = "configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py: discriminator Adam lr "
                        + "1e-4, betas (0.9, 0.99).")]
public partial class RealBasicVSRSharp<T> : VideoSuperResolutionBase<T>
{
    // RealBasicVSRNet's training threshold: dynamic_refine_thres = 255 levels.
    private const double TrainingCleaningThreshold = 1.0;

    private readonly RealBasicVSRSharpOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private RealBasicVsrCore<T>? _core;
    private SpectralNormUNetDiscriminator<T>? _discriminator;
    private Vgg19PerceptualBlock<T>? _vgg;
    private ParameterAverage<T>? _average;
    // The generator's output from the step's generator objective, detached, which the discriminator step scores.
    private Tensor<T>? _lastFake;

    /// <summary>Creates a RealBasicVSR-Sharp model in ONNX inference mode.</summary>
    public RealBasicVSRSharp(NeuralNetworkArchitecture<T> architecture, string modelPath, RealBasicVSRSharpOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new RealBasicVSRSharpOptions();
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        // Predict evaluates with the averaged weights, which the parameter vector does not carry.
        state.Declare(nameof(RealBasicVSRSharp<T>) + ".EmaWeights", () => _average?.ToVector(), restored => _average?.Restore(restored));
    }

    /// <summary>Creates a RealBasicVSR-Sharp model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's defaults when null.</param>
    /// <param name="optimizer">The generator's optimizer; the release's Adam when null. The discriminator always uses
    /// the release's Adam at <see cref="RealBasicVSRSharpOptions.DiscriminatorLearningRate"/>.</param>
    public RealBasicVSRSharp(NeuralNetworkArchitecture<T> architecture, RealBasicVSRSharpOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new RealBasicVSRSharpOptions();
        Options = _options;
        if (_options.MidChannels < 1 || _options.DiscriminatorChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MidChannels and DiscriminatorChannels must be positive.");
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
        var (output, _) = Core.Run(Engine, Layers, new Tensor<T>(new[] { 1, 2, 3, 8, 8 }), TrainingCleaningThreshold);
        var frames = Engine.Reshape(output, new[] { 2, 3, 32, 32 });
        Discriminator.Run(Engine, Layers, frames, training: false);
        _vgg?.Loss(Engine, Layers, frames, frames);
        MarkLayerShapesResolved();
    }

    private RealBasicVsrCore<T> Core => _core ?? throw new InvalidOperationException("RealBasicVSR-Sharp's native layers are not initialized.");

    private SpectralNormUNetDiscriminator<T> Discriminator
        => _discriminator ?? throw new InvalidOperationException("RealBasicVSR-Sharp's native layers are not initialized.");

    private Vgg19PerceptualBlock<T> Vgg => _vgg ?? throw new InvalidOperationException("RealBasicVSR-Sharp's native layers are not initialized.");

    private (Tensor<T> Output, Tensor<T> Cleaned) RunClips(Tensor<T> frames, double threshold)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        var (output, cleaned) = Core.Run(Engine, Layers, clips, threshold);
        return batched
            ? (output, cleaned)
            : (Engine.Reshape(output, output._shape.Skip(1).ToArray()), Engine.Reshape(cleaned, cleaned._shape.Skip(1).ToArray()));
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

    /// <summary>The perceptual VGG19's parameters (never trained).</summary>
    internal IReadOnlyList<Tensor<T>> PerceptualParameters => _vgg is null ? Array.Empty<Tensor<T>>() : _vgg.Parameters(Layers);

    /// <summary>The SPyNet parameters (never trained).</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters => _core is null ? Array.Empty<Tensor<T>>() : _core.FlowParameters(Layers);

    /// <summary>Test hook: the discriminator's per-pixel logits for images [N, 3, H, W] in [0, 1].</summary>
    internal Tensor<T> DiscriminateForTest(Tensor<T> images) => Discriminator.Run(Engine, Layers, images, training: false);

    /// <summary>Test hook: the weighted perceptual loss between images [N, 3, H, W] in [0, 1].</summary>
    internal Tensor<T> PerceptualLossForTest(Tensor<T> output, Tensor<T> target) => Vgg.Loss(Engine, Layers, output, target);

    #endregion

    #region Forward and training

    /// <inheritdoc />
    /// <remarks>The release evaluates its moving average of the generator; cleaning stops below the inference threshold.</remarks>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        double threshold = _options.InferenceCleaningThreshold / 255.0;
        return _average is { HasAverage: true }
            ? _average.WithAverage(GeneratorParameters(), () => RunClips(input, threshold).Output)
            : RunClips(input, threshold).Output;
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunClips(PreprocessFrames(input), TrainingCleaningThreshold).Output;

    // The cleaning passes stop on a host read of the residue, and spectral normalization's power iteration runs on the
    // host, both of which a compiled plan would bake at trace time.
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
        => _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdam(_options.LearningRate), "generator");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildDiscriminatorOptimizer()
        => _discriminatorOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdam(_options.DiscriminatorLearningRate), "discriminator");

    // RealBasicVSRNet: self.basicvsr.spynet.requires_grad_(False).
    private IReadOnlyList<Tensor<T>> TrainableGeneratorParameters()
    {
        var flow = new HashSet<Tensor<T>>(FlowParameters);
        return GeneratorParameters().Where(p => !flow.Contains(p)).ToList();
    }

    private Tensor<T> Sharpened(Tensor<T> normalizedTarget)
        => _options.SharpenTarget
            ? RealBasicVsrCore<T>.UnsharpMask(normalizedTarget, _options.UnsharpWeight, _options.UnsharpKernelSize, _options.UnsharpThreshold)
            : normalizedTarget;

    // mmagic GANLoss('vanilla'): BCE with logits against an all-real or all-fake label, averaged over pixels.
    private Tensor<T> VanillaGanLoss(Tensor<T> logits, bool targetIsReal)
    {
        var softplus = Engine.Softplus(targetIsReal ? Engine.TensorNegate(logits) : logits);
        return Engine.ReduceMean(softplus, Enumerable.Range(0, softplus.Rank).ToArray(), false);
    }

    private Tensor<T> Weighted(Tensor<T> loss, double weight) => Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(weight));

    /// <inheritdoc />
    /// <remarks>
    /// RealBasicVSR.train_step: the generator steps on the output, perceptual, cleaning and adversarial losses with the
    /// discriminator held fixed, then the discriminator steps on the real clip and the generator's detached output.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        var generatorOptimizer = BuildPaperOptimizer();
        var discriminatorOptimizer = BuildDiscriminatorOptimizer();
        var target = NormalizeFrames(expectedOutput);
        var sharp = Sharpened(target);
        var cleanTarget = RealBasicVsrCore<T>.AreaDownsample4(Engine, sharp);
        _lastFake = null;
        return new[]
        {
            new TrainingObjective<T>(new[] { input, sharp, cleanTarget }, data =>
            {
                var (output, cleaned) = RunClips(PreprocessFrames(data[0]), TrainingCleaningThreshold);
                _lastFake = Engine.StopGradient(output);
                var images = AsImages(output);
                var loss = Weighted(LossFunction.ComputeTapeLoss(output, data[1]), _options.PixelLossWeight);
                if (_vgg is not null)
                    loss = Engine.TensorAdd(loss, Weighted(_vgg.Loss(Engine, Layers, images, AsImages(data[1])), _options.PerceptualWeight));
                loss = Engine.TensorAdd(loss, Weighted(VanillaGanLoss(Discriminator.Run(Engine, Layers, images, training: true), true), _options.GANWeight));
                return Engine.TensorAdd(loss, Weighted(LossFunction.ComputeTapeLoss(cleaned, data[2]), _options.CleaningLossWeight));
            })
            {
                Optimizer = generatorOptimizer,
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
                Optimizer = discriminatorOptimizer,
                Parameters = () => DiscriminatorParameters,
            },
        };
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
            Name = _useNativeMode ? "RealBasicVSRSharp-Native" : "RealBasicVSRSharp-ONNX",
            Description = "RealBasicVSR-Sharp adversarial real-world VSR (Chan et al., CVPR 2022)",
            Complexity = _options.NumPropagationBlocks
        };
        m.AdditionalInfo["MidChannels"] = _options.MidChannels.ToString();
        m.AdditionalInfo["NumPropagationBlocks"] = _options.NumPropagationBlocks.ToString();
        m.AdditionalInfo["NumCleaningBlocks"] = _options.NumCleaningBlocks.ToString();
        m.AdditionalInfo["DiscriminatorChannels"] = _options.DiscriminatorChannels.ToString();
        m.AdditionalInfo["PerceptualWeight"] = _options.PerceptualWeight.ToString();
        m.AdditionalInfo["GANWeight"] = _options.GANWeight.ToString();
        m.AdditionalInfo["ScaleFactor"] = "4";
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RealBasicVSRSharp<T>));
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
