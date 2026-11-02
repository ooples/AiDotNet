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
/// Real-ESRGAN Video: Real-ESRGAN real-world super-resolution applied to every frame of a video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Wang et al. 2021 disclose no video network: the authors' release upscales video with
/// inference_realesrgan_video.py, which runs the image model on each frame independently. This class is that
/// pipeline with the paper's generator, RRDBNet (64 features, 23 residual-in-residual dense blocks, growth 32, x4),
/// trained as RealESRGAN_x4plus (options/train_realesrgan_x4plus.yml): an L1 loss and a VGG19 perceptual loss
/// (conv1_2 .. conv5_4 before ReLU, weights 0.1, 0.1, 1, 1, 1) against the unsharp-masked target and a vanilla GAN loss
/// weighted 0.1 from a spectrally normalized U-Net discriminator; generator and discriminator alternate with Adam
/// (0.9, 0.99) at 1e-4, and inference uses a 0.999 moving average of the generator.
/// </para>
/// <para>
/// The release synthesizes training pairs with its second-order degradation pipeline and starts this stage from the
/// L1-trained Real-ESRNet and an ImageNet-pretrained VGG19; here the caller supplies degraded/clean pairs and the
/// networks start from the release's initializations.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real-ESRGAN is one of the most widely-used practical upscaling
/// tools. It learns to handle all the messy artifacts (compression, noise, blur) that real
/// videos have, not just the simple downscaling used in lab benchmarks, and upscales each
/// frame of a video by 4x.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 4, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new RealESRGANVideo&lt;float&gt;(arch, new RealESRGANVideoOptions());
/// model.Train(degradedFrames, cleanFrames);     // [4, 3, 64, 64] -> [4, 3, 256, 256], pixel values 0-255
/// var upscaled = model.Upscale(degradedFrames);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data",
    "https://arxiv.org/abs/2107.10833",
    Year = 2021,
    Authors = "Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta2 = 0.99, ReferenceBatchSize = 48, Component = "generator",
                Source = "Wang et al. 2021, Training details; options/train_realesrgan_x4plus.yml: generator Adam lr 1e-4, "
                        + "betas (0.9, 0.99), batch 48, 400k iterations, no schedule.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta2 = 0.99, ReferenceBatchSize = 48, Component = "discriminator",
                Source = "options/train_realesrgan_x4plus.yml: discriminator Adam lr 1e-4, betas (0.9, 0.99).")]
public partial class RealESRGANVideo<T> : VideoSuperResolutionBase<T>
{
    private readonly RealESRGANVideoOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private RrdbNetBlock<T>? _generator;
    private SpectralNormUNetDiscriminator<T>? _discriminator;
    private Vgg19PerceptualBlock<T>? _vgg;
    private ParameterAverage<T>? _average;
    // The generator's output from the step's generator objective, detached, which the discriminator step scores.
    private Tensor<T>? _lastFake;

    /// <summary>Creates a Real-ESRGAN Video model in ONNX inference mode.</summary>
    public RealESRGANVideo(NeuralNetworkArchitecture<T> architecture, string modelPath, RealESRGANVideoOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new RealESRGANVideoOptions();
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = 4;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a Real-ESRGAN Video model in native training mode.</summary>
    /// <param name="architecture">The input description.</param>
    /// <param name="options">The configuration; the release's defaults when null.</param>
    /// <param name="optimizer">The generator's optimizer; the release's Adam when null. The discriminator always uses
    /// the release's Adam at <see cref="RealESRGANVideoOptions.DiscriminatorLearningRate"/>.</param>
    public RealESRGANVideo(NeuralNetworkArchitecture<T> architecture, RealESRGANVideoOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new RealESRGANVideoOptions();
        Options = _options;
        if (_options.NumFeatures < 1 || _options.GrowthChannels < 1 || _options.DiscriminatorChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Channel counts must be positive.");
        if (_options.NumRRDBBlocks < 0) throw new ArgumentOutOfRangeException(nameof(options), "NumRRDBBlocks cannot be negative.");
        _useNativeMode = true;
        ScaleFactor = 4;
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
        _generator = new RrdbNetBlock<T>(Layers, _options.NumFeatures, _options.NumRRDBBlocks, _options.GrowthChannels);
        var random = _options.Seed is int seed ? RandomHelper.CreateSeededRandom(seed) : RandomHelper.CreateSecureRandom();
        _discriminator = new SpectralNormUNetDiscriminator<T>(Layers, _options.DiscriminatorChannels, random);
        // A zero perceptual weight leaves the 20M-parameter VGG19 out entirely rather than carrying it unused.
        if (_options.PerceptualWeight > 0) _vgg = new Vgg19PerceptualBlock<T>(Layers);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (_usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        var output = Generator.Forward(Engine, Layers, new Tensor<T>(new[] { 1, 3, 8, 8 }));
        Discriminator.Run(Engine, Layers, output, training: false);
        _vgg?.Loss(Engine, Layers, output, output);
    }

    private RrdbNetBlock<T> Generator => _generator ?? throw new InvalidOperationException("Real-ESRGAN Video's native layers are not initialized.");

    private SpectralNormUNetDiscriminator<T> Discriminator
        => _discriminator ?? throw new InvalidOperationException("Real-ESRGAN Video's native layers are not initialized.");

    // inference_realesrgan_video.py: every frame of [..., 3, H, W] is upscaled independently.
    private Tensor<T> RunFrames(Tensor<T> frames)
    {
        int rank = frames.Rank, h = frames.Shape[rank - 2], w = frames.Shape[rank - 1];
        var images = Engine.Reshape(frames, new[] { frames.Length / (3 * h * w), 3, h, w });
        var output = Generator.Forward(Engine, Layers, images);
        var shape = frames.Shape.ToArray();
        shape[rank - 2] = 4 * h;
        shape[rank - 1] = 4 * w;
        return Engine.Reshape(output, shape);
    }

    // Frames [..., 3, H, W] as a batch of images [B, 3, H, W].
    private Tensor<T> AsImages(Tensor<T> frames)
    {
        int rank = frames.Rank;
        int h = frames.Shape[rank - 2], w = frames.Shape[rank - 1];
        return Engine.Reshape(frames, new[] { frames.Length / (3 * h * w), 3, h, w });
    }

    private IReadOnlyList<Tensor<T>> GeneratorParameters()
        => _generator is null ? TapeTrainingStep<T>.CollectParameters(Layers) : _generator.Parameters(Layers);

    /// <summary>The discriminator's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> DiscriminatorParameters
        => _discriminator is null ? Array.Empty<Tensor<T>>() : _discriminator.Parameters(Layers);

    /// <summary>The perceptual VGG19's parameters (never trained).</summary>
    internal IReadOnlyList<Tensor<T>> PerceptualParameters => _vgg is null ? Array.Empty<Tensor<T>>() : _vgg.Parameters(Layers);

    /// <summary>Test hook: the generator's parameters.</summary>
    internal IReadOnlyList<Tensor<T>> GeneratorParametersForTest => GeneratorParameters();

    /// <summary>Test hook: the generator on frames already scaled to [0, 1], with the trained weights.</summary>
    internal Tensor<T> UpscaleNormalizedForTest(Tensor<T> frames) => RunFrames(frames);

    #endregion

    #region Forward and training

    /// <inheritdoc />
    /// <remarks>The release evaluates its moving average of the generator.</remarks>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_usingCustomLayers) return base.Forward(input);
        return _average is { HasAverage: true }
            ? _average.WithAverage(GeneratorParameters(), () => RunFrames(input))
            : RunFrames(input);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunFrames(PreprocessFrames(input));

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
        => _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdam(_options.LearningRate), "generator");

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildDiscriminatorOptimizer()
        => _discriminatorOptimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, ReleaseAdam(_options.DiscriminatorLearningRate), "discriminator");

    private Tensor<T> Sharpened(Tensor<T> normalizedTarget)
        => _options.SharpenTarget
            ? RealBasicVsrCore<T>.UnsharpMask(normalizedTarget, _options.UnsharpWeight, _options.UnsharpKernelSize, _options.UnsharpThreshold)
            : normalizedTarget;

    // BasicSR GANLoss('vanilla'): BCE with logits against an all-real or all-fake label, averaged over pixels.
    private Tensor<T> VanillaGanLoss(Tensor<T> logits, bool targetIsReal)
    {
        var softplus = Engine.Softplus(targetIsReal ? Engine.TensorNegate(logits) : logits);
        return Engine.ReduceMean(softplus, Enumerable.Range(0, softplus.Rank).ToArray(), false);
    }

    private Tensor<T> Weighted(Tensor<T> loss, double weight) => Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(weight));

    /// <inheritdoc />
    /// <remarks>
    /// RealESRGANModel.optimize_parameters: the generator steps on the L1, perceptual and adversarial losses with the
    /// discriminator held fixed, then the discriminator steps on the real frames and the generator's detached output.
    /// </remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        var generatorOptimizer = BuildPaperOptimizer();
        var discriminatorOptimizer = BuildDiscriminatorOptimizer();
        var target = NormalizeFrames(expectedOutput);
        var sharp = Sharpened(target);
        _lastFake = null;
        return new[]
        {
            new TrainingObjective<T>(new[] { input, sharp }, data =>
            {
                var output = RunFrames(PreprocessFrames(data[0]));
                _lastFake = Engine.StopGradient(output);
                var images = AsImages(output);
                var loss = Weighted(LossFunction.ComputeTapeLoss(output, data[1]), _options.PixelLossWeight);
                if (_vgg is not null)
                    loss = Engine.TensorAdd(loss, Weighted(_vgg.Loss(Engine, Layers, images, AsImages(data[1])), _options.PerceptualWeight));
                return Engine.TensorAdd(loss, Weighted(VanillaGanLoss(Discriminator.Run(Engine, Layers, images, training: true), true), _options.GANWeight));
            })
            {
                Optimizer = generatorOptimizer,
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

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "RealESRGANVideo-Native" : "RealESRGANVideo-ONNX",
            Description = "Real-ESRGAN applied frame by frame (Wang et al., 2021)",
            Complexity = _options.NumRRDBBlocks
        };
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumRRDBBlocks"] = _options.NumRRDBBlocks.ToString();
        m.AdditionalInfo["GrowthChannels"] = _options.GrowthChannels.ToString();
        m.AdditionalInfo["PerceptualWeight"] = _options.PerceptualWeight.ToString();
        m.AdditionalInfo["GANWeight"] = _options.GANWeight.ToString();
        m.AdditionalInfo["ScaleFactor"] = "4";
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RealESRGANVideo<T>));
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
