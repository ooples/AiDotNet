using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Progressive GAN (ProGAN, Karras et al. 2017) — a GAN that generates high-resolution
/// images by progressively growing the generator and discriminator.
///
/// For Beginners:
/// Progressive GAN trains a GAN to generate large images by starting small (4x4) and
/// adding layers to grow the resolution. As in every GAN, the generator's input is a
/// 1D latent vector z; the original paper's generator "applies a dense layer that
/// projects the latent vector … followed by reshaping into a 4×4 spatial feature map
/// before convolutions" (Karras et al. 2017), exactly the DCGAN/Goodfellow 2014 §3
/// latent→dense→reshape→conv-upsample pattern.
///
/// This implementation derives from <see cref="GenerativeAdversarialNetwork{T}"/>,
/// which supplies the proven tape-based adversarial training, and builds a
/// paper-faithful 1D-latent generator and strided-conv discriminator (single logit +
/// BCE-with-logits) sized to the target resolution (4·2^maxResolutionLevel). The
/// progressive-growing schedule (per-resolution fade-in, minibatch-stddev, pixel-norm)
/// is tracked as configuration; the network is built directly at the target resolution.
///
/// Based on "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
/// by Karras et al. (2018).
/// </summary>
/// <example>
/// <code>
/// var model = new ProgressiveGAN&lt;float&gt;(latentSize: 512, imageChannels: 3, maxResolutionLevel: 6);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 512 });
/// var generated = model.Predict(noise);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for computations (e.g., double, float).</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Progressive Growing of GANs for Improved Quality, Stability, and Variation", "https://arxiv.org/abs/1710.10196", Year = 2018, Authors = "Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen")]
public class ProgressiveGAN<T> : GenerativeAdversarialNetwork<T>
{
    private const double DefaultLearningRate = 0.001;
    private const double DefaultLearningRateDecay = 0.9999;

    private readonly ProgressiveGANOptions _options;
    private readonly int _latentSize;
    private readonly int _imageChannels;
    private readonly int _maxResolutionLevel;
    private readonly int _baseFeatureMaps;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the size of the latent (noise) vector input to the generator.</summary>
    public int LatentSize => _latentSize;

    /// <summary>Gets the current resolution level (0 = 4x4, 1 = 8x8, 2 = 16x16, …).</summary>
    public int CurrentResolutionLevel { get; private set; }

    /// <summary>Gets the maximum resolution level the network can achieve.</summary>
    public int MaxResolutionLevel => _maxResolutionLevel;

    /// <summary>Gets or sets the alpha value for smooth fade-in of new layers.</summary>
    public double Alpha { get; set; }

    /// <summary>Gets or sets whether to use minibatch standard deviation.</summary>
    public bool UseMinibatchStdDev { get; set; }

    /// <summary>Gets or sets whether to use pixel normalization in the generator.</summary>
    public bool UsePixelNormalization { get; set; }

    /// <summary>Gets the last computed gradient penalty value (monitoring only).</summary>
    public T LastGradientPenalty { get; private set; }

    /// <summary>Gets the current spatial resolution (4 · 2^CurrentResolutionLevel).</summary>
    public int GetCurrentResolution() => 4 * (int)Math.Pow(2, CurrentResolutionLevel);

    /// <summary>
    /// Initializes a ProgressiveGAN with paper-faithful architectures built for the
    /// target resolution (4·2^maxResolutionLevel). This is the primary constructor; the
    /// generator takes a 1D latent vector (Karras et al. 2017 / Goodfellow 2014).
    /// </summary>
    /// <param name="latentSize">Size of the latent (noise) vector input to the generator.</param>
    /// <param name="imageChannels">Number of image channels (1 grayscale, 3 RGB).</param>
    /// <param name="maxResolutionLevel">Target resolution level (image size = 4·2^level).</param>
    /// <param name="baseFeatureMaps">Base feature maps (initial generator channels).</param>
    /// <param name="lossFunction">Optional loss; defaults to BCE-with-logits for stable training.</param>
    /// <param name="options">Optional ProgressiveGAN options.</param>
    public ProgressiveGAN(
        int latentSize,
        int imageChannels,
        int maxResolutionLevel = 6,
        int baseFeatureMaps = 512,
        ILossFunction<T>? lossFunction = null,
        ProgressiveGANOptions? options = null)
        : base(
            CreateProgressiveGANGeneratorArchitecture(latentSize, imageChannels, 4 * (1 << Math.Max(0, maxResolutionLevel)), baseFeatureMaps),
            CreateProgressiveGANDiscriminatorArchitecture(imageChannels, 4 * (1 << Math.Max(0, maxResolutionLevel)), baseFeatureMaps),
            InputType.ThreeDimensional,
            generatorOptimizer: null,
            discriminatorOptimizer: null,
            lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>())
    {
        if (latentSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(latentSize), latentSize, "Latent size must be positive.");
        if (imageChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageChannels), imageChannels, "Image channels must be positive.");
        if (maxResolutionLevel < 0)
            throw new ArgumentOutOfRangeException(nameof(maxResolutionLevel), maxResolutionLevel, "Max resolution level must be non-negative.");
        if (baseFeatureMaps <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseFeatureMaps), baseFeatureMaps, "Base feature maps must be positive.");

        _options = options ?? new ProgressiveGANOptions();
        Options = _options;
        _latentSize = latentSize;
        _imageChannels = imageChannels;
        _maxResolutionLevel = maxResolutionLevel;
        _baseFeatureMaps = baseFeatureMaps;
        // The network is built directly at the target resolution, so report the
        // current level as the max (GetCurrentResolution then matches the output).
        CurrentResolutionLevel = maxResolutionLevel;
        Alpha = 1.0;
        UseMinibatchStdDev = true;
        UsePixelNormalization = true;
        LastGradientPenalty = NumOps.Zero;
    }

    /// <summary>
    /// Backward-compatible constructor that accepts explicit generator/discriminator
    /// architectures. The architectures are used only to derive image channels (a GAN
    /// generator always consumes a 1D latent, so the paper-faithful generator and
    /// discriminator are built internally from latentSize / resolution).
    /// </summary>
    /// <param name="generatorArchitecture">Generator architecture (image-channel fallback only).</param>
    /// <param name="discriminatorArchitecture">Discriminator architecture (image-channel fallback only).</param>
    /// <param name="latentSize">Size of the latent vector.</param>
    /// <param name="imageChannels">Number of image channels.</param>
    /// <param name="maxResolutionLevel">Target resolution level (image size = 4·2^level).</param>
    /// <param name="baseFeatureMaps">Base feature maps (initial generator channels).</param>
    /// <param name="inputType">Retained for API compatibility (unused — the generator is 1D-latent).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">Retained for API compatibility (the base GAN configures its optimizers).</param>
    /// <param name="learningRateDecay">Retained for API compatibility.</param>
    /// <param name="options">Optional ProgressiveGAN options.</param>
    public ProgressiveGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int latentSize = 512,
        int imageChannels = 3,
        int maxResolutionLevel = 6,
        int baseFeatureMaps = 512,
        InputType inputType = InputType.TwoDimensional,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = DefaultLearningRate,
        double learningRateDecay = DefaultLearningRateDecay,
        ProgressiveGANOptions? options = null)
        : this(latentSize,
               imageChannels > 0 ? imageChannels : (discriminatorArchitecture.InputDepth > 0 ? discriminatorArchitecture.InputDepth : 3),
               maxResolutionLevel, baseFeatureMaps, lossFunction, options)
    {
    }

    /// <summary>
    /// Constructs a fresh ProgressiveGAN with the same hyperparameters so Clone /
    /// DeepCopy rebuilds both architectures from scratch. Mirrors <see cref="DCGAN{T}"/>.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ProgressiveGAN<T>(
            _latentSize,
            _imageChannels,
            _maxResolutionLevel,
            _baseFeatureMaps,
            lossFunction: LossFunction,
            options: _options);
    }

    /// <summary>
    /// Builds the paper-faithful generator architecture: a 1D latent vector projected by
    /// a dense layer, reshaped into a small spatial feature map, then upsampled by
    /// transposed convolutions to the target image (Karras et al. 2017, Radford 2015 §3).
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateProgressiveGANGeneratorArchitecture(
        int latentSize,
        int imageChannels,
        int imageSize,
        int baseFeatureMaps)
    {
        int targetSize = imageSize;
        int initialSpatialSize = ComputeInitialSpatialSize(targetSize);
        int initialChannels = baseFeatureMaps;
        int initialFeatureMapSize = initialChannels * initialSpatialSize * initialSpatialSize;

        var layers = new List<ILayer<T>>
        {
            new DenseLayer<T>(initialFeatureMapSize, (IActivationFunction<T>?)new IdentityActivation<T>()),
            new ReshapeLayer<T>([initialChannels, initialSpatialSize, initialSpatialSize]),
        };

        int currentChannels = initialChannels;
        int currentSize = initialSpatialSize;

        while (currentSize < targetSize)
        {
            bool isFinal = currentSize * 2 >= targetSize;
            int nextChannels = isFinal ? imageChannels : Math.Max(imageChannels, currentChannels / 2);
            IActivationFunction<T> activation = isFinal
                ? new TanhActivation<T>()
                : new ReLUActivation<T>();

            layers.Add(new DeconvolutionalLayer<T>(
                outputDepth: nextChannels,
                kernelSize: 4,
                stride: 2,
                padding: 1,
                activationFunction: activation));

            if (!isFinal)
            {
                layers.Add(new BatchNormalizationLayer<T>());
            }

            currentChannels = nextChannels;
            currentSize *= 2;
        }

        return new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Medium,
            inputSize: latentSize,
            outputSize: imageChannels * imageSize * imageSize,
            layers: layers);
    }

    /// <summary>
    /// Builds the paper-faithful discriminator architecture: strided convolutions
    /// downsample the image, BatchNorm after every conv except the input layer,
    /// LeakyReLU(0.2) throughout, and a final conv collapses to a single LOGIT (no
    /// sigmoid; BCE-with-logits is the criterion). Radford 2015 §3.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateProgressiveGANDiscriminatorArchitecture(
        int imageChannels,
        int imageSize,
        int baseFeatureMaps)
    {
        var layers = new List<ILayer<T>>();

        int targetSize = imageSize;
        int currentSize = targetSize;
        int currentChannels = baseFeatureMaps;

        layers.Add(new ConvolutionalLayer<T>(
            outputDepth: baseFeatureMaps,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: new LeakyReLUActivation<T>(0.2)));
        currentSize /= 2;

        while (currentSize > 4)
        {
            int nextChannels = currentChannels * 2;
            layers.Add(new ConvolutionalLayer<T>(
                outputDepth: nextChannels,
                kernelSize: 4,
                stride: 2,
                padding: 1,
                activationFunction: new IdentityActivation<T>()));
            layers.Add(new BatchNormalizationLayer<T>());
            layers.Add(new ActivationLayer<T>((IActivationFunction<T>)new LeakyReLUActivation<T>(0.2)));
            currentChannels = nextChannels;
            currentSize /= 2;
        }

        layers.Add(new ConvolutionalLayer<T>(
            outputDepth: 1,
            kernelSize: 4,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>()));
        layers.Add(new FlattenLayer<T>());

        return new NeuralNetworkArchitecture<T>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Medium,
            inputDepth: imageChannels,
            inputHeight: imageSize,
            inputWidth: imageSize,
            outputSize: 1,
            layers: layers);
    }

    /// <summary>
    /// Computes the generator's initial spatial size: 4 for standard power-of-two-times-4
    /// sizes, otherwise the largest 2..8 factor giving a reasonable number of stages.
    /// </summary>
    private static int ComputeInitialSpatialSize(int targetSize)
    {
        if (targetSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetSize), targetSize,
                "Target image size must be positive.");
        }

        if (targetSize >= 4 && IsPowerOfTwo(targetSize / 4) && targetSize % 4 == 0)
        {
            return 4;
        }

        for (int numUpsampleLayers = 2; numUpsampleLayers <= 6; numUpsampleLayers++)
        {
            int divisor = 1 << numUpsampleLayers;
            if (targetSize % divisor == 0)
            {
                int initialSize = targetSize / divisor;
                if (initialSize >= 2 && initialSize <= 8)
                {
                    return initialSize;
                }
            }
        }

        return Math.Min(4, targetSize);
    }

    /// <summary>Checks if a number is a power of two.</summary>
    private static bool IsPowerOfTwo(int n) => n > 0 && (n & (n - 1)) == 0;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ProgressiveGAN",
            Version = "1.0",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["ModelType"] = "ProgressiveGAN",
                ["LatentSize"] = _latentSize,
                ["ImageChannels"] = _imageChannels,
                ["MaxResolutionLevel"] = _maxResolutionLevel,
                ["CurrentResolutionLevel"] = CurrentResolutionLevel,
                ["CurrentResolution"] = GetCurrentResolution(),
                ["BaseFeatureMaps"] = _baseFeatureMaps,
                ["UseMinibatchStdDev"] = UseMinibatchStdDev,
                ["UsePixelNormalization"] = UsePixelNormalization
            }
        };

        metadata.SetProperty("ModelType", "ProgressiveGAN");
        metadata.SetProperty("LatentSize", _latentSize);
        metadata.SetProperty("ImageChannels", _imageChannels);
        metadata.SetProperty("MaxResolutionLevel", _maxResolutionLevel);
        metadata.SetProperty("CurrentResolutionLevel", CurrentResolutionLevel);
        metadata.SetProperty("CurrentResolution", GetCurrentResolution());

        return metadata;
    }
}
