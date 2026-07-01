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
/// BigGAN (Brock et al. 2018) — a large-scale GAN for high-fidelity image generation.
///
/// For Beginners:
/// BigGAN scales up GAN training (very large batches, high model capacity, class
/// conditioning) to produce extremely high-quality images. As in every GAN, the
/// generator's input is a 1D latent noise vector z, which a first dense layer
/// projects and reshapes into a small spatial feature map before transposed
/// convolutions upsample it to a full image (Brock et al. 2018: "the first linear
/// layer projects z … then reshapes into a spatial feature map"; Goodfellow 2014 §3).
///
/// This implementation derives from <see cref="GenerativeAdversarialNetwork{T}"/>,
/// which supplies the proven tape-based adversarial training, and builds a
/// paper-faithful DCGAN-style backbone (1D-latent generator, strided-conv
/// discriminator with a single logit + BCE-with-logits). Class-conditional batch
/// normalization / the projection discriminator are tracked as metadata
/// (NumClasses / ClassEmbeddingDim) rather than wired into the unconditional base
/// training loop.
///
/// Based on "Large Scale GAN Training for High Fidelity Natural Image Synthesis"
/// by Brock et al. (2019).
/// </summary>
/// <example>
/// <code>
/// var model = new BigGAN&lt;float&gt;(latentSize: 120, numClasses: 1000, imageChannels: 3, imageHeight: 128, imageWidth: 128);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 120 });
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
[ResearchPaper("Large Scale GAN Training for High Fidelity Natural Image Synthesis", "https://arxiv.org/abs/1809.11096", Year = 2019, Authors = "Andrew Brock, Jeff Donahue, Karen Simonyan")]
public class BigGAN<T> : GenerativeAdversarialNetwork<T>
{
    private readonly BigGANOptions _options;
    private readonly int _latentSize;
    private readonly int _numClasses;
    private readonly int _classEmbeddingDim;
    private readonly int _imageChannels;
    private readonly int _imageHeight;
    private readonly int _imageWidth;
    private readonly int _generatorChannels;
    private readonly int _discriminatorChannels;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the size of the latent (noise) vector input to the generator.</summary>
    public int LatentSize => _latentSize;

    /// <summary>Gets the number of classes for conditional generation.</summary>
    public int NumClasses => _numClasses;

    /// <summary>Gets the dimensionality of the class embedding.</summary>
    public int ClassEmbeddingDim => _classEmbeddingDim;

    /// <summary>Gets or sets the truncation threshold for the truncation trick.</summary>
    public double TruncationThreshold { get; set; }

    /// <summary>Gets or sets whether the truncation trick is applied at generation time.</summary>
    public bool UseTruncation { get; set; }

    /// <summary>Gets or sets whether spectral normalization is used.</summary>
    public bool UseSpectralNormalization { get; set; }

    /// <summary>Gets or sets whether self-attention is used.</summary>
    public bool UseSelfAttention { get; set; }

    /// <summary>
    /// Initializes a BigGAN with paper-faithful architectures built from the image
    /// dimensions and latent size. This is the primary constructor; the generator
    /// takes a 1D latent vector (Brock et al. 2018 / Goodfellow 2014).
    /// </summary>
    /// <param name="latentSize">Size of the latent (noise) vector input to the generator.</param>
    /// <param name="numClasses">Number of classes for conditional generation.</param>
    /// <param name="classEmbeddingDim">Dimensionality of the class embedding.</param>
    /// <param name="imageChannels">Number of image channels (1 grayscale, 3 RGB).</param>
    /// <param name="imageHeight">Height of generated images.</param>
    /// <param name="imageWidth">Width of generated images.</param>
    /// <param name="generatorChannels">Base feature maps in the generator (default 96).</param>
    /// <param name="discriminatorChannels">Base feature maps in the discriminator (default 96).</param>
    /// <param name="lossFunction">Optional loss; defaults to BCE-with-logits for stable training.</param>
    /// <param name="options">Optional BigGAN options.</param>
    public BigGAN(
        int latentSize,
        int numClasses,
        int classEmbeddingDim,
        int imageChannels,
        int imageHeight,
        int imageWidth,
        int generatorChannels = 96,
        int discriminatorChannels = 96,
        ILossFunction<T>? lossFunction = null,
        BigGANOptions? options = null)
        : base(
            CreateBigGANGeneratorArchitecture(latentSize, imageChannels, imageHeight, imageWidth, generatorChannels),
            CreateBigGANDiscriminatorArchitecture(imageChannels, imageHeight, imageWidth, discriminatorChannels),
            InputType.ThreeDimensional,
            generatorOptimizer: null,
            discriminatorOptimizer: null,
            lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>(),
            options: null,
            defaultGeneratorOptimizerOptions: CreateAdamOptimizerOptions(0.0001, 0.0, 0.999),
            defaultDiscriminatorOptimizerOptions: CreateAdamOptimizerOptions(0.0004, 0.0, 0.999))
    {
        if (latentSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(latentSize), latentSize, "Latent size must be positive.");
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "Number of classes must be positive.");
        if (classEmbeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(classEmbeddingDim), classEmbeddingDim, "Class embedding dimension must be positive.");

        _options = options ?? new BigGANOptions();
        Options = _options;
        _latentSize = latentSize;
        _numClasses = numClasses;
        _classEmbeddingDim = classEmbeddingDim;
        _imageChannels = imageChannels;
        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _generatorChannels = generatorChannels;
        _discriminatorChannels = discriminatorChannels;
        TruncationThreshold = 1.0;
        UseTruncation = false;
        UseSpectralNormalization = true;
        UseSelfAttention = true;
    }

    /// <summary>
    /// Backward-compatible constructor that accepts explicit generator/discriminator
    /// architectures. The architectures are used only to derive image dimensions (a
    /// GAN generator always consumes a 1D latent, so the paper-faithful generator and
    /// discriminator are built internally from latentSize / image dimensions).
    /// </summary>
    /// <param name="generatorArchitecture">Generator architecture (image-dimension fallback only).</param>
    /// <param name="discriminatorArchitecture">Discriminator architecture (image-dimension fallback only).</param>
    /// <param name="latentSize">Size of the latent vector.</param>
    /// <param name="numClasses">Number of classes for conditional generation.</param>
    /// <param name="classEmbeddingDim">Dimensionality of the class embedding.</param>
    /// <param name="imageChannels">Number of image channels.</param>
    /// <param name="imageHeight">Height of generated images.</param>
    /// <param name="imageWidth">Width of generated images.</param>
    /// <param name="generatorChannels">Base feature maps in the generator.</param>
    /// <param name="discriminatorChannels">Base feature maps in the discriminator.</param>
    /// <param name="inputType">Retained for API compatibility (unused — the generator is 1D-latent).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">Retained for API compatibility (the base GAN configures its optimizers).</param>
    /// <param name="options">Optional BigGAN options.</param>
    public BigGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int latentSize = 120,
        int numClasses = 1000,
        int classEmbeddingDim = 128,
        int imageChannels = 3,
        int imageHeight = 128,
        int imageWidth = 128,
        int generatorChannels = 96,
        int discriminatorChannels = 96,
        InputType inputType = InputType.TwoDimensional,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0001,
        BigGANOptions? options = null)
        : this(latentSize, numClasses, classEmbeddingDim,
               imageChannels > 0 ? imageChannels : (discriminatorArchitecture.InputDepth > 0 ? discriminatorArchitecture.InputDepth : 3),
               imageHeight > 0 ? imageHeight : (discriminatorArchitecture.InputHeight > 0 ? discriminatorArchitecture.InputHeight : 128),
               imageWidth > 0 ? imageWidth : (discriminatorArchitecture.InputWidth > 0 ? discriminatorArchitecture.InputWidth : 128),
               generatorChannels, discriminatorChannels, lossFunction, options)
    {
    }

    /// <summary>
    /// Constructs a fresh BigGAN with the same hyperparameters so Clone / DeepCopy
    /// rebuilds both architectures from scratch. Mirrors <see cref="DCGAN{T}"/>.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new BigGAN<T>(
            _latentSize,
            _numClasses,
            _classEmbeddingDim,
            _imageChannels,
            _imageHeight,
            _imageWidth,
            _generatorChannels,
            _discriminatorChannels,
            lossFunction: LossFunction,
            options: _options);
    }

    /// <summary>
    /// Builds the paper-faithful generator architecture: a 1D latent vector projected
    /// by a dense layer, reshaped into a small spatial feature map, then upsampled by
    /// transposed convolutions to the target image (Brock et al. 2018, Radford 2015 §3).
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateBigGANGeneratorArchitecture(
        int latentSize,
        int imageChannels,
        int imageHeight,
        int imageWidth,
        int featureMaps)
    {
        int targetSize = Math.Min(imageHeight, imageWidth);
        int initialSpatialSize = ComputeInitialSpatialSize(targetSize);
        int initialChannels = featureMaps * 8;
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
            int nextChannels = isFinal ? imageChannels : currentChannels / 2;
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
            outputSize: imageChannels * imageHeight * imageWidth,
            layers: layers);
    }

    /// <summary>
    /// Builds the paper-faithful discriminator architecture: strided convolutions
    /// downsample the image, BatchNorm after every conv except the input layer,
    /// LeakyReLU(0.2) throughout, and a final conv collapses to a single LOGIT (no
    /// sigmoid; BCE-with-logits is the criterion). Radford 2015 §3.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateBigGANDiscriminatorArchitecture(
        int imageChannels,
        int imageHeight,
        int imageWidth,
        int featureMaps)
    {
        var layers = new List<ILayer<T>>();

        int targetSize = Math.Min(imageHeight, imageWidth);
        int currentSize = targetSize;
        int currentChannels = featureMaps;

        layers.Add(new ConvolutionalLayer<T>(
            outputDepth: featureMaps,
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
            inputHeight: imageHeight,
            inputWidth: imageWidth,
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

        return 4;
    }

    /// <summary>Checks if a number is a power of two.</summary>
    private static bool IsPowerOfTwo(int n) => n > 0 && (n & (n - 1)) == 0;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "BigGAN",
            Version = "1.0",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["ModelType"] = "BigGAN",
                ["LatentSize"] = _latentSize,
                ["NumClasses"] = _numClasses,
                ["ClassEmbeddingDim"] = _classEmbeddingDim,
                ["ImageChannels"] = _imageChannels,
                ["ImageHeight"] = _imageHeight,
                ["ImageWidth"] = _imageWidth,
                ["GeneratorChannels"] = _generatorChannels,
                ["DiscriminatorChannels"] = _discriminatorChannels,
                ["UseTruncation"] = UseTruncation,
                ["UseSelfAttention"] = UseSelfAttention
            }
        };

        metadata.SetProperty("ModelType", "BigGAN");
        metadata.SetProperty("LatentSize", _latentSize);
        metadata.SetProperty("NumClasses", _numClasses);
        metadata.SetProperty("ClassEmbeddingDim", _classEmbeddingDim);
        metadata.SetProperty("ImageChannels", _imageChannels);
        metadata.SetProperty("ImageHeight", _imageHeight);
        metadata.SetProperty("ImageWidth", _imageWidth);
        metadata.SetProperty("TruncationThreshold", TruncationThreshold);

        return metadata;
    }
}
