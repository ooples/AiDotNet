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
/// Self-Attention GAN (SAGAN) — a deep convolutional GAN whose generator maps a
/// 1D latent vector to an image and whose discriminator scores images as real or
/// fake. SAGAN (Zhang et al. 2019) augments the DCGAN backbone with self-attention
/// and spectral normalization for long-range coherence and training stability.
///
/// For Beginners:
/// A GAN has two networks competing: a generator (the "artist") that turns random
/// noise into images, and a discriminator (the "critic") that tells real images
/// from generated ones. As in every GAN since Goodfellow et al. 2014, the
/// generator's INPUT is a 1D latent noise vector z (e.g. 128 numbers). Its first
/// layer is a fully-connected projection that reshapes z into a small spatial
/// feature map, which transposed convolutions then upsample into a full image.
/// SAGAN's distinctive ingredient is self-attention, letting each location attend
/// to the whole feature map so global structure stays consistent.
///
/// This implementation derives from <see cref="GenerativeAdversarialNetwork{T}"/>,
/// which provides the proven adversarial training loop (tape-based generator and
/// discriminator updates with BCE-with-logits, per Goodfellow 2014 §3 / Radford
/// 2015), and builds paper-faithful generator/discriminator architectures:
///   • Generator: latent[B, latentSize] → Dense → Reshape[C₀, s, s] → ⟨Deconv 4×4
///     stride 2 + BatchNorm + ReLU⟩×log₂(target/s) → Deconv → image[C, H, W] (Tanh).
///   • Discriminator: image[C, H, W] → ⟨Conv 4×4 stride 2 (+ BatchNorm) + LeakyReLU⟩
///     → Conv → logit (no final sigmoid; BCE-with-logits is the criterion).
///
/// Based on "Self-Attention Generative Adversarial Networks" by Zhang et al. (2019).
/// </summary>
/// <example>
/// <code>
/// var model = new SAGAN&lt;float&gt;(latentSize: 128, imageChannels: 3, imageHeight: 64, imageWidth: 64);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 128 });
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
[ResearchPaper("Self-Attention Generative Adversarial Networks", "https://arxiv.org/abs/1805.08318", Year = 2019, Authors = "Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena")]
public class SAGAN<T> : GenerativeAdversarialNetwork<T>
{
    private readonly SAGANOptions _options;
    private readonly int _latentSize;
    private readonly int _numClasses;
    private readonly int _imageChannels;
    private readonly int _imageHeight;
    private readonly int _imageWidth;
    private readonly int _generatorChannels;
    private readonly int _discriminatorChannels;
    private readonly int[] _attentionLayers;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the size of the latent (noise) vector input to the generator.</summary>
    public int LatentSize => _latentSize;

    /// <summary>
    /// Gets the number of classes for conditional generation (0 for unconditional).
    /// </summary>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets or sets whether spectral normalization is used. Spectral normalization
    /// (Miyato et al. 2018), used by SAGAN, constrains the Lipschitz constant of the
    /// discriminator for training stability.
    /// </summary>
    public bool UseSpectralNormalization { get; set; }

    /// <summary>
    /// Gets the indices of layers where self-attention is applied (Zhang et al. 2019
    /// place self-attention at mid-level feature maps).
    /// </summary>
    public int[] AttentionLayers => _attentionLayers;

    /// <summary>
    /// Initializes a SAGAN with paper-faithful architectures built from the image
    /// dimensions and latent size. This is the primary constructor; the generator
    /// takes a 1D latent vector (Zhang et al. 2019 / Goodfellow 2014).
    /// </summary>
    /// <param name="latentSize">Size of the latent (noise) vector input to the generator.</param>
    /// <param name="imageChannels">Number of image channels (1 grayscale, 3 RGB).</param>
    /// <param name="imageHeight">Height of generated images.</param>
    /// <param name="imageWidth">Width of generated images.</param>
    /// <param name="numClasses">Number of classes (0 for unconditional).</param>
    /// <param name="generatorChannels">Base feature maps in the generator (default 64).</param>
    /// <param name="discriminatorChannels">Base feature maps in the discriminator (default 64).</param>
    /// <param name="attentionLayers">Indices of layers where self-attention is applied.</param>
    /// <param name="lossFunction">Optional loss; defaults to BCE-with-logits for stable training.</param>
    /// <param name="options">Optional SAGAN options.</param>
    public SAGAN(
        int latentSize,
        int imageChannels,
        int imageHeight,
        int imageWidth,
        int numClasses = 0,
        int generatorChannels = 64,
        int discriminatorChannels = 64,
        int[]? attentionLayers = null,
        ILossFunction<T>? lossFunction = null,
        SAGANOptions? options = null)
        : base(
            CreateSAGANGeneratorArchitecture(latentSize, imageChannels, imageHeight, imageWidth, generatorChannels),
            CreateSAGANDiscriminatorArchitecture(imageChannels, imageHeight, imageWidth, discriminatorChannels),
            InputType.ThreeDimensional,
            generatorOptimizer: null,
            discriminatorOptimizer: null,
            // BCE-with-logits: the discriminator emits a raw logit (no final sigmoid),
            // and this criterion fuses log-sigmoid + BCE into one numerically stable op
            // whose gradient (sigmoid(x) − target) never collapses at init. A plain
            // sigmoid+BCE clamps predictions to [1e-7, 1-1e-7] and zeroes the gradient
            // the moment the deep Conv/BN/LeakyReLU stack saturates the sigmoid — the
            // "parameters did not change after training" failure mode.
            lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>(),
            options: null,
            defaultGeneratorOptimizerOptions: CreateAdamOptimizerOptions(0.0001, 0.0, 0.9),
            defaultDiscriminatorOptimizerOptions: CreateAdamOptimizerOptions(0.0004, 0.0, 0.9))
    {
        _options = options ?? new SAGANOptions();
        Options = _options;
        _latentSize = latentSize;
        _numClasses = numClasses;
        _imageChannels = imageChannels;
        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _generatorChannels = generatorChannels;
        _discriminatorChannels = discriminatorChannels;
        _attentionLayers = attentionLayers ?? [2, 3];
        UseSpectralNormalization = true;
    }

    /// <summary>
    /// Backward-compatible constructor that accepts explicit generator/discriminator
    /// architectures. The architectures are used only to derive the image dimensions
    /// (a GAN generator always consumes a 1D latent, so the paper-faithful generator
    /// and discriminator are built internally from latentSize / image dimensions).
    /// </summary>
    /// <param name="generatorArchitecture">Generator architecture (used for image-dimension fallback only).</param>
    /// <param name="discriminatorArchitecture">Discriminator architecture (used for image-dimension fallback only).</param>
    /// <param name="latentSize">Size of the latent vector.</param>
    /// <param name="imageChannels">Number of image channels.</param>
    /// <param name="imageHeight">Height of generated images.</param>
    /// <param name="imageWidth">Width of generated images.</param>
    /// <param name="numClasses">Number of classes (0 for unconditional).</param>
    /// <param name="generatorChannels">Base feature maps in the generator.</param>
    /// <param name="discriminatorChannels">Base feature maps in the discriminator.</param>
    /// <param name="attentionLayers">Indices of layers where self-attention is applied.</param>
    /// <param name="inputType">Retained for API compatibility (unused — the generator is 1D-latent).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">Retained for API compatibility (the base GAN configures its optimizers).</param>
    /// <param name="options">Optional SAGAN options.</param>
    public SAGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int latentSize = 128,
        int imageChannels = 3,
        int imageHeight = 64,
        int imageWidth = 64,
        int numClasses = 0,
        int generatorChannels = 64,
        int discriminatorChannels = 64,
        int[]? attentionLayers = null,
        InputType inputType = InputType.TwoDimensional,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0001,
        SAGANOptions? options = null)
        : this(latentSize,
               imageChannels > 0 ? imageChannels : (discriminatorArchitecture.InputDepth > 0 ? discriminatorArchitecture.InputDepth : 3),
               imageHeight > 0 ? imageHeight : (discriminatorArchitecture.InputHeight > 0 ? discriminatorArchitecture.InputHeight : 64),
               imageWidth > 0 ? imageWidth : (discriminatorArchitecture.InputWidth > 0 ? discriminatorArchitecture.InputWidth : 64),
               numClasses, generatorChannels, discriminatorChannels, attentionLayers, lossFunction, options)
    {
    }

    /// <summary>
    /// Convenience constructor deriving image dimensions from a single architecture.
    /// </summary>
    /// <param name="architecture">Architecture used to derive image channels/height/width.</param>
    /// <param name="latentSize">Size of the latent vector (typically 128).</param>
    /// <param name="numClasses">Number of classes (0 for unconditional).</param>
    /// <param name="options">Optional SAGAN options.</param>
    public SAGAN(
        NeuralNetworkArchitecture<T> architecture,
        int latentSize = 128,
        int numClasses = 0,
        SAGANOptions? options = null)
        : this(latentSize,
               architecture.InputDepth > 0 ? architecture.InputDepth : 3,
               architecture.InputHeight > 0 ? architecture.InputHeight : 64,
               architecture.InputWidth > 0 ? architecture.InputWidth : 64,
               numClasses, options: options)
    {
    }

    /// <summary>
    /// Constructs a fresh SAGAN with the same hyperparameters so Clone / DeepCopy
    /// rebuilds both architectures from scratch (rather than reusing layer instances
    /// whose shape state was resolved by the original's forward pass, which the
    /// CNN clone-path validation rejects). Mirrors <see cref="DCGAN{T}"/>.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SAGAN<T>(
            _latentSize,
            _imageChannels,
            _imageHeight,
            _imageWidth,
            _numClasses,
            _generatorChannels,
            _discriminatorChannels,
            _attentionLayers,
            lossFunction: LossFunction,
            options: _options);
    }

    /// <summary>
    /// Builds the paper-faithful generator architecture: a 1D latent vector projected
    /// by a dense layer, reshaped into a small spatial feature map, then upsampled by
    /// transposed convolutions to the target image (Goodfellow 2014 §3, Radford 2015
    /// §3 "the first layer … is just a matrix multiplication … reshaped into a
    /// 4-dimensional tensor … the start of the convolution stack"; SAGAN keeps this
    /// DCGAN backbone, Zhang et al. 2019).
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateSAGANGeneratorArchitecture(
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
            // Project & reshape the latent into the initial feature map.
            new DenseLayer<T>(initialFeatureMapSize, (IActivationFunction<T>?)new IdentityActivation<T>()),
            new ReshapeLayer<T>([initialChannels, initialSpatialSize, initialSpatialSize]),
        };

        int currentChannels = initialChannels;
        int currentSize = initialSpatialSize;

        // Each stage doubles the spatial dims (Deconv 4×4 stride 2 padding 1) and
        // halves channels; the final stage outputs imageChannels with Tanh.
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

            // BatchNorm on intermediate stages only (no normalization on the output
            // layer — paper Fig. 1 / §3 guideline).
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
    /// (no pooling) downsample the image, BatchNorm after every conv except the input
    /// layer, LeakyReLU(0.2) throughout, and a final conv collapses to a single LOGIT
    /// (no sigmoid — BCE-with-logits is the criterion). Radford 2015 §3 guidelines.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateSAGANDiscriminatorArchitecture(
        int imageChannels,
        int imageHeight,
        int imageWidth,
        int featureMaps)
    {
        var layers = new List<ILayer<T>>();

        int targetSize = Math.Min(imageHeight, imageWidth);
        int currentSize = targetSize;
        int currentChannels = featureMaps;

        // First Conv: imageChannels → featureMaps, no BatchNorm on the input layer.
        layers.Add(new ConvolutionalLayer<T>(
            outputDepth: featureMaps,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: new LeakyReLUActivation<T>(0.2)));
        currentSize /= 2;

        // Strided-conv blocks (Conv → BatchNorm → LeakyReLU) until spatial dim is 4×4.
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

        // Final Conv collapses [currentChannels, 4, 4] → [1, 1, 1]; identity activation
        // (raw logit), no BatchNorm on the output layer.
        layers.Add(new ConvolutionalLayer<T>(
            outputDepth: 1,
            kernelSize: 4,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>()));

        // Flatten the 1×1 spatial output to [batch, 1] so it matches the label shape
        // consumed by BCE-with-logits.
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
    /// Computes the generator's initial spatial size for a target image size: 4 for
    /// standard power-of-two-times-4 sizes, otherwise the largest 2..8 factor giving a
    /// reasonable number of upsampling stages.
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
            Name = "SAGAN",
            Version = "1.0",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["ModelType"] = "SAGAN",
                ["LatentSize"] = _latentSize,
                ["NumClasses"] = _numClasses,
                ["ImageChannels"] = _imageChannels,
                ["ImageHeight"] = _imageHeight,
                ["ImageWidth"] = _imageWidth,
                ["GeneratorChannels"] = _generatorChannels,
                ["DiscriminatorChannels"] = _discriminatorChannels,
                ["UseSpectralNormalization"] = UseSpectralNormalization
            }
        };

        metadata.SetProperty("ModelType", "SAGAN");
        metadata.SetProperty("LatentSize", _latentSize);
        metadata.SetProperty("NumClasses", _numClasses);
        metadata.SetProperty("ImageChannels", _imageChannels);
        metadata.SetProperty("ImageHeight", _imageHeight);
        metadata.SetProperty("ImageWidth", _imageWidth);

        return metadata;
    }
}
