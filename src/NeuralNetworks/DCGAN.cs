using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Deep Convolutional Generative Adversarial Network (DCGAN), an architecture that uses
/// convolutional and transposed convolutional layers with specific design guidelines for stable training.
/// </summary>
/// <remarks>
/// <para>
/// DCGAN introduces several architectural constraints that improve training stability:
/// - Replace pooling layers with strided convolutions (discriminator) and fractional-strided
///   convolutions/transposed convolutions (generator)
/// - Use batch normalization in both generator and discriminator
/// - Remove fully connected hidden layers for deeper architectures
/// - Use ReLU activation in generator for all layers except output (uses Tanh)
/// - Use LeakyReLU activation in discriminator for all layers
/// </para>
/// <para><b>For Beginners:</b> DCGAN is an improved version of the basic GAN that uses specific
/// design patterns to make training more stable and produce higher quality images.
///
/// Key improvements over vanilla GAN:
/// - Uses convolutional layers specifically designed for images
/// - Includes batch normalization to stabilize training
/// - Follows proven architectural guidelines
/// - Produces sharper, more realistic images
///
/// Reference: Radford et al., "Unsupervised Representation Learning with Deep Convolutional
/// Generative Adversarial Networks" (2015)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new DCGANOptions { LatentSize = 100, ImageSize = 64 };
/// var model = new DCGAN&lt;float&gt;(options);
/// var noise = Tensor&lt;float&gt;.Random(new[] { 1, 100 });
/// var generated = model.Predict(noise);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", "https://arxiv.org/abs/1511.06434", Year = 2016, Authors = "Alec Radford, Luke Metz, Soumith Chintala")]
public class DCGAN<T> : GenerativeAdversarialNetwork<T>
{
    private readonly DCGANOptions _options;
    private readonly int _latentSize;
    private readonly int _imageChannels;
    private readonly int _imageHeight;
    private readonly int _imageWidth;
    private readonly int _generatorFeatureMaps;
    private readonly int _discriminatorFeatureMaps;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Creates a DCGAN with default dimensions derived from the architecture.
    /// Per Radford et al. 2016: latent size 100, 64×64 images, 64 feature maps.
    /// </summary>
    /// <param name="architecture">The architecture used to derive image channels/height/width.</param>
    /// <param name="latentSize">The size of the latent (noise) vector. Default is 100.</param>
    /// <param name="generatorFeatureMaps">Feature maps in generator's first layer. Default is 64.</param>
    /// <param name="discriminatorFeatureMaps">Feature maps in discriminator's first layer. Default is 64.</param>
    /// <param name="options">Optional DCGAN options.</param>
    public DCGAN(
        NeuralNetworkArchitecture<T> architecture,
        int latentSize = 100,
        int generatorFeatureMaps = 64,
        int discriminatorFeatureMaps = 64,
        DCGANOptions? options = null)
        : this(latentSize,
               architecture.InputDepth > 0 ? architecture.InputDepth : 3,
               architecture.InputHeight > 0 ? architecture.InputHeight : 64,
               architecture.InputWidth > 0 ? architecture.InputWidth : 64,
               generatorFeatureMaps, discriminatorFeatureMaps,
               lossFunction: null, options: options)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DCGAN{T}"/> class with default DCGAN architecture.
    /// </summary>
    /// <param name="latentSize">The size of the latent (noise) vector input to the generator.</param>
    /// <param name="imageChannels">The number of channels in the output images (e.g., 1 for grayscale, 3 for RGB).</param>
    /// <param name="imageHeight">The height of the output images.</param>
    /// <param name="imageWidth">The width of the output images.</param>
    /// <param name="generatorFeatureMaps">The number of feature maps in the generator's first layer.</param>
    /// <param name="discriminatorFeatureMaps">The number of feature maps in the discriminator's first layer.</param>
    /// <param name="lossFunction">Optional loss function. If not provided, binary cross-entropy will be used.</param>
    /// <param name="options">Optional DCGAN options.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a DCGAN with the standard architecture following the guidelines
    /// from the original paper. The generator uses transposed convolutions to upsample the latent
    /// vector into a full image, while the discriminator uses strided convolutions to downsample
    /// images into a classification score.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a DCGAN with proven settings from the research paper.
    ///
    /// Parameters explained:
    /// - latentSize: How many random numbers to use as input (typically 100)
    /// - imageChannels: 1 for black/white images, 3 for color images
    /// - imageHeight/imageWidth: The size of images to generate (e.g., 64x64)
    /// - generatorFeatureMaps: Controls the generator's capacity (typically 64)
    /// - discriminatorFeatureMaps: Controls the discriminator's capacity (typically 64)
    ///
    /// The default learning rate (0.0002) is lower than typical GANs for more stable training.
    /// </para>
    /// </remarks>
    public DCGAN(
        int latentSize,
        int imageChannels,
        int imageHeight,
        int imageWidth,
        int generatorFeatureMaps = 64,
        int discriminatorFeatureMaps = 64,
        ILossFunction<T>? lossFunction = null,
        DCGANOptions? options = null)
        : base(
            CreateDCGANGeneratorArchitecture(latentSize, imageChannels, imageHeight, imageWidth, generatorFeatureMaps),
            CreateDCGANDiscriminatorArchitecture(imageChannels, imageHeight, imageWidth, discriminatorFeatureMaps),
            InputType.ThreeDimensional,
            generatorOptimizer: null,
            discriminatorOptimizer: null,
            // Default to BinaryCrossEntropyWithLogitsLoss for paper-faithful
            // training: the Discriminator's architecture emits a LOGIT (no
            // final sigmoid — see CreateDCGANDiscriminatorArchitecture) and
            // BCEWithLogits fuses log-sigmoid + BCE in one numerically
            // stable op (gradient = sigmoid(x) − target, which never
            // saturates regardless of how extreme the disc's pre-activation
            // gets at init). The previous default (a plain BCELoss on
            // sigmoid-activated output, derived from
            // GetDefaultLossFunction(BinaryClassification)) killed gradients
            // at init via the `TensorClamp(predicted, 1e-7, 1-1e-7)` step:
            // DCGAN's deep Conv+BN+LeakyReLU stack saturates the sigmoid
            // before any training has happened, and clamp's gradient is
            // identically zero outside the [eps, 1-eps] interval — the
            // exact "Parameters did not change after training" / "No
            // parameters changed after training" cluster the per-step
            // invariants caught. Callers can still pass an explicit loss
            // function to override this.
            lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>(),
            options: null,
            defaultGeneratorOptimizerOptions: CreateAdamOptimizerOptions(0.0002, 0.5),
            defaultDiscriminatorOptimizerOptions: CreateAdamOptimizerOptions(0.0002, 0.5))
    {
        _options = options ?? new DCGANOptions();
        Options = _options;
        // Remember construction params so CreateNewInstance can rebuild a
        // fresh DCGAN with identical architecture rather than going through
        // GenerativeAdversarialNetwork.CreateNewInstance — that one reuses
        // the existing Generator/Discriminator architectures' Layers list,
        // which after a forward pass carries resolved shape state that
        // ConvolutionalNeuralNetwork.ValidateCustomLayers flags as
        // "Layer N not compatible with Layer N+1" on the clone path.
        _latentSize = latentSize;
        _imageChannels = imageChannels;
        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _generatorFeatureMaps = generatorFeatureMaps;
        _discriminatorFeatureMaps = discriminatorFeatureMaps;
    }

    /// <summary>
    /// Constructs a fresh DCGAN with the same paper-faithful hyperparameters
    /// so Clone / DeepCopy produces a deep-independent network whose layer
    /// list isn't shared with the original. The base
    /// <see cref="GenerativeAdversarialNetwork{T}.CreateNewInstance"/> passes
    /// the existing <c>Generator.Architecture</c> and
    /// <c>Discriminator.Architecture</c> straight through to the GAN ctor,
    /// which wraps them in fresh
    /// <see cref="ConvolutionalNeuralNetwork{T}"/> shells whose
    /// <c>InitializeLayers</c> calls <c>ValidateCustomLayers</c> against
    /// layer instances that already had their shape state resolved by the
    /// original network's forward pass — and that validation rejects the
    /// resolved shape chain. Going through DCGAN's own ctor instead rebuilds
    /// both architectures (and their layer lists) from scratch.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Pass the current LossFunction through so cloning a model trained
        // with a custom objective doesn't silently downgrade to the default.
        return new DCGAN<T>(
            _latentSize,
            _imageChannels,
            _imageHeight,
            _imageWidth,
            _generatorFeatureMaps,
            _discriminatorFeatureMaps,
            lossFunction: LossFunction,
            options: _options);
    }

    /// <summary>
    /// Creates the architecture for the DCGAN generator following the original paper's guidelines.
    /// </summary>
    /// <param name="latentSize">The size of the latent vector.</param>
    /// <param name="imageChannels">The number of output image channels.</param>
    /// <param name="imageHeight">The height of output images.</param>
    /// <param name="imageWidth">The width of output images.</param>
    /// <param name="featureMaps">The number of feature maps in the first layer.</param>
    /// <returns>A neural network architecture configured for the DCGAN generator.</returns>
    /// <remarks>
    /// <para>
    /// The generator architecture follows these DCGAN guidelines:
    /// 1. Starts with a dense layer that reshapes the latent vector
    /// 2. Uses transposed convolutions (fractional-strided convolutions) to upsample
    /// 3. Applies batch normalization after each layer except the output
    /// 4. Uses ReLU activation for all layers except the output
    /// 5. Uses Tanh activation for the output layer
    /// </para>
    /// <para><b>For Beginners:</b> This creates the "artist" part of DCGAN.
    ///
    /// The generator architecture:
    /// - Takes random noise as input
    /// - Gradually upsamples it through several layers
    /// - Each layer doubles the spatial dimensions
    /// - Produces a full-resolution image at the output
    /// - Uses batch normalization to stabilize training
    /// </para>
    /// </remarks>
    private static NeuralNetworkArchitecture<T> CreateDCGANGeneratorArchitecture(
        int latentSize,
        int imageChannels,
        int imageHeight,
        int imageWidth,
        int featureMaps)
    {
        // Paper-faithful DCGAN generator (Radford et al. 2015 §3):
        //   "The first layer of the GAN, which takes a uniform noise distribution
        //    Z as input, could be called fully connected as it is just a matrix
        //    multiplication, but the result is reshaped into a 4-dimensional
        //    tensor and used as the start of the convolution stack."
        //
        // Layout:
        //   latent[B, latentSize]
        //     → DenseLayer  (latentSize → 8·featureMaps · 4 · 4) [linear]
        //     → ReshapeLayer to [8·featureMaps, 4, 4]
        //     → log2(target/4) × { Deconv 4×4 stride 2 [ReLU] + BatchNorm }
        //     → final Deconv 4×4 stride 2 → [imageChannels, H, W] [Tanh]
        //
        // BatchNorm sits AFTER the deconv (per paper Fig. 1) and ReLU is the
        // deconv's built-in activation; the final layer uses Tanh and no BN
        // so the [-1, 1] output range matches the pre-processed image distribution.

        int targetSize = Math.Min(imageHeight, imageWidth);
        int initialSpatialSize = ComputeInitialSpatialSize(targetSize);
        int initialChannels = featureMaps * 8;
        int initialFeatureMapSize = initialChannels * initialSpatialSize * initialSpatialSize;

        var layers = new List<ILayer<T>>
        {
            // Project & reshape latent to initial feature map
            new DenseLayer<T>(initialFeatureMapSize, (IActivationFunction<T>?)new IdentityActivation<T>()),
            new ReshapeLayer<T>([initialChannels, initialSpatialSize, initialSpatialSize]),
        };

        int currentChannels = initialChannels;
        int currentSize = initialSpatialSize;

        // Upsample stages: each doubles spatial dims and halves channels until
        // we reach the target size. Last stage outputs imageChannels with Tanh.
        while (currentSize < targetSize)
        {
            bool isFinal = currentSize * 2 >= targetSize;
            int nextChannels = isFinal ? imageChannels : currentChannels / 2;
            IActivationFunction<T> activation = isFinal
                ? new TanhActivation<T>()
                : new ReLUActivation<T>();

            // Deconv 4×4 stride 2 padding 1 doubles spatial dim exactly:
            //   out = (in - 1) · stride − 2·padding + kernel = (in − 1)·2 − 2 + 4 = 2·in
            layers.Add(new DeconvolutionalLayer<T>(
                outputDepth: nextChannels,
                kernelSize: 4,
                stride: 2,
                padding: 1,
                activationFunction: activation));

            // BatchNorm only on intermediate stages — final layer outputs raw
            // image so no normalization (paper Fig. 1 / Sec. 3 guideline).
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
    /// Computes the initial spatial size for the generator based on target image size.
    /// For standard DCGAN architecture, the generator upsamples by doubling at each layer.
    /// </summary>
    private static int ComputeInitialSpatialSize(int targetSize)
    {
        // Standard initial spatial sizes are 4 or 2
        // For target sizes like 28 (MNIST), we use 7 as initial size (7->14->28)
        // For target sizes like 32 (CIFAR), we use 4 as initial size (4->8->16->32)
        // For target sizes like 64, we use 4 as initial size (4->8->16->32->64)

        if (targetSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetSize), targetSize,
                "Target image size must be positive.");
        }

        // Check if targetSize is a power of 2 times 4 (standard DCGAN)
        if (targetSize >= 4 && IsPowerOfTwo(targetSize / 4) && targetSize % 4 == 0)
        {
            return 4;  // Standard DCGAN initial size
        }

        // For non-standard sizes (like 28 for MNIST), find largest factor that divides evenly
        // and results in a reasonable number of upsampling steps (2-6)
        for (int numUpsampleLayers = 2; numUpsampleLayers <= 6; numUpsampleLayers++)
        {
            int divisor = 1 << numUpsampleLayers;  // 2^numUpsampleLayers
            if (targetSize % divisor == 0)
            {
                int initialSize = targetSize / divisor;
                if (initialSize >= 2 && initialSize <= 8)
                {
                    return initialSize;
                }
            }
        }

        // Fallback: use 4 and let architecture handle any mismatch
        return 4;
    }

    /// <summary>
    /// Checks if a number is a power of 2.
    /// </summary>
    private static bool IsPowerOfTwo(int n)
    {
        return n > 0 && (n & (n - 1)) == 0;
    }

    /// <summary>
    /// Creates the architecture for the DCGAN discriminator following the original paper's guidelines.
    /// </summary>
    /// <param name="imageChannels">The number of input image channels.</param>
    /// <param name="imageHeight">The height of input images.</param>
    /// <param name="imageWidth">The width of input images.</param>
    /// <param name="featureMaps">The number of feature maps in the first layer.</param>
    /// <returns>A neural network architecture configured for the DCGAN discriminator.</returns>
    /// <remarks>
    /// <para>
    /// The discriminator architecture follows these DCGAN guidelines:
    /// 1. Uses strided convolutions instead of pooling for downsampling
    /// 2. Applies batch normalization after each layer except the first and output
    /// 3. Uses LeakyReLU activation with slope 0.2 for all layers except output
    /// 4. Uses Sigmoid activation for the output layer (binary classification)
    /// 5. No fully connected hidden layers
    /// </para>
    /// <para><b>For Beginners:</b> This creates the "detective" part of DCGAN.
    ///
    /// The discriminator architecture:
    /// - Takes an image as input (real or generated)
    /// - Gradually downsamples it through several convolutional layers
    /// - Each layer halves the spatial dimensions
    /// - Produces a single probability score (0-1) at the output
    /// - Uses LeakyReLU to allow small negative signals
    /// </para>
    /// </remarks>
    private static NeuralNetworkArchitecture<T> CreateDCGANDiscriminatorArchitecture(
        int imageChannels,
        int imageHeight,
        int imageWidth,
        int featureMaps)
    {
        // Paper-faithful DCGAN discriminator (Radford et al. 2015 §3,
        // "Architecture guidelines for stable Deep Convolutional GANs"):
        //
        //   • Strided convolutions (no max-pool) for downsampling.
        //   • Batch normalization after every Conv EXCEPT the input layer
        //     (paper §3 bullet 2: "Use batchnorm in both the generator and
        //     discriminator … Directly applying batchnorm to all layers
        //     resulted in sample oscillation and model instability. This was
        //     avoided by not applying batchnorm to the generator output
        //     layer and the discriminator input layer.").
        //   • LeakyReLU activation with slope 0.2 for all layers (§3 bullet
        //     5: "Use LeakyReLU activation in the discriminator for all
        //     layers" — slope reported in §4 / Table 1).
        //   • Final layer emits a single LOGIT (not a sigmoid probability):
        //     stable training pairs this with BinaryCrossEntropyWithLogitsLoss
        //     in the GAN ctor (Goodfellow 2014 §3 numerical-stability note,
        //     standard PyTorch convention nn.BCEWithLogitsLoss). Using a
        //     sigmoid + plain BCE collapses gradients at init the moment
        //     the deep Conv+BN+LeakyReLU stack saturates the final sigmoid,
        //     and the optimizer step leaves every weight at its initial
        //     value — observed directly as the
        //     DCGANTests.Training_ShouldChangeParameters / GradientFlow
        //     "Parameters did not change after training" failures.
        //
        // Per-stage spatial-dim arithmetic with the 4×4 kernel / stride 2 /
        // padding 1 contract: out = (in + 2·padding − kernel) / stride + 1
        // = (in − 2) / 2 + 1 = in / 2 (exact for even `in`). Channels grow
        // featureMaps → featureMaps·2 → featureMaps·4 → … until the spatial
        // dim reaches 4×4, at which point a final 4×4 / stride 1 / padding 0
        // Conv with `outputDepth = 1` collapses the spatial dimensions to
        // 1×1 — equivalent to Flatten + Dense(1) but matches the paper's
        // strided-conv-only block layout.

        var layers = new List<ILayer<T>>();

        int targetSize = Math.Min(imageHeight, imageWidth);
        int currentSize = targetSize;
        int currentChannels = featureMaps;

        var leakyReLU = new LeakyReLUActivation<T>(0.2);

        // First Conv: imageChannels → featureMaps. NO batch norm on the
        // input layer per the paper. LeakyReLU activation built in.
        layers.Add(new ConvolutionalLayer<T>(
            outputDepth: featureMaps,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: leakyReLU));
        currentSize /= 2;

        // Repeat strided-conv blocks (Conv → BN → LeakyReLU) doubling
        // channels and halving spatial dim, until spatial dim is 4×4.
        while (currentSize > 4)
        {
            int nextChannels = currentChannels * 2;
            layers.Add(new ConvolutionalLayer<T>(
                outputDepth: nextChannels,
                kernelSize: 4,
                stride: 2,
                padding: 1,
                activationFunction: new IdentityActivation<T>()));
            // BatchNorm after Conv per paper Fig. 1 / §3 bullet 2.
            layers.Add(new BatchNormalizationLayer<T>());
            // LeakyReLU as a separate layer (Conv emits an identity-
            // activated pre-norm output, then BN normalizes, then
            // LeakyReLU applies the non-linearity). Matches the canonical
            // PyTorch DCGAN reference impl ordering.
            layers.Add(new ActivationLayer<T>((IActivationFunction<T>)new LeakyReLUActivation<T>(0.2)));
            currentChannels = nextChannels;
            currentSize /= 2;
        }

        // Final Conv: collapse [currentChannels, 4, 4] → [1, 1, 1] using
        // 4×4 / stride 1 / padding 0. Identity activation — the logit is
        // consumed by BCEWithLogitsLoss. NO BatchNorm on the output layer
        // per the paper (§3 bullet 2).
        layers.Add(new ConvolutionalLayer<T>(
            outputDepth: 1,
            kernelSize: 4,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>()));

        // Flatten the 1×1 spatial output to a rank-2 [batch, 1] tensor so
        // the consumer (BCEWithLogitsLoss) sees the same shape as the
        // realLabels / fakeLabels (CreateLabelTensor returns
        // [batchSize, 1]).
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
}
