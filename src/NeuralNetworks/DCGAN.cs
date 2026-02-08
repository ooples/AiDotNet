using AiDotNet.NeuralNetworks.Options;

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
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DCGAN<T> : GenerativeAdversarialNetwork<T>
{
    private readonly DCGANOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
            lossFunction)
    {
        _options = options ?? new DCGANOptions();
        Options = _options;
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
        // For DCGAN generator, the latent vector is first projected and reshaped to an initial
        // 3D feature map. The typical starting spatial size is 4x4 which gets upsampled through
        // transposed convolutions. The depth represents the number of feature channels.
        // Note: The actual latent vector (1D) handling is done by the first projection layer.

        // Compute the initial spatial size based on image dimensions.
        // Standard DCGAN uses powers of 2 (4->8->16->32->64...).
        // We compute the smallest valid initial size that can upsample to target dimensions.
        int targetSize = Math.Min(imageHeight, imageWidth);
        int initialSpatialSize = ComputeInitialSpatialSize(targetSize);
        int initialChannels = featureMaps * 8;  // Standard DCGAN uses 8x feature maps initially

        return new NeuralNetworkArchitecture<T>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Medium,
            inputDepth: initialChannels,
            inputHeight: initialSpatialSize,
            inputWidth: initialSpatialSize,
            outputSize: imageChannels * imageHeight * imageWidth);
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
        // DCGAN discriminator takes 3D images as input (channels x height x width)
        // and outputs a single probability value for real/fake classification
        return new NeuralNetworkArchitecture<T>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Medium,
            inputDepth: imageChannels,
            inputHeight: imageHeight,
            inputWidth: imageWidth,
            outputSize: 1);
    }
}
