using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for ResNet (Residual Network) neural network architectures.
/// </summary>
/// <remarks>
/// <para>
/// This configuration class provides all the settings needed to instantiate a ResNet network.
/// It follows the AiDotNet pattern where users provide minimal configuration and the library
/// supplies sensible defaults.
/// </para>
/// <para>
/// <b>For Beginners:</b> ResNet networks are deep convolutional neural networks that use skip connections
/// to enable training of very deep architectures. This configuration lets you choose which ResNet variant
/// to use (ResNet18, ResNet34, ResNet50, ResNet101, or ResNet152), set the number of output classes for
/// your classification task, and optionally customize the input image dimensions and other parameters.
/// </para>
/// <para>
/// <b>Typical Usage:</b>
/// <code>
/// var config = new ResNetConfiguration(ResNetVariant.ResNet50, numClasses: 1000);
/// var architecture = config.ToArchitecture&lt;float&gt;();
/// var network = new ResNetNetwork&lt;float&gt;(architecture);
/// </code>
/// </para>
/// </remarks>
public class ResNetConfiguration
{
    /// <summary>
    /// Gets the ResNet variant to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variant determines how deep the network is. ResNet50 is the most
    /// commonly used variant. Deeper variants (ResNet101, ResNet152) have more capacity but require
    /// more computational resources.
    /// </para>
    /// </remarks>
    public ResNetVariant Variant { get; }

    /// <summary>
    /// Gets the number of output classes for classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the number of categories your model will classify images into.
    /// For example, if you're classifying cats vs dogs, this would be 2. For ImageNet, it's 1000.
    /// For CIFAR-10, it's 10.
    /// </para>
    /// </remarks>
    public int NumClasses { get; }

    /// <summary>
    /// Gets the height of input images in pixels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is 224, which is the standard ImageNet input size.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ResNet networks were designed for 224x224 images. While you can use
    /// different sizes, 224x224 is recommended for best results, especially when using
    /// pre-trained weights.
    /// </para>
    /// </remarks>
    public int InputHeight { get; }

    /// <summary>
    /// Gets the width of input images in pixels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is 224, which is the standard ImageNet input size.
    /// </para>
    /// </remarks>
    public int InputWidth { get; }

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is 3 for RGB images. Use 1 for grayscale images.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Color images typically have 3 channels (Red, Green, Blue).
    /// Grayscale images have 1 channel. Most pre-trained ResNet models expect 3-channel inputs.
    /// </para>
    /// </remarks>
    public int InputChannels { get; }

    /// <summary>
    /// Gets whether to include the fully connected classifier layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is true. Set to false when using ResNet as a feature extractor for transfer learning.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The classifier is the final part of the network that produces
    /// class predictions. When using ResNet for transfer learning or as a feature extractor,
    /// you may want to remove the classifier and add your own custom layers.
    /// </para>
    /// </remarks>
    public bool IncludeClassifier { get; }

    /// <summary>
    /// Gets whether to use zero-initialization for the last batch normalization in each residual block.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is true. This technique (from "Bag of Tricks" paper) improves training stability
    /// by initializing the last BN layer in each residual branch to zero, making the initial residual
    /// branch act as an identity.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a training trick that helps the network start learning
    /// more effectively. When enabled, each residual block initially acts like an identity function,
    /// making the network easier to optimize.
    /// </para>
    /// </remarks>
    public bool ZeroInitResidual { get; }

    /// <summary>
    /// Gets or sets whether to use automatic differentiation for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is false, which uses the optimized manual backward implementation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation is how neural networks learn from their mistakes.
    /// Autodiff automatically computes gradients, which is more flexible but slightly slower.
    /// The manual implementation is optimized for ResNet's specific architecture.
    /// </para>
    /// </remarks>
    public bool UseAutodiff { get; set; }

    /// <summary>
    /// Gets the computed input shape as [channels, height, width].
    /// </summary>
    public int[] InputShape => [InputChannels, InputHeight, InputWidth];

    /// <summary>
    /// Gets the total number of input features (channels * height * width).
    /// </summary>
    public int TotalInputSize => InputChannels * InputHeight * InputWidth;

    /// <summary>
    /// Gets whether this variant uses BasicBlock (ResNet18/34) or BottleneckBlock (ResNet50/101/152).
    /// </summary>
    public bool UsesBottleneck => Variant is ResNetVariant.ResNet50 or ResNetVariant.ResNet101 or ResNetVariant.ResNet152;

    /// <summary>
    /// Gets the block counts for each of the 4 stages based on the variant.
    /// </summary>
    /// <remarks>
    /// ResNet architectures have 4 stages after the initial convolution:
    /// - Stage 1 (conv2_x): 56x56 spatial, 64/256 channels
    /// - Stage 2 (conv3_x): 28x28 spatial, 128/512 channels
    /// - Stage 3 (conv4_x): 14x14 spatial, 256/1024 channels
    /// - Stage 4 (conv5_x): 7x7 spatial, 512/2048 channels
    /// </remarks>
    public int[] BlockCounts => Variant switch
    {
        ResNetVariant.ResNet18 => [2, 2, 2, 2],
        ResNetVariant.ResNet34 => [3, 4, 6, 3],
        ResNetVariant.ResNet50 => [3, 4, 6, 3],
        ResNetVariant.ResNet101 => [3, 4, 23, 3],
        ResNetVariant.ResNet152 => [3, 8, 36, 3],
        _ => throw new NotSupportedException($"ResNet variant {Variant} is not supported.")
    };

    /// <summary>
    /// Gets the base channel counts for each stage.
    /// </summary>
    /// <remarks>
    /// For BasicBlock: actual channels = base channels
    /// For BottleneckBlock: actual channels = base channels * 4 (due to expansion)
    /// </remarks>
    public int[] BaseChannels => [64, 128, 256, 512];

    /// <summary>
    /// Gets the expansion factor for the blocks.
    /// </summary>
    /// <remarks>
    /// BasicBlock has expansion 1, BottleneckBlock has expansion 4.
    /// </remarks>
    public int Expansion => UsesBottleneck ? 4 : 1;

    /// <summary>
    /// Gets the total number of convolutional layers in the network.
    /// </summary>
    public int NumConvLayers
    {
        get
        {
            int blocksPerStage = BlockCounts.Sum();
            int convsPerBlock = UsesBottleneck ? 3 : 2;
            return 1 + (blocksPerStage * convsPerBlock); // 1 for initial conv + blocks
        }
    }

    /// <summary>
    /// Gets the total number of weight layers (conv + FC).
    /// </summary>
    public int NumWeightLayers => NumConvLayers + 1; // +1 for final FC

    /// <summary>
    /// Initializes a new instance of the <see cref="ResNetConfiguration"/> class.
    /// </summary>
    /// <param name="variant">The ResNet variant to use.</param>
    /// <param name="numClasses">The number of output classes for classification.</param>
    /// <param name="inputHeight">The height of input images (default: 224).</param>
    /// <param name="inputWidth">The width of input images (default: 224).</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <param name="includeClassifier">Whether to include the classifier layers (default: true).</param>
    /// <param name="zeroInitResidual">Whether to use zero-initialization for residual branches (default: true).</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when numClasses is less than or equal to 0, or when image dimensions are invalid.
    /// </exception>
    public ResNetConfiguration(
        ResNetVariant variant,
        int numClasses,
        int inputHeight = 224,
        int inputWidth = 224,
        int inputChannels = 3,
        bool includeClassifier = true,
        bool zeroInitResidual = true)
    {
        if (numClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses),
                "Number of classes must be greater than 0.");
        }

        if (inputHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputHeight),
                "Input height must be greater than 0.");
        }

        if (inputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputWidth),
                "Input width must be greater than 0.");
        }

        if (inputChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputChannels),
                "Input channels must be greater than 0.");
        }

        // Validate minimum input size for ResNet (needs at least 7x7 after 5 pooling/stride operations)
        // For standard ResNet: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        // Minimum practical is 32x32 (like CIFAR)
        if (inputHeight < 32)
        {
            throw new ArgumentOutOfRangeException(nameof(inputHeight),
                "ResNet requires input dimensions of at least 32x32 pixels.");
        }

        if (inputWidth < 32)
        {
            throw new ArgumentOutOfRangeException(nameof(inputWidth),
                "ResNet requires input dimensions of at least 32x32 pixels.");
        }

        Variant = variant;
        NumClasses = numClasses;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        InputChannels = inputChannels;
        IncludeClassifier = includeClassifier;
        ZeroInitResidual = zeroInitResidual;
    }

    /// <summary>
    /// Creates a new configuration for ResNet50.
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <returns>A ResNetConfiguration for ResNet50.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the recommended ResNet configuration for most use cases.
    /// ResNet50 provides a good balance of accuracy and computational efficiency.
    /// </para>
    /// </remarks>
    public static ResNetConfiguration CreateResNet50(int numClasses)
    {
        return new ResNetConfiguration(ResNetVariant.ResNet50, numClasses);
    }

    /// <summary>
    /// Creates a configuration optimized for CIFAR-10/CIFAR-100 datasets.
    /// </summary>
    /// <param name="variant">The ResNet variant to use.</param>
    /// <param name="numClasses">The number of output classes (10 for CIFAR-10, 100 for CIFAR-100).</param>
    /// <returns>A ResNetConfiguration optimized for CIFAR datasets.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CIFAR images are 32x32 pixels, which is smaller than ImageNet's 224x224.
    /// This configuration adjusts the input size accordingly. Note that for CIFAR-sized images,
    /// a modified ResNet architecture is typically used that removes the initial 7x7 conv and pooling.
    /// </para>
    /// </remarks>
    public static ResNetConfiguration CreateForCIFAR(ResNetVariant variant, int numClasses)
    {
        return new ResNetConfiguration(variant, numClasses, inputHeight: 32, inputWidth: 32);
    }

    /// <summary>
    /// Creates a lightweight configuration using ResNet18.
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <returns>A ResNetConfiguration for ResNet18.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you need faster training and inference,
    /// or when working with smaller datasets where larger models might overfit.
    /// </para>
    /// </remarks>
    public static ResNetConfiguration CreateLightweight(int numClasses)
    {
        return new ResNetConfiguration(ResNetVariant.ResNet18, numClasses);
    }

    /// <summary>
    /// Creates a minimal ResNet configuration optimized for fast test execution.
    /// </summary>
    /// <remarks>
    /// Uses ResNet18 (smallest variant) with 32x32 input resolution,
    /// resulting in minimal network construction time suitable for unit tests.
    /// Construction time is typically under 50ms.
    /// </remarks>
    /// <param name="numClasses">The number of output classes.</param>
    /// <returns>A minimal ResNet configuration for testing.</returns>
    public static ResNetConfiguration CreateForTesting(int numClasses)
    {
        return new ResNetConfiguration(
            variant: ResNetVariant.ResNet18,
            numClasses: numClasses,
            inputHeight: 32,
            inputWidth: 32,
            inputChannels: 3);
    }
}
