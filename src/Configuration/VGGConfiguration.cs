using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for VGG neural network architectures.
/// </summary>
/// <remarks>
/// <para>
/// This configuration class provides all the settings needed to instantiate a VGG network.
/// It follows the AiDotNet pattern where users provide minimal configuration and the library
/// supplies sensible defaults.
/// </para>
/// <para>
/// <b>For Beginners:</b> VGG networks are deep convolutional neural networks designed for image
/// classification. This configuration lets you choose which VGG variant to use (VGG11, VGG13,
/// VGG16, or VGG19), set the number of output classes for your classification task, and optionally
/// customize the input image dimensions and other parameters.
/// </para>
/// <para>
/// <b>Typical Usage:</b>
/// <code>
/// var config = new VGGConfiguration(VGGVariant.VGG16_BN, numClasses: 10);
/// var architecture = config.ToArchitecture&lt;float&gt;();
/// var network = new VGGNetwork&lt;float&gt;(architecture);
/// </code>
/// </para>
/// </remarks>
public class VGGConfiguration
{
    /// <summary>
    /// Gets the VGG variant to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variant determines how deep the network is. VGG16 is the most
    /// commonly used. Variants with "_BN" suffix include batch normalization which usually
    /// improves training stability and final accuracy.
    /// </para>
    /// </remarks>
    public VGGVariant Variant { get; }

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
    /// <b>For Beginners:</b> VGG networks were designed for 224x224 images. While you can use
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
    /// Grayscale images have 1 channel. Most pre-trained VGG models expect 3-channel inputs.
    /// </para>
    /// </remarks>
    public int InputChannels { get; }

    /// <summary>
    /// Gets the dropout rate applied to the fully connected layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is 0.5, which is the original VGG dropout rate.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Dropout is a regularization technique that randomly "drops out"
    /// (sets to zero) a fraction of neurons during training. This helps prevent overfitting.
    /// A rate of 0.5 means 50% of neurons are dropped during each training step.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; }

    /// <summary>
    /// Gets whether to use batch normalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is automatically determined based on the selected variant (variants ending in "_BN").
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Batch normalization normalizes the activations of each layer,
    /// which helps the network train faster and more stably. Variants with batch normalization
    /// typically achieve better accuracy.
    /// </para>
    /// </remarks>
    public bool UseBatchNormalization { get; }

    /// <summary>
    /// Gets whether to include the fully connected classifier layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is true. Set to false when using VGG as a feature extractor for transfer learning.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The classifier is the final part of the network that produces
    /// class predictions. When using VGG for transfer learning or as a feature extractor,
    /// you may want to remove the classifier and add your own custom layers.
    /// </para>
    /// </remarks>
    public bool IncludeClassifier { get; }

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
    /// The manual implementation is optimized for VGG's specific architecture.
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
    /// Gets the number of convolutional layers based on the variant.
    /// </summary>
    public int NumConvLayers => Variant switch
    {
        VGGVariant.VGG11 or VGGVariant.VGG11_BN => 8,
        VGGVariant.VGG13 or VGGVariant.VGG13_BN => 10,
        VGGVariant.VGG16 or VGGVariant.VGG16_BN => 13,
        VGGVariant.VGG19 or VGGVariant.VGG19_BN => 16,
        _ => throw new NotSupportedException($"VGG variant {Variant} is not supported.")
    };

    /// <summary>
    /// Gets the total number of weight layers (conv + FC).
    /// </summary>
    public int NumWeightLayers => NumConvLayers + 3; // 3 fully connected layers

    /// <summary>
    /// Gets the layer configuration for each VGG block.
    /// </summary>
    /// <remarks>
    /// Each inner array represents a block, containing the number of filters for each conv layer.
    /// Blocks are separated by max pooling layers.
    /// </remarks>
    public int[][] BlockConfiguration => Variant switch
    {
        VGGVariant.VGG11 or VGGVariant.VGG11_BN =>
        [
            [64],           // Block 1
            [128],          // Block 2
            [256, 256],     // Block 3
            [512, 512],     // Block 4
            [512, 512]      // Block 5
        ],
        VGGVariant.VGG13 or VGGVariant.VGG13_BN =>
        [
            [64, 64],       // Block 1
            [128, 128],     // Block 2
            [256, 256],     // Block 3
            [512, 512],     // Block 4
            [512, 512]      // Block 5
        ],
        VGGVariant.VGG16 or VGGVariant.VGG16_BN =>
        [
            [64, 64],           // Block 1
            [128, 128],         // Block 2
            [256, 256, 256],    // Block 3
            [512, 512, 512],    // Block 4
            [512, 512, 512]     // Block 5
        ],
        VGGVariant.VGG19 or VGGVariant.VGG19_BN =>
        [
            [64, 64],               // Block 1
            [128, 128],             // Block 2
            [256, 256, 256, 256],   // Block 3
            [512, 512, 512, 512],   // Block 4
            [512, 512, 512, 512]    // Block 5
        ],
        _ => throw new NotSupportedException($"VGG variant {Variant} is not supported.")
    };

    /// <summary>
    /// Initializes a new instance of the <see cref="VGGConfiguration"/> class.
    /// </summary>
    /// <param name="variant">The VGG variant to use.</param>
    /// <param name="numClasses">The number of output classes for classification.</param>
    /// <param name="inputHeight">The height of input images (default: 224).</param>
    /// <param name="inputWidth">The width of input images (default: 224).</param>
    /// <param name="inputChannels">The number of input channels (default: 3 for RGB).</param>
    /// <param name="dropoutRate">The dropout rate for FC layers (default: 0.5).</param>
    /// <param name="includeClassifier">Whether to include the classifier layers (default: true).</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when numClasses is less than or equal to 0, or when image dimensions are invalid.
    /// </exception>
    public VGGConfiguration(
        VGGVariant variant,
        int numClasses,
        int inputHeight = 224,
        int inputWidth = 224,
        int inputChannels = 3,
        double dropoutRate = 0.5,
        bool includeClassifier = true)
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

        if (dropoutRate < 0.0 || dropoutRate >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(dropoutRate),
                "Dropout rate must be between 0.0 (inclusive) and 1.0 (exclusive).");
        }

        // Validate minimum input size for VGG (needs at least 32x32 after 5 pooling layers)
        if (inputHeight < 32 || inputWidth < 32)
        {
            throw new ArgumentOutOfRangeException(nameof(inputHeight),
                "VGG requires input dimensions of at least 32x32 pixels.");
        }

        Variant = variant;
        NumClasses = numClasses;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        InputChannels = inputChannels;
        DropoutRate = dropoutRate;
        IncludeClassifier = includeClassifier;

        // Determine if batch normalization should be used based on variant
        UseBatchNormalization = variant is VGGVariant.VGG11_BN or VGGVariant.VGG13_BN
            or VGGVariant.VGG16_BN or VGGVariant.VGG19_BN;
    }

    /// <summary>
    /// Creates a new configuration for VGG16 with batch normalization.
    /// </summary>
    /// <param name="numClasses">The number of output classes.</param>
    /// <returns>A VGGConfiguration for VGG16_BN.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the recommended VGG configuration for most use cases.
    /// VGG16 with batch normalization provides a good balance of accuracy and training stability.
    /// </para>
    /// </remarks>
    public static VGGConfiguration CreateVGG16BN(int numClasses)
    {
        return new VGGConfiguration(VGGVariant.VGG16_BN, numClasses);
    }

    /// <summary>
    /// Creates a configuration optimized for CIFAR-10/CIFAR-100 datasets.
    /// </summary>
    /// <param name="variant">The VGG variant to use.</param>
    /// <param name="numClasses">The number of output classes (10 for CIFAR-10, 100 for CIFAR-100).</param>
    /// <returns>A VGGConfiguration optimized for CIFAR datasets.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CIFAR images are 32x32 pixels, which is smaller than ImageNet's 224x224.
    /// This configuration adjusts the input size accordingly while maintaining the VGG architecture.
    /// </para>
    /// </remarks>
    public static VGGConfiguration CreateForCIFAR(VGGVariant variant, int numClasses)
    {
        return new VGGConfiguration(variant, numClasses, inputHeight: 32, inputWidth: 32);
    }
}
