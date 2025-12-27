namespace AiDotNet.Enums;

/// <summary>
/// Defines the available VGG network architecture variants.
/// </summary>
/// <remarks>
/// <para>
/// VGG networks are a family of deep convolutional neural networks developed by the Visual Geometry Group
/// at Oxford University. They are characterized by their use of small (3x3) convolution filters stacked
/// in increasing depth, which allows them to learn complex features while keeping the number of parameters
/// manageable.
/// </para>
/// <para>
/// <b>For Beginners:</b> VGG networks are named after the Visual Geometry Group that created them.
/// The number in the name (e.g., VGG16) refers to the total number of weight layers in the network.
/// For example, VGG16 has 13 convolutional layers and 3 fully connected layers, totaling 16 weight layers.
/// These networks were groundbreaking because they showed that network depth is critical for good performance.
/// Despite being older architectures, they remain popular for transfer learning and as baselines.
/// </para>
/// <para>
/// <b>Batch Normalization Variants:</b> The "_BN" suffix indicates variants that include batch normalization
/// layers after each convolutional layer. Batch normalization helps stabilize training and often allows
/// for faster convergence and better final accuracy.
/// </para>
/// </remarks>
public enum VGGVariant
{
    /// <summary>
    /// VGG-11: 11 weight layers (8 conv + 3 FC). The smallest VGG variant.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: [64] - [128] - [256, 256] - [512, 512] - [512, 512] - FC - FC - FC
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> VGG11 is the lightest VGG variant with approximately 133 million parameters.
    /// It's a good choice when you have limited computational resources or when training from scratch
    /// on smaller datasets. The brackets show the number of filters in each convolutional block,
    /// separated by max pooling layers.
    /// </para>
    /// </remarks>
    VGG11,

    /// <summary>
    /// VGG-11 with batch normalization after each convolutional layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is VGG11 with batch normalization added. Batch normalization
    /// normalizes the inputs to each layer, which helps the network train faster and more stably.
    /// This variant typically achieves better accuracy than the original VGG11.
    /// </para>
    /// </remarks>
    VGG11_BN,

    /// <summary>
    /// VGG-13: 13 weight layers (10 conv + 3 FC).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: [64, 64] - [128, 128] - [256, 256] - [512, 512] - [512, 512] - FC - FC - FC
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> VGG13 adds one more convolutional layer to each of the first two blocks
    /// compared to VGG11. This gives it more capacity to learn features at different scales.
    /// It has approximately 133 million parameters (similar to VGG11 due to the small conv layers).
    /// </para>
    /// </remarks>
    VGG13,

    /// <summary>
    /// VGG-13 with batch normalization after each convolutional layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VGG13 with batch normalization for improved training stability and accuracy.
    /// </para>
    /// </remarks>
    VGG13_BN,

    /// <summary>
    /// VGG-16: 16 weight layers (13 conv + 3 FC). The most commonly used VGG variant.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: [64, 64] - [128, 128] - [256, 256, 256] - [512, 512, 512] - [512, 512, 512] - FC - FC - FC
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> VGG16 is the most popular VGG variant and is often used as a baseline
    /// for image classification tasks. It has approximately 138 million parameters. The deeper
    /// architecture allows it to learn more complex features, making it suitable for a wide range
    /// of image recognition tasks. It's commonly used for transfer learning where you take a
    /// pre-trained VGG16 and fine-tune it for your specific task.
    /// </para>
    /// </remarks>
    VGG16,

    /// <summary>
    /// VGG-16 with batch normalization after each convolutional layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VGG16 with batch normalization. This is often the recommended VGG variant
    /// for new projects because it combines the proven architecture of VGG16 with the training benefits
    /// of batch normalization. It typically achieves 1-2% better accuracy than the original VGG16.
    /// </para>
    /// </remarks>
    VGG16_BN,

    /// <summary>
    /// VGG-19: 19 weight layers (16 conv + 3 FC). The deepest VGG variant.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: [64, 64] - [128, 128] - [256, 256, 256, 256] - [512, 512, 512, 512] - [512, 512, 512, 512] - FC - FC - FC
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> VGG19 is the deepest VGG variant with approximately 144 million parameters.
    /// It adds one more convolutional layer to each of the last three blocks compared to VGG16.
    /// While it can learn slightly more complex features, the additional depth provides diminishing
    /// returns and significantly increases training time and memory requirements. VGG16 is often
    /// preferred unless you have a specific need for the additional capacity.
    /// </para>
    /// </remarks>
    VGG19,

    /// <summary>
    /// VGG-19 with batch normalization after each convolutional layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VGG19 with batch normalization. Use this when you need maximum model
    /// capacity and have sufficient computational resources. The batch normalization helps mitigate
    /// some of the training difficulties associated with very deep networks.
    /// </para>
    /// </remarks>
    VGG19_BN
}
