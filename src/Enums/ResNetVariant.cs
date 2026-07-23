namespace AiDotNet.Enums;

/// <summary>
/// Defines the available ResNet (Residual Network) architecture variants.
/// </summary>
/// <remarks>
/// <para>
/// ResNet architectures are a family of deep convolutional neural networks that introduced skip connections
/// (residual connections) to enable training of very deep networks. The key innovation is learning residual
/// functions with reference to the layer inputs, rather than learning unreferenced functions.
/// </para>
/// <para>
/// <b>For Beginners:</b> ResNet networks are named after their total number of weight layers.
/// For example, ResNet50 has 50 convolutional and fully-connected layers. These networks can be
/// much deeper than earlier architectures (like VGG) because the skip connections allow gradients
/// to flow more easily during training, solving the "vanishing gradient" problem.
/// </para>
/// <para>
/// <b>Architecture Types:</b>
/// <list type="bullet">
/// <item><description>ResNet18/34 use "BasicBlock" with two 3x3 convolutions</description></item>
/// <item><description>ResNet50/101/152 use "BottleneckBlock" with 1x1, 3x3, 1x1 convolutions for efficiency</description></item>
/// </list>
/// </para>
/// </remarks>
public enum ResNetVariant
{
    /// <summary>
    /// ResNet-18: 18 weight layers using BasicBlock. Suitable for smaller datasets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: conv1 + [2, 2, 2, 2] BasicBlocks + FC = 18 layers
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ResNet18 is the smallest ResNet variant with approximately 11.7 million parameters.
    /// It's a good choice for smaller datasets, limited computational resources, or when you need faster
    /// inference times. Despite being relatively shallow, it still benefits from residual connections
    /// and typically outperforms VGG networks with similar parameter counts.
    /// </para>
    /// </remarks>
    ResNet18,

    /// <summary>
    /// ResNet-34: 34 weight layers using BasicBlock. Good balance of depth and efficiency.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: conv1 + [3, 4, 6, 3] BasicBlocks + FC = 34 layers
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ResNet34 has approximately 21.8 million parameters and provides a good
    /// balance between model capacity and training efficiency. It uses the same BasicBlock structure
    /// as ResNet18 but with more blocks, allowing it to learn more complex features. Often used when
    /// ResNet18 underfits but computational resources are limited.
    /// </para>
    /// </remarks>
    ResNet34,

    /// <summary>
    /// ResNet-50: 50 weight layers using BottleneckBlock. The most commonly used ResNet variant.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: conv1 + [3, 4, 6, 3] BottleneckBlocks + FC = 50 layers
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ResNet50 is the most popular ResNet variant with approximately 25.6 million
    /// parameters. It switches to BottleneckBlocks (1x1-3x3-1x1 convolutions) which are more parameter
    /// efficient than BasicBlocks. Despite having more layers than ResNet34, the bottleneck design
    /// keeps the parameter count manageable while significantly increasing depth. This is the go-to
    /// architecture for most image classification tasks and transfer learning.
    /// </para>
    /// </remarks>
    ResNet50,

    /// <summary>
    /// ResNet-101: 101 weight layers using BottleneckBlock. For complex tasks requiring more capacity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: conv1 + [3, 4, 23, 3] BottleneckBlocks + FC = 101 layers
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ResNet101 has approximately 44.5 million parameters and provides more
    /// capacity than ResNet50. The additional depth (especially in the third stage with 23 blocks)
    /// allows it to learn more complex patterns. Use this when ResNet50 underfits your data or when
    /// you're working with very complex visual recognition tasks. Requires more GPU memory and
    /// longer training times than ResNet50.
    /// </para>
    /// </remarks>
    ResNet101,

    /// <summary>
    /// ResNet-152: 152 weight layers using BottleneckBlock. Maximum capacity ResNet variant.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Architecture: conv1 + [3, 8, 36, 3] BottleneckBlocks + FC = 152 layers
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ResNet152 is the deepest standard ResNet variant with approximately
    /// 60.2 million parameters. It significantly increases depth in the second and third stages.
    /// This model is typically used for competition-level performance on very large and complex
    /// datasets like ImageNet. It requires substantial computational resources and is often slower
    /// to train and run inference than smaller variants. For most practical applications, ResNet50
    /// or ResNet101 provide better speed/accuracy tradeoffs.
    /// </para>
    /// </remarks>
    ResNet152
}
