using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Base class for backbone networks used in object detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A backbone network is the first part of a detection model.
/// It takes an input image and extracts meaningful features at multiple scales.
/// Think of it as the "eyes" of the detector that learns to recognize patterns like
/// edges, textures, and shapes.</para>
///
/// <para>Common backbones include:
/// - ResNet: Residual networks with skip connections
/// - CSPDarknet: Used in YOLO models, efficient for real-time detection
/// - Swin Transformer: Vision transformer with shifted windows
/// - EfficientNet: Scalable and efficient convolutional network
/// </para>
/// </remarks>
public abstract class BackboneBase<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Whether the backbone is in training mode.
    /// </summary>
    protected bool IsTrainingMode;

    /// <summary>
    /// Whether the backbone weights are frozen (not updated during training).
    /// </summary>
    public bool IsFrozen { get; protected set; }

    /// <summary>
    /// Name of this backbone architecture.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Number of output channels for each feature level.
    /// </summary>
    /// <remarks>
    /// <para>Modern detectors use multi-scale features. This array contains the
    /// number of channels at each scale, typically from high resolution (small objects)
    /// to low resolution (large objects).</para>
    /// </remarks>
    public abstract int[] OutputChannels { get; }

    /// <summary>
    /// The stride (downsampling factor) at each feature level.
    /// </summary>
    /// <remarks>
    /// <para>A stride of 8 means the feature map is 1/8 the size of the input.
    /// Common strides are [8, 16, 32] for 3-level feature pyramids.</para>
    /// </remarks>
    public abstract int[] Strides { get; }

    /// <summary>
    /// Creates a new backbone with numeric operations for type T.
    /// </summary>
    protected BackboneBase()
    {
        NumOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        IsTrainingMode = false;
        IsFrozen = false;
    }

    /// <summary>
    /// Extracts multi-scale features from an input image tensor.
    /// </summary>
    /// <param name="input">Input image tensor with shape [batch, channels, height, width].</param>
    /// <returns>List of feature maps at different scales, from highest to lowest resolution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method runs the input image through the backbone
    /// and returns feature maps at multiple scales. Small objects need high-resolution
    /// features, while large objects are detected in low-resolution features.</para>
    /// </remarks>
    public abstract List<Tensor<T>> ExtractFeatures(Tensor<T> input);

    /// <summary>
    /// Sets whether the backbone is in training mode.
    /// </summary>
    /// <param name="training">True for training, false for inference.</param>
    public virtual void SetTrainingMode(bool training)
    {
        IsTrainingMode = training;
    }

    /// <summary>
    /// Freezes the backbone weights so they are not updated during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When fine-tuning a pre-trained model on a small dataset,
    /// it's often beneficial to freeze the backbone and only train the detection head.
    /// This prevents the backbone from "forgetting" its pre-trained features.</para>
    /// </remarks>
    public virtual void Freeze()
    {
        IsFrozen = true;
    }

    /// <summary>
    /// Unfreezes the backbone weights for training.
    /// </summary>
    public virtual void Unfreeze()
    {
        IsFrozen = false;
    }

    /// <summary>
    /// Gets the total number of parameters in the backbone.
    /// </summary>
    /// <returns>Number of trainable parameters.</returns>
    public abstract long GetParameterCount();

    /// <summary>
    /// Gets the expected input size for this backbone.
    /// </summary>
    /// <returns>Tuple of (height, width) for optimal input size.</returns>
    public virtual (int Height, int Width) GetExpectedInputSize()
    {
        return (640, 640); // Default size for most detection models
    }

    /// <summary>
    /// Validates that the input tensor has the correct shape.
    /// </summary>
    /// <param name="input">Input tensor to validate.</param>
    /// <exception cref="ArgumentException">Thrown if the input shape is invalid.</exception>
    protected void ValidateInput(Tensor<T> input)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(
                $"Expected 4D input tensor [batch, channels, height, width], got {input.Rank}D",
                nameof(input));
        }

        if (input.Shape[1] != 3)
        {
            throw new ArgumentException(
                $"Expected 3 input channels (RGB), got {input.Shape[1]}",
                nameof(input));
        }
    }
}

/// <summary>
/// Feature map output from a backbone network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FeatureMap<T>
{
    /// <summary>
    /// The feature tensor with shape [batch, channels, height, width].
    /// </summary>
    public Tensor<T> Features { get; set; }

    /// <summary>
    /// The stride (downsampling factor) of this feature map.
    /// </summary>
    public int Stride { get; set; }

    /// <summary>
    /// The level index in the feature pyramid (0 = highest resolution).
    /// </summary>
    public int Level { get; set; }

    /// <summary>
    /// Creates a new feature map.
    /// </summary>
    /// <param name="features">The feature tensor.</param>
    /// <param name="stride">The downsampling stride.</param>
    /// <param name="level">The pyramid level index.</param>
    public FeatureMap(Tensor<T> features, int stride, int level)
    {
        Features = features;
        Stride = stride;
        Level = level;
    }

    /// <summary>
    /// Gets the spatial dimensions of the feature map.
    /// </summary>
    public (int Height, int Width) SpatialSize => (Features.Shape[2], Features.Shape[3]);

    /// <summary>
    /// Gets the number of channels in the feature map.
    /// </summary>
    public int Channels => Features.Shape[1];
}

/// <summary>
/// Configuration for backbone networks.
/// </summary>
public class BackboneConfig
{
    /// <summary>
    /// Whether to use pre-trained weights.
    /// </summary>
    public bool UsePretrained { get; set; } = true;

    /// <summary>
    /// Whether to freeze the backbone during training.
    /// </summary>
    public bool Freeze { get; set; } = false;

    /// <summary>
    /// Which feature levels to output (e.g., [2, 3, 4] for P3, P4, P5).
    /// </summary>
    public int[] OutputLevels { get; set; } = new[] { 2, 3, 4 };

    /// <summary>
    /// URL to download pre-trained weights from.
    /// </summary>
    public string? WeightsUrl { get; set; }

    /// <summary>
    /// Random seed for weight initialization.
    /// </summary>
    public int? RandomSeed { get; set; } = 42;
}
