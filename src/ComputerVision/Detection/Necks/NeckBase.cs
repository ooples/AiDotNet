using System.IO;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Necks;

/// <summary>
/// Base class for neck modules that perform multi-scale feature fusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The "neck" sits between the backbone and the detection head.
/// It takes multi-scale features from the backbone and fuses them together so that
/// each feature level contains information from both higher and lower resolutions.
/// This helps detect objects of various sizes more accurately.</para>
///
/// <para>Common neck architectures:
/// - FPN (Feature Pyramid Network): Top-down feature fusion
/// - PANet (Path Aggregation Network): Top-down + bottom-up paths
/// - BiFPN (Bidirectional FPN): Weighted bidirectional fusion
/// </para>
/// </remarks>
public abstract class NeckBase<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Whether the neck is in training mode.
    /// </summary>
    protected bool IsTrainingMode;

    /// <summary>
    /// Name of this neck architecture.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Number of output channels for all feature levels.
    /// </summary>
    /// <remarks>
    /// <para>Necks typically project all feature levels to the same number of channels
    /// (e.g., 256) to simplify the detection head.</para>
    /// </remarks>
    public abstract int OutputChannels { get; }

    /// <summary>
    /// Number of feature levels output by the neck.
    /// </summary>
    public abstract int NumLevels { get; }

    /// <summary>
    /// Creates a new neck module.
    /// </summary>
    protected NeckBase()
    {
        NumOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        IsTrainingMode = false;
    }

    /// <summary>
    /// Performs multi-scale feature fusion.
    /// </summary>
    /// <param name="features">List of feature maps from the backbone, ordered from highest to lowest resolution.</param>
    /// <returns>Fused feature maps at multiple scales.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes the raw features from the backbone
    /// and combines them across scales. After fusion, each feature level "knows about"
    /// features from other scales, making detection more accurate.</para>
    /// </remarks>
    public abstract List<Tensor<T>> Forward(List<Tensor<T>> features);

    /// <summary>
    /// Sets whether the neck is in training mode.
    /// </summary>
    /// <param name="training">True for training, false for inference.</param>
    public virtual void SetTrainingMode(bool training)
    {
        IsTrainingMode = training;
    }

    /// <summary>
    /// Gets the total number of parameters in the neck.
    /// </summary>
    /// <returns>Number of trainable parameters.</returns>
    public abstract long GetParameterCount();

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    public abstract void WriteParameters(BinaryWriter writer);

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public abstract void ReadParameters(BinaryReader reader);

    /// <summary>
    /// Validates that the input features are compatible with this neck.
    /// </summary>
    /// <param name="features">Features to validate.</param>
    /// <param name="expectedInputChannels">Expected input channels at each level.</param>
    /// <exception cref="ArgumentException">Thrown if features are incompatible.</exception>
    protected void ValidateFeatures(List<Tensor<T>> features, int[] expectedInputChannels)
    {
        if (features.Count != expectedInputChannels.Length)
        {
            throw new ArgumentException(
                $"Expected {expectedInputChannels.Length} feature levels, got {features.Count}",
                nameof(features));
        }

        for (int i = 0; i < features.Count; i++)
        {
            if (features[i].Rank != 4)
            {
                throw new ArgumentException(
                    $"Feature level {i}: Expected 4D tensor, got {features[i].Rank}D",
                    nameof(features));
            }

            if (features[i].Shape[1] != expectedInputChannels[i])
            {
                throw new ArgumentException(
                    $"Feature level {i}: Expected {expectedInputChannels[i]} channels, got {features[i].Shape[1]}",
                    nameof(features));
            }
        }
    }

    /// <summary>
    /// Upsample a feature map by a factor of 2 using nearest neighbor interpolation.
    /// </summary>
    /// <param name="input">Input feature map.</param>
    /// <returns>Upsampled feature map.</returns>
    protected Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>(new[] { batch, channels, height * 2, width * 2 });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height * 2; h++)
                {
                    for (int w = 0; w < width * 2; w++)
                    {
                        int srcH = h / 2;
                        int srcW = w / 2;
                        output[b, c, h, w] = input[b, c, srcH, srcW];
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Downsample a feature map by a factor of 2 using max pooling.
    /// </summary>
    /// <param name="input">Input feature map.</param>
    /// <returns>Downsampled feature map.</returns>
    protected Tensor<T> Downsample2x(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        int outHeight = height / 2;
        int outWidth = width / 2;

        var output = new Tensor<T>(new[] { batch, channels, outHeight, outWidth });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outHeight; h++)
                {
                    for (int w = 0; w < outWidth; w++)
                    {
                        // Max pooling 2x2
                        T maxVal = input[b, c, h * 2, w * 2];
                        T val1 = input[b, c, h * 2, w * 2 + 1];
                        T val2 = input[b, c, h * 2 + 1, w * 2];
                        T val3 = input[b, c, h * 2 + 1, w * 2 + 1];

                        if (NumOps.GreaterThan(val1, maxVal)) maxVal = val1;
                        if (NumOps.GreaterThan(val2, maxVal)) maxVal = val2;
                        if (NumOps.GreaterThan(val3, maxVal)) maxVal = val3;

                        output[b, c, h, w] = maxVal;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Applies a 1x1 convolution to change the number of channels.
    /// </summary>
    /// <param name="input">Input feature map.</param>
    /// <param name="weights">Convolution weights [out_channels, in_channels].</param>
    /// <param name="bias">Optional bias [out_channels].</param>
    /// <returns>Feature map with new channel count.</returns>
    protected Tensor<T> Conv1x1(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias = null)
    {
        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];
        int outChannels = weights.Shape[0];

        var output = new Tensor<T>(new[] { batch, outChannels, height, width });

        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        T sum = NumOps.Zero;
                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(input[b, ic, h, w], weights[oc, ic]));
                        }
                        if (bias is not null)
                        {
                            sum = NumOps.Add(sum, bias[oc]);
                        }
                        output[b, oc, h, w] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Adds two feature maps element-wise.
    /// </summary>
    /// <param name="a">First feature map.</param>
    /// <param name="b">Second feature map.</param>
    /// <returns>Element-wise sum.</returns>
    protected Tensor<T> Add(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException("Feature maps must have the same shape for addition");
        }

        var output = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            output[i] = NumOps.Add(a[i], b[i]);
        }
        return output;
    }
}

/// <summary>
/// Configuration for neck modules.
/// </summary>
public class NeckConfig
{
    /// <summary>
    /// Number of output channels for all feature levels.
    /// </summary>
    public int OutputChannels { get; set; } = 256;

    /// <summary>
    /// Input channels from the backbone at each level.
    /// </summary>
    public int[] InputChannels { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Number of feature pyramid levels.
    /// </summary>
    public int NumLevels { get; set; } = 3;

    /// <summary>
    /// Whether to add extra convolution layers for feature refinement.
    /// </summary>
    public bool UseExtraConvs { get; set; } = true;

    /// <summary>
    /// Activation function to use (e.g., "relu", "silu", "gelu").
    /// </summary>
    public string Activation { get; set; } = "relu";

    /// <summary>
    /// Whether to use batch normalization.
    /// </summary>
    public bool UseBatchNorm { get; set; } = true;
}
