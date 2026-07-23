using System.IO;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
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
public abstract class NeckBase<T> : ModelBase<T, Tensor<T>, Tensor<T>>
{
    // NumOps and Engine inherited from ModelBase

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

        // Use ceiling division so a 5x5 input produces a 3x3 output (matching the
        // dynamic-spatial pyramid alignment used elsewhere). Floor division would
        // silently drop the last row/column for odd-sized features and break
        // multi-scale detection heads at non-power-of-two input sizes.
        int outHeight = (height + 1) / 2;
        int outWidth = (width + 1) / 2;

        var output = new Tensor<T>(new[] { batch, channels, outHeight, outWidth });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outHeight; h++)
                {
                    for (int w = 0; w < outWidth; w++)
                    {
                        int srcRow = h * 2;
                        int srcCol = w * 2;
                        // Max pooling 2x2 with bounds-checked sampling: the right/bottom
                        // edge of an odd-sized window covers fewer than 4 source cells,
                        // so we take the max only over the in-bounds entries.
                        T maxVal = input[b, c, srcRow, srcCol];
                        if (srcCol + 1 < width)
                        {
                            T v = input[b, c, srcRow, srcCol + 1];
                            if (NumOps.GreaterThan(v, maxVal)) maxVal = v;
                        }
                        if (srcRow + 1 < height)
                        {
                            T v = input[b, c, srcRow + 1, srcCol];
                            if (NumOps.GreaterThan(v, maxVal)) maxVal = v;
                        }
                        if (srcRow + 1 < height && srcCol + 1 < width)
                        {
                            T v = input[b, c, srcRow + 1, srcCol + 1];
                            if (NumOps.GreaterThan(v, maxVal)) maxVal = v;
                        }

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

        // 1x1 conv = matmul: reshape [B,C_in,H,W] -> [B*H*W, C_in] @ W^T -> [B*H*W, C_out]
        // Transpose input from NCHW to NHWC: [B, C_in, H, W] -> permute to get [B*H*W, C_in]
        int spatialSize = height * width;
        var inputFlat = new Tensor<T>(new[] { batch * spatialSize, inChannels });
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int spatialIdx = b * spatialSize + h * width + w;
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        inputFlat[spatialIdx, ic] = input[b, ic, h, w];
                    }
                }
            }
        }

        // MatMul: [B*H*W, C_in] @ [C_out, C_in]^T = [B*H*W, C_out]
        var weightsT = weights.Transpose(new[] { 1, 0 });
        var outputFlat = Engine.TensorMatMul(inputFlat, weightsT);

        // Add bias if present
        if (bias is not null)
        {
            var biasBroadcast = bias.Reshape(1, outChannels);
            outputFlat = Engine.TensorBroadcastAdd(outputFlat, biasBroadcast);
        }

        // Reshape back to NCHW: [B*H*W, C_out] -> [B, C_out, H, W]
        var output = new Tensor<T>(new[] { batch, outChannels, height, width });
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int spatialIdx = b * spatialSize + h * width + w;
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        output[b, oc, h, w] = outputFlat[spatialIdx, oc];
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
        if (!a._shape.SequenceEqual(b._shape))
        {
            throw new ArgumentException("Feature maps must have the same shape for addition");
        }

        var output = new Tensor<T>(a._shape);
        for (int i = 0; i < a.Length; i++)
        {
            output[i] = NumOps.Add(a[i], b[i]);
        }
        return output;
    }

    #region ModelBase Overrides

    /// <summary>
    /// Single-tensor <c>Predict</c> is not a meaningful operation for a detection neck:
    /// concrete necks (FPN, PANet, BiFPN) operate on the full backbone feature pyramid
    /// (a <see cref="List{Tensor}"/> with one tensor per level) and would fail their own
    /// feature-count validation if handed a single tensor. Use
    /// <see cref="Forward(List{Tensor{T}})"/> directly instead — that is the public API
    /// for running a neck.
    /// </summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: detection necks consume the full backbone feature pyramid, " +
            "not a single tensor. Call Forward(List<Tensor<T>>) with one tensor per level " +
            "instead, or run the parent detection model whose pipeline supplies the pyramid.");
    }

    /// <summary>
    /// Detection necks (FPN, PANet, BiFPN) are not standalone-trainable: they are trained
    /// as part of a parent detector that orchestrates the joint backbone+neck+head pass.
    /// Calling <c>Train</c> directly on a neck is almost always a programming error.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: detection necks are trained as part of a parent detector " +
            "(e.g. FasterRCNN, YOLOv8) and do not support standalone Train(). " +
            "Train the parent detection model instead.");
    }

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Neck parameters live inside per-stage Conv2D wrappers and are serialized via
    /// <c>WriteParameters</c>/<c>ReadParameters</c> on the concrete neck subclass.
    /// The flat-vector contract is not the right shape for neck parameters, so callers
    /// should round-trip through binary streams instead.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        throw new NotSupportedException(
            $"{GetType().Name}: necks do not expose a flat parameter vector. " +
            "Use the concrete neck's WriteParameters(BinaryWriter) / ReadParameters(BinaryReader) " +
            "to round-trip weights, or train as part of a parent detection model.");
    }

    /// <summary>
    /// See <see cref="GetParameters"/>.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: necks do not accept a flat parameter vector. " +
            "Use the concrete neck's ReadParameters(BinaryReader) to load saved weights.");
    }

    /// <summary>
    /// See <see cref="GetParameters"/>.
    /// </summary>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: WithParameters(Vector<T>) is unsupported on necks. " +
            "Use ReadParameters(BinaryReader) on a fresh instance.");
    }

    /// <summary>
    /// Concrete necks are responsible for producing a true deep copy of their internal
    /// Conv2D wrappers and config. Returning <see cref="object.MemberwiseClone"/> here
    /// would silently share tensor references, so we require an explicit override.
    /// </summary>
    public override abstract IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy();

    #endregion
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
