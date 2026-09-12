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
public abstract partial class NeckBase<T> : ModelBase<T, Tensor<T>, Tensor<T>>
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

    // GetParameters and SetParameters are NOT overridden here. They used to throw
    // NotSupportedException, saying necks do not expose a flat parameter vector -- which was
    // true only because nothing could SEE their weights: a neck keeps them in bare
    // List<Tensor<T>> fields, not layers. Each concrete neck now declares those lists through
    // RegisterComponents, so ModelBase folds count, vector and restore from one enumeration.
    // WriteParameters/ReadParameters stay as the binary path; they are no longer the ONLY path.
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
    protected Tensor<T> Upsample2x(Tensor<T> input) => CvTensorOps<T>.Upsample2xNearest(input);

    /// <summary>
    /// Downsample a feature map by a factor of 2 using max pooling.
    /// </summary>
    /// <param name="input">Input feature map.</param>
    /// <returns>Downsampled feature map.</returns>
    protected Tensor<T> Downsample2x(Tensor<T> input)
        // 2x2 max pooling in CEIL mode, so a 5x5 input produces a 3x3 output (matching the
        // dynamic-spatial pyramid alignment used elsewhere) and the partial right/bottom window takes
        // its max over the in-bounds cells only.
        => CvTensorOps<T>.MaxPool2x2Ceil(input);

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

        // A 1x1 convolution is a matmul over the channel axis: NCHW -> [B*H*W, C_in] @ W^T -> NCHW.
        // Every reshape and transpose here is an ENGINE op. Tensor<T>.Reshape and .Transpose bypass
        // the autodiff tape, so using them on the input severed the gradient to the backbone, and
        // using them on the WEIGHTS meant the neck's own weights never received a gradient either.
        var inputFlat = Engine.Reshape(
            Engine.TensorPermute(input, new[] { 0, 2, 3, 1 }),
            new[] { batch * height * width, inChannels });

        var outputFlat = Engine.TensorMatMul(inputFlat, Engine.TensorPermute(weights, new[] { 1, 0 }));

        if (bias is not null)
        {
            outputFlat = Engine.TensorAdd(
                outputFlat,
                Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, outChannels }), new[] { batch * height * width, outChannels }));
        }

        return Engine.TensorPermute(
            Engine.Reshape(outputFlat, new[] { batch, height, width, outChannels }),
            new[] { 0, 3, 1, 2 });
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

        // Engine op rather than a scalar loop so the tape records the addition: FPN's top-down
        // pathway adds the upsampled higher level into the lateral one, and a severed add there
        // cuts every level below it out of the gradient.
        return Engine.TensorAdd(a, b);
    }

    #region ModelBase Overrides

    /// <summary>
    /// Single-tensor <c>Predict</c> is not a meaningful operation for a detection neck:
    /// concrete necks (FPN, PANet, BiFPN) operate on the full backbone feature pyramid
    /// (a <see cref="List{Tensor}"/> with one tensor per level) and would fail their own
    /// feature-count validation if handed a single tensor. Use
    /// <see cref="Forward(List{Tensor{T}})"/> directly instead â€” that is the public API
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
    /// See <see cref="GetParameters"/>.
    /// </summary>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: WithParameters(Vector<T>) is unsupported on necks. " +
            "Use ReadParameters(BinaryReader) on a fresh instance.");
    }

    // The re-declaration that used to sit here forced every neck to hand-write DeepCopy, on the
    // reasoning that a memberwise copy would share tensor references. ModelBase does not do a
    // memberwise copy: it rebuilds the neck from its recorded constructor and then reloads state
    // through Serialize/Deserialize, so the tensors are new storage and the reason no longer holds.

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
