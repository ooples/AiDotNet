using System.IO;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Base class for backbone networks used in object detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Detection backbones extend <see cref="NeuralNetworkBase{T}"/> so they satisfy the
/// <see cref="AiDotNet.Interfaces.INeuralNetworkModel{T}"/> contract that the test
/// auto-generator and other consumers expect, while also implementing
/// <see cref="AiDotNet.Interfaces.IFeatureMapProvider{T}"/> for multi-scale feature
/// extraction needed by detection heads (FPN, anchor generators, DETR transformer).
/// </para>
/// <para><b>For Beginners:</b> A backbone network is the first part of a detection model.
/// It takes an input image and extracts meaningful features at multiple scales.
/// Think of it as the "eyes" of the detector that learns to recognize patterns like
/// edges, textures, and shapes.</para>
/// </remarks>
public abstract class BackboneBase<T> : NeuralNetworkBase<T>, AiDotNet.Interfaces.IDetectionBackbone<T>
{
    /// <summary>
    /// Whether the backbone is in training mode (separate from base IsTrainingMode for backwards compat).
    /// </summary>
    protected new bool IsTrainingMode;

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
    public abstract int[] OutputChannels { get; }

    /// <summary>
    /// The stride (downsampling factor) at each feature level.
    /// </summary>
    public abstract int[] Strides { get; }

    /// <summary>
    /// Creates a new backbone with a default dynamic-spatial architecture.
    /// </summary>
    protected BackboneBase()
        : base(NeuralNetworkArchitecture<T>.CreateDynamicSpatial(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.ImageClassification,
                channels: 3,
                outputSize: 1),
              new MeanSquaredErrorLoss<T>())
    {
        IsTrainingMode = false;
        IsFrozen = false;
    }

    /// <summary>
    /// Extracts multi-scale features from an input image tensor.
    /// </summary>
    /// <param name="input">Input image tensor with shape [batch, channels, height, width].</param>
    /// <returns>List of feature maps at different scales, from highest to lowest resolution.</returns>
    public abstract List<Tensor<T>> ExtractFeatures(Tensor<T> input);

    /// <inheritdoc />
    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <summary>
    /// Sets whether the backbone is in training mode.
    /// </summary>
    public override void SetTrainingMode(bool training)
    {
        IsTrainingMode = training;
        base.SetTrainingMode(training);
    }

    /// <summary>
    /// Freezes the backbone weights so they are not updated during training.
    /// </summary>
    public virtual void Freeze() => IsFrozen = true;

    /// <summary>
    /// Unfreezes the backbone weights for training.
    /// </summary>
    public virtual void Unfreeze() => IsFrozen = false;

    /// <summary>
    /// Returns the total number of parameters owned by the backbone, computed by
    /// summing across the backbone's internal Conv2D/Dense/MultiHeadSelfAttention
    /// wrappers. Concrete backbones implement this. The inherited
    /// <see cref="NeuralNetworkBase{T}.ParameterCount"/> property and
    /// <see cref="NeuralNetworkBase{T}.GetParameterCount"/> method are overridden
    /// below to dispatch to this value, so polymorphic callers via
    /// <see cref="INeuralNetworkModel{T}"/> see the correct count.
    /// </summary>
    public abstract long GetBackboneParameterCount();

    /// <inheritdoc />
    public override int ParameterCount
    {
        get
        {
            long count = GetBackboneParameterCount();
            // ParameterCount on the inherited contract is int; clamp to int.MaxValue
            // to avoid overflow on extremely large backbones.
            return count > int.MaxValue ? int.MaxValue : (int)count;
        }
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public abstract void WriteParameters(BinaryWriter writer);

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public abstract void ReadParameters(BinaryReader reader);

    /// <summary>
    /// Gets the expected input size for this backbone.
    /// </summary>
    public virtual (int Height, int Width) GetExpectedInputSize() => (640, 640);

    /// <summary>
    /// Validates that the input tensor has the correct shape.
    /// </summary>
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

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var features = ExtractFeatures(input);
        if (features.Count == 0)
        {
            throw new InvalidOperationException(
                $"{GetType().Name}.ExtractFeatures returned no feature maps for input shape [{string.Join(",", input.Shape)}]. " +
                $"A backbone must produce at least one feature map.");
        }
        return features[features.Count - 1];
    }

    /// <summary>
    /// Backbones intentionally do not populate the inherited <c>Layers</c> list.
    /// Each derived backbone (ResNet, CSPDarknet, EfficientNet, SwinTransformer) constructs
    /// its own internal Conv2D / Dense / MultiHeadSelfAttention wrappers in its constructor
    /// and orchestrates them directly inside <see cref="ExtractFeatures"/>. Parameter
    /// serialization is handled by <see cref="WriteParameters"/> / <see cref="ReadParameters"/>
    /// rather than the layer-list machinery, so this hook is a documented no-op.
    /// </summary>
    protected override void InitializeLayers()
    {
        // Intentional no-op — see XML doc above.
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        WriteParameters(writer);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        ReadParameters(reader);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Concrete backbones (ResNet, CSPDarknet, EfficientNet, SwinTransformer) MUST
    /// implement this by constructing a brand-new instance with their configured
    /// variant/channel-multiplier parameters — <see cref="object.MemberwiseClone"/>
    /// would alias the internal Conv2D / Dense / MultiHeadSelfAttention wrappers and
    /// every nested tensor, so any framework path that deserializes into the returned
    /// instance would also mutate the original model.
    /// </remarks>
    protected abstract override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance();

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = Name,
        AdditionalInfo = new Dictionary<string, object>
        {
            ["BackboneName"] = Name,
            ["OutputChannels"] = OutputChannels,
            ["Strides"] = Strides
        }
    };

    /// <summary>
    /// Detection backbones are not standalone-trainable: they are trained as part of a
    /// parent object/text detector (FasterRCNN, YOLOv8, DETR, …) which orchestrates the
    /// joint forward/backward pass across backbone, neck, and head. Calling <c>Train</c>
    /// directly on a backbone is almost always a programming error, so we fail fast.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: detection backbones are trained as part of a parent " +
            "detector (e.g. FasterRCNN, YOLOv8, DETR) and do not support standalone Train(). " +
            "Train the parent detection model instead.");
    }

    /// <summary>
    /// Backbone parameters live inside per-stage Conv2D/Dense/MultiHeadSelfAttention
    /// wrappers and are serialized via <see cref="WriteParameters"/>/<see cref="ReadParameters"/>.
    /// The flat-vector contract is not the right shape for backbone parameters, so
    /// callers should round-trip through binary streams instead.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not expose a flat parameter vector. " +
            "Use WriteParameters(BinaryWriter) / ReadParameters(BinaryReader) to round-trip " +
            "weights, or train the backbone as part of a parent detection model.");
    }

    /// <summary>
    /// See <see cref="GetParameters"/> for why a flat-vector setter is unsupported on backbones.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not accept a flat parameter vector. " +
            "Use ReadParameters(BinaryReader) to load saved weights.");
    }

    /// <summary>
    /// See <see cref="GetParameters"/> for why a flat-vector update is unsupported on backbones.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not accept a flat parameter update vector. " +
            "Update happens inside the parent detector's optimizer step.");
    }

    /// <summary>
    /// See <see cref="GetParameters"/>. Backbones do not consume a flat parameter vector,
    /// so <c>WithParameters</c> would degrade to a confusing partial-clone. Round-trip
    /// weights through <see cref="WriteParameters"/> / <see cref="ReadParameters"/> instead.
    /// </summary>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        throw new NotSupportedException(
            $"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones. " +
            "Use ReadParameters(BinaryReader) on a fresh instance.");
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
        => (BackboneBase<T>)MemberwiseClone();

    #endregion
}

/// <summary>
/// Feature map output from a backbone network.
/// </summary>
public class FeatureMap<T>
{
    public Tensor<T> Features { get; set; }
    public int Stride { get; set; }
    public int Level { get; set; }

    public FeatureMap(Tensor<T> features, int stride, int level)
    {
        Features = features;
        Stride = stride;
        Level = level;
    }

    public (int Height, int Width) SpatialSize => (Features.Shape[2], Features.Shape[3]);
    public int Channels => Features.Shape[1];
}

/// <summary>
/// Configuration for backbone networks.
/// </summary>
public class BackboneConfig
{
    public bool UsePretrained { get; set; } = true;
    public bool Freeze { get; set; } = false;
    public int[] OutputLevels { get; set; } = new[] { 2, 3, 4 };
    public string? WeightsUrl { get; set; }
    public int? RandomSeed { get; set; } = 42;
}
