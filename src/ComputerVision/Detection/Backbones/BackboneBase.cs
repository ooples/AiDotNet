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
public abstract class BackboneBase<T> : NeuralNetworkBase<T>, AiDotNet.Interfaces.IFeatureMapProvider<T>
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
    /// Gets the total number of parameters in the backbone (subclass-specific count).
    /// </summary>
    public new abstract long GetParameterCount();

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
        return features.Count > 0 ? features[features.Count - 1] : new Tensor<T>(new[] { 1, 0 });
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        // Backbones manage their own layer construction in derived ctors via Conv2D etc.
        // Layers list stays empty; ExtractFeatures drives all forward computation.
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => (BackboneBase<T>)MemberwiseClone();

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

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) { }

    /// <inheritdoc />
    public override Vector<T> GetParameters() => new Vector<T>(0);

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters) { }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters) { }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        InterfaceGuard.Parameterizable(copy).SetParameters(parameters);
        return copy;
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
