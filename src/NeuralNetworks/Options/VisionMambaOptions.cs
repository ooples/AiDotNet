using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the image-classifying VisionMambaModel.
/// </summary>
public class VisionMambaOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VisionMambaOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public VisionMambaOptions()
    {
        ImageHeight = 224;
        ImageWidth = 224;
        PatchSize = 16;
        Channels = 3;
        ModelDimension = 192;
        NumLayers = 24;
        NumClasses = 10;
        StateDimension = 16;
    }


    /// <summary>
    /// Gets or sets the image height.
    /// </summary>
    public int ImageHeight { get; set; }

    /// <summary>
    /// Gets or sets the image width.
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Gets or sets the patch size.
    /// </summary>
    public int PatchSize { get; set; }

    /// <summary>
    /// Gets or sets the channels.
    /// </summary>
    public int Channels { get; set; }

    /// <summary>
    /// Gets or sets the num classes.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Gets or sets the patch stride; unset means the patch size (non-overlapping patches, the standard Vim models).
    /// The overlapping "stride 8" models use half the patch size.
    /// </summary>
    public int? Stride { get; set; }

    /// <summary>The stride in effect: <see cref="Stride"/>, or the patch size when it is unset.</summary>
    internal int EffectiveStride => Stride ?? PatchSize;

    /// <summary>Gets or sets the stochastic-depth rate (the official default). Default: <c>0.1</c>.</summary>
    public double DropPathRate { get; set; } = 0.1;

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        // This is an image classifier, not a tokenizer-backed language model. The inherited
        // vocabulary/context settings are deliberately irrelevant to its construction.
        Require(PatchSize, nameof(PatchSize));
        if (Stride is { } stride) Require(stride, nameof(Stride));
        ValidateImageDimension(ImageHeight, nameof(ImageHeight));
        ValidateImageDimension(ImageWidth, nameof(ImageWidth));
        Require(Channels, nameof(Channels));
        Require(ModelDimension, nameof(ModelDimension));
        Require(NumLayers, nameof(NumLayers));
        Require(NumClasses, nameof(NumClasses));
        Require(StateDimension, nameof(StateDimension));
    }

    private void ValidateImageDimension(int dimension, string propertyName)
    {
        Require(dimension, propertyName);
        if (dimension < PatchSize || (dimension - PatchSize) % EffectiveStride != 0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.{propertyName} ({dimension}) must hold a whole number of " +
                $"{GetType().Name}.{nameof(PatchSize)} ({PatchSize}) patches at {GetType().Name}.{nameof(Stride)} ({EffectiveStride}).",
                OptionsParameterName);
        }
    }
}
