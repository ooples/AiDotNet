using AiDotNet.Enums;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the Depth Anything V2 monocular depth model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class carries the settings Depth Anything V2 was published with,
/// so you can use the model without configuring anything. Pick a size if you want a different
/// speed/accuracy trade-off:
/// </para>
/// <code>
/// var options = new DepthAnythingV2Options(DepthAnythingV2ModelSize.Large);
/// </code>
/// <para>
/// Choosing a size sets the feature width and encoder depth the paper pairs with it. You can
/// still override either afterwards; the size is recorded but no longer consulted once set.
/// </para>
/// </remarks>
public class DepthAnythingV2Options : VideoHyperparameterOptions
{
    /// <summary>
    /// Initializes a new instance carrying the Depth Anything V2 Base defaults.
    /// </summary>
    public DepthAnythingV2Options()
        : this(DepthAnythingV2ModelSize.Base)
    {
    }

    /// <summary>
    /// Initializes a new instance carrying the published defaults for the given encoder size.
    /// </summary>
    /// <param name="modelSize">The encoder size variant to take defaults from.</param>
    public DepthAnythingV2Options(DepthAnythingV2ModelSize modelSize)
    {
        ModelSize = modelSize;

        // The DINOv2 encoder each size is built on, from the Depth Anything V2 paper.
        NumFeatures = modelSize switch
        {
            DepthAnythingV2ModelSize.Small => 384,
            DepthAnythingV2ModelSize.Large => 1024,
            _ => 768,
        };

        NumEncoderBlocks = modelSize switch
        {
            DepthAnythingV2ModelSize.Large => 24,
            _ => 12,
        };

        // Carried over from the model, which hardcoded 16. Depth Anything V2 builds on DINOv2,
        // whose patch size is 14, so this is likely wrong — but correcting it changes the
        // model's shape, which is the paper-fidelity phase's job, not the migration's.
        PatchSize = 16;
    }

    /// <summary>
    /// Initializes a new instance by copying another options instance.
    /// </summary>
    /// <param name="other">The instance to copy.</param>
    public DepthAnythingV2Options(DepthAnythingV2Options other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        ModelSize = other.ModelSize;
        NumFeatures = other.NumFeatures;
        NumEncoderBlocks = other.NumEncoderBlocks;
        PatchSize = other.PatchSize;
    }

    /// <summary>
    /// Gets or sets the encoder size variant. Default: <see cref="DepthAnythingV2ModelSize.Base"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Records which size the other defaults came from. Set the size
    /// through the constructor rather than here — assigning it afterwards does not recompute
    /// <see cref="VideoHyperparameterOptions.NumFeatures"/> or <see cref="NumEncoderBlocks"/>.
    /// </para>
    /// </remarks>
    public DepthAnythingV2ModelSize ModelSize { get; set; }

    /// <summary>
    /// Gets or sets the number of DINOv2 transformer encoder blocks. Default: 12 (Base).
    /// </summary>
    public int NumEncoderBlocks { get; set; }

    /// <summary>
    /// Gets or sets the side length in pixels of each image patch fed to the encoder.
    /// Default: 16.
    /// </summary>
    public int PatchSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        Require(NumFeatures, nameof(NumFeatures));
        Require(NumEncoderBlocks, nameof(NumEncoderBlocks));
        Require(PatchSize, nameof(PatchSize));
    }
}
