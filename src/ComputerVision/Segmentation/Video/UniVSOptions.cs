using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// Configuration options for UniVS universal video segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UniVS model. Default values follow the original paper settings.</para>
/// </remarks>
public class UniVSOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public UniVSOptions()
    {
        NumClasses = 80;
        DropRate = 0.1;
        ModelSize = UniVSModelSize.R50;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UniVSOptions(UniVSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        NumPromptPoints = other.NumPromptPoints;
        PromptMemoryLength = other.PromptMemoryLength;
    }

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>UniVSModelSize.R50</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Most of these models were published in several sizes that
    /// trade accuracy against speed and memory. Picking a variant selects the widths and
    /// depths the paper reports for it; it is not a hint, it changes the network that gets
    /// built.
    /// </para>
    /// <para>
    /// Declared here rather than on <see cref="SegmentationModelOptions"/> because each
    /// model names its own variants with its own enum, so there is no shared type to
    /// declare.
    /// </para>
    /// </remarks>
    public UniVSModelSize ModelSize { get; set; }

    /// <summary>
    /// Gets or sets how many points the visual sampler takes from each visual prompt (a mask) to make its prompt
    /// tokens. Default: 32 (Li et al. 2024: "We sample 32 points per visual prompt").
    /// </summary>
    public int NumPromptPoints { get; set; } = 32;

    /// <summary>
    /// Gets or sets how many visual prompts (the first one plus the most recent predictions) each object's prompt
    /// memory pool keeps. The paper does not state a pool size; the default, 5, is its stage-3 clip length.
    /// </summary>
    public int PromptMemoryLength { get; set; } = 5;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
