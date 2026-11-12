using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// Configuration options for DEVA decoupled video segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DEVA model. Default values follow the original paper settings.</para>
/// </remarks>
public class DEVAOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DEVAOptions()
    {
        NumClasses = 1;
        DropRate = 0;
        ModelSize = DEVAModelSize.Base;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DEVAOptions(DEVAOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        ChannelDimensions = other.ChannelDimensions?.ToArray();
        StageDepths = other.StageDepths?.ToArray();
        DecoderDimension = other.DecoderDimension;
        UseGroupNormalization = other.UseGroupNormalization;
        MemoryTopK = other.MemoryTopK;
        MergeEvery = other.MergeEvery;
        MaxMemoryFrames = other.MaxMemoryFrames;
        AssociationIoUThreshold = other.AssociationIoUThreshold;
        MaxMissedMerges = other.MaxMissedMerges;
    }

    /// <summary>
    /// Gets or sets the four encoder-stage channel dimensions. A null value uses
    /// the selected <see cref="AiDotNet.Enums.DEVAModelSize"/> research configuration.
    /// </summary>
    public int[]? ChannelDimensions { get; set; }

    /// <summary>
    /// Gets or sets the number of blocks in each of the four encoder stages. A
    /// null value uses the selected model-size research configuration.
    /// </summary>
    public int[]? StageDepths { get; set; }

    /// <summary>
    /// Gets or sets the decoder feature dimension. A null value uses the selected
    /// model-size research configuration.
    /// </summary>
    public int? DecoderDimension { get; set; }

    /// <summary>
    /// Gets or sets whether the native encoder uses group normalization instead
    /// of the research-default batch normalization. This is useful for genuine
    /// single-frame training batches; the default remains <see langword="false"/>.
    /// </summary>
    public bool UseGroupNormalization { get; set; }

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>DEVAModelSize.Base</c>.
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
    public DEVAModelSize ModelSize { get; set; }

    /// <summary>
    /// Gets or sets how many memory pixels each query pixel reads from (XMem's top-k filtering). Default: 30
    /// (Cheng et al. 2023).
    /// </summary>
    public int MemoryTopK { get; set; } = 30;

    /// <summary>
    /// Gets or sets how often, in frames, the image-level segmentation is integrated with the propagated
    /// segments and the frame is added to memory. Default: 5 (Cheng et al. 2023).
    /// </summary>
    public int MergeEvery { get; set; } = 5;

    /// <summary>
    /// Gets or sets the most frames the working memory holds; the first frame is always kept. Default: 10
    /// (<c>T_max</c>, Cheng et al. 2023).
    /// </summary>
    public int MaxMemoryFrames { get; set; } = 10;

    /// <summary>
    /// Gets or sets the IoU a propagated segment and a new segment must exceed to be associated. Default: 0.5.
    /// </summary>
    public double AssociationIoUThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets after how many consecutive unmatched integrations a propagated segment is deleted. Default:
    /// 5 (<c>L</c>, Cheng et al. 2023).
    /// </summary>
    public int MaxMissedMerges { get; set; } = 5;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
