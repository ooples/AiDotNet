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
    /// <value>A positive memory-pixel count. The default is 30.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each pixel of a new frame reads from only this many of the best-matching pixels
    /// stored in memory. If memory holds fewer pixels, all of them are used.</para>
    /// </remarks>
    public int MemoryTopK { get; set; } = 30;

    /// <summary>
    /// Gets or sets how often, in frames, the image-level segmentation is integrated with the propagated
    /// segments and the frame is added to memory. Default: 5 (Cheng et al. 2023).
    /// </summary>
    /// <value>A non-negative frame count. The default is 5; zero disables periodic integration.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Every this many frames, the per-frame image segmentation is merged with the segments
    /// carried from earlier frames, which lets new objects appear and corrects drift.</para>
    /// </remarks>
    public int MergeEvery { get; set; } = 5;

    /// <summary>
    /// Gets or sets the most frames the working memory holds; the first frame is always kept. Default: 10
    /// (<c>T_max</c>, Cheng et al. 2023).
    /// </summary>
    /// <value>A positive frame count. The default is 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Memory keeps at most this many frames. The first frame is always kept, and the
    /// oldest of the others is dropped when the limit is reached.</para>
    /// </remarks>
    public int MaxMemoryFrames { get; set; } = 10;

    /// <summary>
    /// Gets or sets the IoU a propagated segment and a new segment must exceed to be associated. Default: 0.5.
    /// </summary>
    /// <value>A fraction in [0, 1]. The default is 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A carried segment and a newly detected one are treated as the same object when their
    /// overlap (intersection over union) is greater than this value.</para>
    /// </remarks>
    public double AssociationIoUThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets after how many consecutive unmatched integrations a propagated segment is deleted. Default:
    /// 5 (<c>L</c>, Cheng et al. 2023).
    /// </summary>
    /// <value>A positive count. The default is 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A tracked object that finds no matching detection in this many consecutive
    /// integrations is removed.</para>
    /// </remarks>
    public int MaxMissedMerges { get; set; } = 5;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1), when <see cref="MemoryTopK"/>,
    /// <see cref="MaxMemoryFrames"/> or <see cref="MaxMissedMerges"/> is not positive, when <see cref="MergeEvery"/> is
    /// negative (zero disables integration), or when <see cref="AssociationIoUThreshold"/> is not a fraction in [0, 1].
    /// </exception>
    public void Validate()
    {
        ValidateSegmentationCore();
        if (MemoryTopK < 1)
            throw new ArgumentOutOfRangeException(nameof(MemoryTopK), MemoryTopK, "At least one memory pixel must be read.");
        if (MergeEvery < 0)
            throw new ArgumentOutOfRangeException(nameof(MergeEvery), MergeEvery, "The integration period cannot be negative; zero disables it.");
        if (MaxMemoryFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(MaxMemoryFrames), MaxMemoryFrames, "Memory must hold at least one frame.");
        // A NaN fails both comparisons, so it is rejected too.
        if (!(AssociationIoUThreshold >= 0 && AssociationIoUThreshold <= 1))
            throw new ArgumentOutOfRangeException(nameof(AssociationIoUThreshold), AssociationIoUThreshold, "The association IoU must be a fraction in [0, 1].");
        if (MaxMissedMerges < 1)
            throw new ArgumentOutOfRangeException(nameof(MaxMissedMerges), MaxMissedMerges, "At least one missed integration must be allowed.");
    }
}
