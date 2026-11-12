using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// Configuration options for kMaX-DeepLab panoptic segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the KMaXDeepLab model. Default values follow the original paper settings.</para>
/// </remarks>
public class KMaXDeepLabOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public KMaXDeepLabOptions()
    {
        NumClasses = 133;
        DropRate = 0.1;
        ModelSize = KMaXDeepLabModelSize.R50;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KMaXDeepLabOptions(KMaXDeepLabOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        NumClusterCenters = other.NumClusterCenters;
        PixelEmbeddingDimension = other.PixelEmbeddingDimension;
        NumKMeansIterations = other.NumKMeansIterations;
        ClassThresholdThing = other.ClassThresholdThing;
        ClassThresholdStuff = other.ClassThresholdStuff;
        PixelConfidenceThreshold = other.PixelConfidenceThreshold;
        OverlapThreshold = other.OverlapThreshold;
        ReorderClassWeight = other.ReorderClassWeight;
        ReorderMaskWeight = other.ReorderMaskWeight;
    }

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>KMaXDeepLabModelSize.R50</c>.
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
    public KMaXDeepLabModelSize ModelSize { get; set; }

    /// <summary>
    /// Gets or sets the number of cluster centers (mask slots). Default: 128 (Yu et al. 2022,
    /// <c>NUM_OBJECT_QUERIES = 128</c>).
    /// </summary>
    public int NumClusterCenters { get; set; } = 128;

    /// <summary>
    /// Gets or sets the width of the pixel embedding the cluster centers are compared with. Default: 128.
    /// </summary>
    public int PixelEmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of k-means cross-attention updates. Default: 6 (the kMaX decoder's
    /// <c>DEC_LAYERS = [2, 2, 2]</c>).
    /// </summary>
    public int NumKMeansIterations { get; set; } = 6;

    /// <summary>
    /// Gets or sets the minimum class probability of a thing cluster in the panoptic merge. Default: 0.7.
    /// </summary>
    public double ClassThresholdThing { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the minimum class probability of a stuff cluster in the panoptic merge. Default: 0.5.
    /// </summary>
    public double ClassThresholdStuff { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the mask probability a pixel needs to belong to a cluster's mask. Default: 0.4
    /// (<c>PIXEL_CONFIDENCE_THRESHOLD</c>).
    /// </summary>
    public double PixelConfidenceThreshold { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets the share of a cluster's mask that must still be unclaimed for it to become a segment.
    /// Default: 0.8.
    /// </summary>
    public double OverlapThreshold { get; set; } = 0.8;

    /// <summary>Gets or sets the class-score exponent of the merge order. Default: 1.0.</summary>
    public double ReorderClassWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the mask-score exponent of the merge order. Default: 1.0.</summary>
    public double ReorderMaskWeight { get; set; } = 1.0;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
