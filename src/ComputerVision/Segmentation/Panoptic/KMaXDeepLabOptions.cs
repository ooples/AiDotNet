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
        QueryDimension = other.QueryDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDimension = other.FeedForwardDimension;
        PanopticQualityLossWeight = other.PanopticQualityLossWeight;
        MaskIdLossWeight = other.MaskIdLossWeight;
        InstanceDiscriminationLossWeight = other.InstanceDiscriminationLossWeight;
        InstanceDiscriminationTemperature = other.InstanceDiscriminationTemperature;
        InstanceDiscriminationSampleCount = other.InstanceDiscriminationSampleCount;
        InstanceDiscriminationSampleTemperature = other.InstanceDiscriminationSampleTemperature;
        SemanticLossWeight = other.SemanticLossWeight;
        SemanticSampleCount = other.SemanticSampleCount;
        SemanticSampleTemperature = other.SemanticSampleTemperature;
        PixelHeadDimension = other.PixelHeadDimension;
        VoidPriorProbability = other.VoidPriorProbability;
        SemanticAsppDimension = other.SemanticAsppDimension;
        SemanticAtrousRates = (int[])other.SemanticAtrousRates.Clone();
        SemanticAsppDropRate = other.SemanticAsppDropRate;
        SemanticDecoderDimension = other.SemanticDecoderDimension;
        SemanticLowLevelProjections = (int[])other.SemanticLowLevelProjections.Clone();
        SemanticHeadDimension = other.SemanticHeadDimension;
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
    /// Gets or sets the width of the pixel embedding the cluster centers are compared with. Default: 128
    /// (deeplab2 configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto L42, <c>pixel_space_head.output_channels</c>).
    /// </summary>
    public int PixelEmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the width of the pixel-space head's 1x1 convolution, between its 5x5 depthwise convolution and the
    /// pixel embedding. Default: 256 (kmax_meta_r50_os32.textproto L43, <c>pixel_space_head.head_channels</c>; the
    /// PyTorch port's <c>_pixel_space_head_conv1bnact</c>).
    /// </summary>
    public int PixelHeadDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the void-class prior of the class head: a fixed bias <c>log((C - 1) * p / (1 - p))</c> on the void
    /// logit (C the classes plus void) makes an uninformative class head predict void with probability p. Default: 0.9
    /// (deeplab2 model/decoder/max_deeplab.py L60 <c>add_bias_towards_void(..., void_prior_prob=0.9)</c>, the bias at
    /// L91-L96; the PyTorch port's kmax_transformer_decoder.py L49-L54). Must lie in (0, 1).
    /// </summary>
    public double VoidPriorProbability { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the number of kMaX transformer decoder stages (each a k-means cross-attention, a self-attention
    /// among the cluster centers and an FFN). Default: 6 (the kMaX decoder's <c>DEC_LAYERS = [2, 2, 2]</c>).
    /// </summary>
    public int NumKMeansIterations { get; set; } = 6;

    /// <summary>
    /// Gets or sets the width of the cluster centers (the decoder's query and bottleneck width). Default: 256
    /// (<c>in_channel_query = 256</c>; the keys take half of it, the values all of it, as base_filters 128 with key
    /// expansion 1 and value expansion 2 give).
    /// </summary>
    public int QueryDimension { get; set; } = 256;

    /// <summary>Gets or sets the self-attention heads among the cluster centers. Default: 8.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the hidden width of the cluster centers' FFN. Default: 2048.</summary>
    public int FeedForwardDimension { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the weight of the PQ-style loss. Default: 3.0 (Yu et al. 2022, Sec. 4.1; deeplab2
    /// configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto L73).
    /// </summary>
    public double PanopticQualityLossWeight { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the weight of the mask-ID cross-entropy. Default: 0.3 (Sec. 4.1; kmax_meta_r50_os32.textproto L76).
    /// </summary>
    public double MaskIdLossWeight { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the weight of the pixel-wise instance discrimination loss (MaX-DeepLab's auxiliary loss, kept by
    /// kMaX-DeepLab). Default: 1.0 (Sec. 4.1; kmax_meta_r50_os32.textproto L79).
    /// </summary>
    public double InstanceDiscriminationLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the instance discrimination similarity temperature. Default: 0.3 (deeplab2
    /// model/loss/max_deeplab_loss.py L214; the PyTorch port's criterion.py L294).
    /// </summary>
    public double InstanceDiscriminationTemperature { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets how many pixels per image the instance discrimination loss samples. Default: 4096
    /// (kmax_meta_r50_os32.textproto L57, <c>instance_discrimination_sample_k</c>).
    /// </summary>
    public int InstanceDiscriminationSampleCount { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the temperature of the instance discrimination pixel sampling (pixels are drawn with probability
    /// proportional to their segment's inverse area raised to this power). Default: 1.5 (kmax_meta_r50_os32.textproto
    /// L58, <c>instance_discrimination_sample_temperature</c>; the PyTorch port's config.py L46).
    /// </summary>
    public double InstanceDiscriminationSampleTemperature { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the weight of the auxiliary semantic segmentation loss. Default: 1.0 (Sec. 4.1;
    /// kmax_meta_r50_os32.textproto L70).
    /// </summary>
    public double SemanticLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets how many pixels per image the auxiliary semantic loss samples. Default: 4096
    /// (kmax_meta_r50_os32.textproto L59, <c>semantic_sample_k</c>).
    /// </summary>
    public int SemanticSampleCount { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the temperature of the auxiliary semantic loss's pixel sampling. Default: 2.0
    /// (kmax_meta_r50_os32.textproto L60, <c>semantic_sample_temperature</c>; the PyTorch port's config.py L48).
    /// </summary>
    public double SemanticSampleTemperature { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the width of the auxiliary semantic decoder's ASPP branches and projection. Default: 256
    /// (kmax_meta_r50_os32.textproto L35, <c>decoder.aspp_channels</c>).
    /// </summary>
    public int SemanticAsppDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the dilation rates of the ASPP's 3x3 branches, on the stride-32 backbone features. Default: 6, 12,
    /// 18 (kmax_meta_r50_os32.textproto L36-L38, <c>decoder.atrous_rates</c>).
    /// </summary>
    public int[] SemanticAtrousRates { get; set; } = [6, 12, 18];

    /// <summary>
    /// Gets or sets the dropout after the ASPP projection. Default: 0.1 (deeplab2 model/decoder/aspp.py,
    /// <c>self._proj_drop = layers.Dropout(rate=0.1)</c>).
    /// </summary>
    public double SemanticAsppDropRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the width of the semantic decoder's fusion convolutions. Default: 256
    /// (kmax_meta_r50_os32.textproto L34, <c>decoder.decoder_channels</c>).
    /// </summary>
    public int SemanticDecoderDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the widths the semantic decoder projects its low-level backbone features to, coarsest first.
    /// Default: 64 (stride 8, <c>res3</c>) and 32 (stride 4, <c>res2</c>) (kmax_meta_r50_os32.textproto L45-L52,
    /// <c>auxiliary_low_level</c>).
    /// </summary>
    public int[] SemanticLowLevelProjections { get; set; } = [64, 32];

    /// <summary>
    /// Gets or sets the width of the semantic head's 5x5 depthwise-separable convolution. Default: 256
    /// (kmax_meta_r50_os32.textproto L55, <c>auxiliary_semantic_head.head_channels</c>).
    /// </summary>
    public int SemanticHeadDimension { get; set; } = 256;

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
