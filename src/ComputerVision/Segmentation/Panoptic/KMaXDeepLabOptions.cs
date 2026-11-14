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
    /// <value>A positive count of mask slots. The default is 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each cluster center is one slot that can predict one segment (an object or a stuff
    /// region). More slots allow more segments per image but make the decoder slower.</para>
    /// </remarks>
    public int NumClusterCenters { get; set; } = 128;

    /// <summary>
    /// Gets or sets the width of the pixel embedding the cluster centers are compared with. Default: 128
    /// (deeplab2 configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto L42, <c>pixel_space_head.output_channels</c>).
    /// </summary>
    /// <value>A positive channel count. The default is 128.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Every pixel is described by a vector of this length, and each cluster center is
    /// compared with it to decide which pixels belong to that center's mask. Larger vectors can separate more segments
    /// but cost memory.</para>
    /// </remarks>
    public int PixelEmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the width of the pixel-space head's 1x1 convolution, between its 5x5 depthwise convolution and the
    /// pixel embedding. Default: 256 (kmax_meta_r50_os32.textproto L43, <c>pixel_space_head.head_channels</c>; the
    /// PyTorch port's <c>_pixel_space_head_conv1bnact</c>).
    /// </summary>
    /// <value>A positive channel count. The default is 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the width of the small convolution that produces the pixel embeddings. It
    /// changes the capacity of that head, not the size of its output.</para>
    /// </remarks>
    public int PixelHeadDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the void-class prior of the class head: a fixed bias <c>log((C - 1) * p / (1 - p))</c> on the void
    /// logit (C the classes plus void) makes an uninformative class head predict void with probability p. Default: 0.9
    /// (deeplab2 model/decoder/max_deeplab.py L60 <c>add_bias_towards_void(..., void_prior_prob=0.9)</c>, the bias at
    /// L91-L96; the PyTorch port's kmax_transformer_decoder.py L49-L54). Must lie in (0, 1).
    /// </summary>
    /// <value>A probability strictly between 0 and 1. The default is 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before training, the class head is biased so that an untrained slot predicts "no
    /// object" with this probability. This keeps empty slots from producing spurious segments early in training.</para>
    /// </remarks>
    public double VoidPriorProbability { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the number of kMaX transformer decoder stages (each a k-means cross-attention, a self-attention
    /// among the cluster centers and an FFN). Default: 6 (the kMaX decoder's <c>DEC_LAYERS = [2, 2, 2]</c>).
    /// </summary>
    /// <value>A positive stage count. The default is 6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each decoder stage refines the cluster centers once, like one step of k-means. More
    /// stages give more refinement at the cost of speed.</para>
    /// </remarks>
    public int NumKMeansIterations { get; set; } = 6;

    /// <summary>
    /// Gets or sets the width of the cluster centers (the decoder's query and bottleneck width). Default: 256
    /// (<c>in_channel_query = 256</c>; the keys take half of it, the values all of it, as base_filters 128 with key
    /// expansion 1 and value expansion 2 give).
    /// </summary>
    /// <value>A positive channel count, divisible by NumAttentionHeads. The default is 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the length of the vector each cluster center carries through the decoder.
    /// Larger values give each slot more capacity.</para>
    /// </remarks>
    public int QueryDimension { get; set; } = 256;

    /// <summary>Gets or sets the self-attention heads among the cluster centers. Default: 8.</summary>
    /// <value>A positive head count that divides QueryDimension. The default is 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The cluster centers exchange information through attention split into this many
    /// heads, each looking at a different part of the vector.</para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the hidden width of the cluster centers' FFN. Default: 2048.</summary>
    /// <value>A positive channel count. The default is 2048.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the hidden width of the small network applied to each cluster center after
    /// attention. Larger values add capacity and memory use.</para>
    /// </remarks>
    public int FeedForwardDimension { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the weight of the PQ-style loss. Default: 3.0 (Yu et al. 2022, Sec. 4.1; deeplab2
    /// configs/coco/kmax_deeplab/kmax_meta_r50_os32.textproto L73).
    /// </summary>
    /// <value>A non-negative weight. The default is 3.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This scales the main training loss, which rewards slots for matching both the class
    /// and the mask of a ground-truth segment. Zero turns that loss off.</para>
    /// </remarks>
    public double PanopticQualityLossWeight { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the weight of the mask-ID cross-entropy. Default: 0.3 (Sec. 4.1; kmax_meta_r50_os32.textproto L76).
    /// </summary>
    /// <value>A non-negative weight. The default is 0.3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This scales the loss that teaches every pixel which slot it belongs to. Zero turns
    /// that loss off.</para>
    /// </remarks>
    public double MaskIdLossWeight { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the weight of the pixel-wise instance discrimination loss (MaX-DeepLab's auxiliary loss, kept by
    /// kMaX-DeepLab). Default: 1.0 (Sec. 4.1; kmax_meta_r50_os32.textproto L79).
    /// </summary>
    /// <value>A non-negative weight. The default is 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This scales an auxiliary loss that pulls together the embeddings of pixels in the
    /// same segment and pushes apart those of different segments. Zero turns it off.</para>
    /// </remarks>
    public double InstanceDiscriminationLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the instance discrimination similarity temperature. Default: 0.3 (deeplab2
    /// model/loss/max_deeplab_loss.py L214; the PyTorch port's criterion.py L294).
    /// </summary>
    /// <value>A positive temperature. The default is 0.3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Similarities in the instance discrimination loss are divided by this value. A
    /// smaller value makes the loss sharper and more sensitive to small differences.</para>
    /// </remarks>
    public double InstanceDiscriminationTemperature { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets how many pixels per image the instance discrimination loss samples. Default: 4096
    /// (kmax_meta_r50_os32.textproto L57, <c>instance_discrimination_sample_k</c>).
    /// </summary>
    /// <value>A positive pixel count per image. The default is 4096.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The instance discrimination loss uses only this many sampled pixels per image, which
    /// keeps it affordable on large images.</para>
    /// </remarks>
    public int InstanceDiscriminationSampleCount { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the temperature of the instance discrimination pixel sampling (pixels are drawn with probability
    /// proportional to their segment's inverse area raised to this power). Default: 1.5 (kmax_meta_r50_os32.textproto
    /// L58, <c>instance_discrimination_sample_temperature</c>; the PyTorch port's config.py L46).
    /// </summary>
    /// <value>A non-negative exponent. The default is 1.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pixels of small segments are sampled more often so that small objects are not
    /// ignored. A larger value favours small segments more strongly; zero samples every pixel equally.</para>
    /// </remarks>
    public double InstanceDiscriminationSampleTemperature { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the weight of the auxiliary semantic segmentation loss. Default: 1.0 (Sec. 4.1;
    /// kmax_meta_r50_os32.textproto L70).
    /// </summary>
    /// <value>A non-negative weight. The default is 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This scales an auxiliary per-pixel class loss from a separate semantic head. Zero
    /// turns it off.</para>
    /// </remarks>
    public double SemanticLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets how many pixels per image the auxiliary semantic loss samples. Default: 4096
    /// (kmax_meta_r50_os32.textproto L59, <c>semantic_sample_k</c>).
    /// </summary>
    /// <value>A positive pixel count per image. The default is 4096.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The auxiliary semantic loss is computed on this many sampled pixels per
    /// image.</para>
    /// </remarks>
    public int SemanticSampleCount { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the temperature of the auxiliary semantic loss's pixel sampling. Default: 2.0
    /// (kmax_meta_r50_os32.textproto L60, <c>semantic_sample_temperature</c>; the PyTorch port's config.py L48).
    /// </summary>
    /// <value>A non-negative exponent. The default is 2.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like the instance sampling exponent, this makes pixels of small segments more likely
    /// to be sampled for the semantic loss; zero samples every pixel equally.</para>
    /// </remarks>
    public double SemanticSampleTemperature { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the width of the auxiliary semantic decoder's ASPP branches and projection. Default: 256
    /// (kmax_meta_r50_os32.textproto L35, <c>decoder.aspp_channels</c>).
    /// </summary>
    /// <value>A positive channel count. The default is 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the width of the multi-scale context block of the auxiliary semantic
    /// head.</para>
    /// </remarks>
    public int SemanticAsppDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the dilation rates of the ASPP's 3x3 branches, on the stride-32 backbone features. Default: 6, 12,
    /// 18 (kmax_meta_r50_os32.textproto L36-L38, <c>decoder.atrous_rates</c>).
    /// </summary>
    /// <value>Positive dilation rates, one per context branch. The default is [6, 12, 18].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each rate is how far apart a branch's convolution samples, so larger rates see wider
    /// context. One branch is built per entry.</para>
    /// </remarks>
    public int[] SemanticAtrousRates { get; set; } = [6, 12, 18];

    /// <summary>
    /// Gets or sets the dropout after the ASPP projection. Default: 0.1 (deeplab2 model/decoder/aspp.py,
    /// <c>self._proj_drop = layers.Dropout(rate=0.1)</c>).
    /// </summary>
    /// <value>A fraction in [0, 1). The default is 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the dropout applied in the semantic head's context block during training to
    /// reduce overfitting.</para>
    /// </remarks>
    public double SemanticAsppDropRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the width of the semantic decoder's fusion convolutions. Default: 256
    /// (kmax_meta_r50_os32.textproto L34, <c>decoder.decoder_channels</c>).
    /// </summary>
    /// <value>A positive channel count. The default is 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the width of the layers that upsample and fuse features in the semantic
    /// head.</para>
    /// </remarks>
    public int SemanticDecoderDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the widths the semantic decoder projects its low-level backbone features to, coarsest first.
    /// Default: 64 (stride 8, <c>res3</c>) and 32 (stride 4, <c>res2</c>) (kmax_meta_r50_os32.textproto L45-L52,
    /// <c>auxiliary_low_level</c>).
    /// </summary>
    /// <value>Positive channel counts, one per fused encoder stage. The default is [64, 32].</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Fine detail from earlier encoder stages is narrowed to these widths before it is
    /// fused into the semantic head.</para>
    /// </remarks>
    public int[] SemanticLowLevelProjections { get; set; } = [64, 32];

    /// <summary>
    /// Gets or sets the width of the semantic head's 5x5 depthwise-separable convolution. Default: 256
    /// (kmax_meta_r50_os32.textproto L55, <c>auxiliary_semantic_head.head_channels</c>).
    /// </summary>
    /// <value>A positive channel count. The default is 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the width of the last convolution before the semantic head's per-class
    /// output.</para>
    /// </remarks>
    public int SemanticHeadDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the minimum class probability of a thing cluster in the panoptic merge. Default: 0.7.
    /// </summary>
    /// <value>A probability in [0, 1]. The default is 0.7.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A slot predicting an object class is kept in the panoptic result only if its class
    /// probability reaches this value.</para>
    /// </remarks>
    public double ClassThresholdThing { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the minimum class probability of a stuff cluster in the panoptic merge. Default: 0.5.
    /// </summary>
    /// <value>A probability in [0, 1]. The default is 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A slot predicting a stuff class (such as sky or road) is kept only if its class
    /// probability reaches this value.</para>
    /// </remarks>
    public double ClassThresholdStuff { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the mask probability a pixel needs to belong to a cluster's mask. Default: 0.4
    /// (<c>PIXEL_CONFIDENCE_THRESHOLD</c>).
    /// </summary>
    /// <value>A probability in [0, 1]. The default is 0.4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A pixel joins a slot's mask only if the mask is at least this confident at that
    /// pixel.</para>
    /// </remarks>
    public double PixelConfidenceThreshold { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets the share of a cluster's mask that must still be unclaimed for it to become a segment.
    /// Default: 0.8.
    /// </summary>
    /// <value>A fraction in [0, 1]. The default is 0.8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Segments are placed from the most to the least confident. A segment is kept only if
    /// at least this share of its mask is still free, which removes duplicates.</para>
    /// </remarks>
    public double OverlapThreshold { get; set; } = 0.8;

    /// <summary>Gets or sets the class-score exponent of the merge order. Default: 1.0.</summary>
    /// <value>A non-negative exponent. The default is 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Segments are ranked by class score raised to this power times mask score raised to
    /// ReorderMaskWeight. A larger value lets the class score dominate the order.</para>
    /// </remarks>
    public double ReorderClassWeight { get; set; } = 1.0;

    /// <summary>Gets or sets the mask-score exponent of the merge order. Default: 1.0.</summary>
    /// <value>A non-negative exponent. The default is 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The counterpart of ReorderClassWeight: a larger value lets the mask score dominate
    /// the order in which segments are placed.</para>
    /// </remarks>
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
