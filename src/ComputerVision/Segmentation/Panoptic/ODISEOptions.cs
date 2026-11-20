using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// Configuration options for ODISE open-vocabulary panoptic segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ODISE model. Default values follow the original paper settings.</para>
/// </remarks>
public class ODISEOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ODISEOptions()
    {
        NumClasses = 133;
        DropRate = 0.1;
        ModelSize = ODISEModelSize.Base;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ODISEOptions(ODISEOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        ClassNames = other.ClassNames?.ToArray();
        InVocabularyClassNames = other.InVocabularyClassNames?.ToArray();
        Temperature = other.Temperature;
        InVocabularyClipWeight = other.InVocabularyClipWeight;
        OutOfVocabularyClipWeight = other.OutOfVocabularyClipWeight;
        NumQueries = other.NumQueries;
        PixelEmbeddingDimension = other.PixelEmbeddingDimension;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDimension = other.FeedForwardDimension;
        ObjectMaskThreshold = other.ObjectMaskThreshold;
        OverlapThreshold = other.OverlapThreshold;
    }

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>ODISEModelSize.Base</c>.
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
    public ODISEModelSize ModelSize { get; set; }

    /// <summary>
    /// Gets or sets the names of the closed-set classes, in class-index order (stuff classes first). When null,
    /// the names are "class 0" .. "class N-1". <see cref="SegmentationModelOptions.NumClasses"/> names are needed.
    /// </summary>
    /// <value>Null by default; otherwise exactly NumClasses names in class-index order, stuff classes first.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Give the category names (such as "road" or "car") so the text encoder can turn each
    /// into a classifier. When this is null the classes are named "class 0" to "class N-1", which only suits a closed
    /// set the model was trained on.</para>
    /// </remarks>
    public string[]? ClassNames { get; set; }

    /// <summary>
    /// Gets or sets the class names seen in training (ODISE's in-vocabulary categories). A test class whose name
    /// is in this list fuses its scores with <see cref="InVocabularyClipWeight"/>, any other class with
    /// <see cref="OutOfVocabularyClipWeight"/>. When null, every class counts as in-vocabulary.
    /// </summary>
    /// <value>Null by default, meaning every class is treated as seen in training.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> List the categories the model was trained on. Classes in this list mix in the CLIP
    /// prediction with InVocabularyClipWeight, and unseen classes with OutOfVocabularyClipWeight, because the trained
    /// classifier is less reliable on unseen ones.</para>
    /// </remarks>
    public string[]? InVocabularyClassNames { get; set; }

    /// <summary>
    /// Gets or sets the initial temperature <c>tau</c> of the diffusion classifier: its logits are cosine
    /// similarities times a learned scale initialized to <c>1 / tau</c> and clamped at 100 (ODISE's learnable
    /// <c>logit_scale</c>). Default: 0.07 (ODISE's initial temperature).
    /// </summary>
    /// <value>A positive temperature. The default is 0.07.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Class scores are cosine similarities multiplied by 1 / Temperature (a learned scale,
    /// capped at 100). A smaller value makes predictions more confident.</para>
    /// </remarks>
    public double Temperature { get; set; } = 0.07;

    /// <summary>
    /// Gets or sets the weight <c>alpha</c> of the frozen CLIP image encoder's prediction for in-vocabulary classes:
    /// <c>p = p_diffusion^(1 - alpha) * p_clip^alpha</c>. Default: 0.35 (Xu et al. 2023; lambda = 0.65 on the
    /// diffusion prediction).
    /// </summary>
    /// <value>A weight in [0, 1]. The default is 0.35.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For a class seen in training, the final probability blends the trained classifier
    /// and the frozen CLIP classifier; this is CLIP's share. Zero uses only the trained classifier.</para>
    /// </remarks>
    public double InVocabularyClipWeight { get; set; } = 0.35;

    /// <summary>
    /// Gets or sets the weight <c>beta</c> of the frozen CLIP image encoder's prediction for classes not seen in
    /// training.
    /// Default: 0.65.
    /// </summary>
    /// <value>A weight in [0, 1]. The default is 0.65.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The same blend for classes not seen in training, where CLIP gets a larger share by
    /// default because the trained classifier never saw them.</para>
    /// </remarks>
    public double OutOfVocabularyClipWeight { get; set; } = 0.65;

    /// <summary>Gets or sets the number of mask queries. Default: 100 (Mask2Former, as ODISE uses).</summary>
    /// <value>A positive count of mask queries. The default is 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each query can predict one segment, so this bounds how many segments one image can
    /// contain.</para>
    /// </remarks>
    public int NumQueries { get; set; } = 100;

    /// <summary>Gets or sets the width of the pixel embedding the queries attend to. Default: 256.</summary>
    /// <value>A positive channel count. The default is 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the length of the per-pixel feature vector the queries compare against to
    /// form their masks.</para>
    /// </remarks>
    public int PixelEmbeddingDimension { get; set; } = 256;

    /// <summary>Gets or sets the number of masked-attention decoder layers. Default: 9 (Mask2Former).</summary>
    /// <value>A positive layer count. The default is 9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each decoder layer refines the queries once by looking at the image features. More
    /// layers refine more and cost more time.</para>
    /// </remarks>
    public int NumDecoderLayers { get; set; } = 9;

    /// <summary>
    /// Gets or sets the attention heads of the Mask2Former decoder's cross- and self-attention. Default: 8 (Mask2Former's
    /// <c>NHEADS</c>, as ODISE uses).
    /// </summary>
    /// <value>A positive head count that divides PixelEmbeddingDimension. The default is 8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention in the decoder is split into this many heads, each looking at a different
    /// part of the feature vector.</para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the hidden width of the Mask2Former decoder's FFN. Default: 2048 (<c>DIM_FEEDFORWARD</c>).</summary>
    /// <value>A positive channel count. The default is 2048.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the hidden width of the small network applied to each query after
    /// attention.</para>
    /// </remarks>
    public int FeedForwardDimension { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the minimum class probability of a query kept by the panoptic merge. Default: 0.8 (Mask2Former).
    /// </summary>
    /// <value>A probability in [0, 1]. The default is 0.8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A query is kept in the panoptic result only if its best class probability reaches
    /// this value.</para>
    /// </remarks>
    public double ObjectMaskThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the minimum share of its own mask a query must win to become a segment. Default: 0.8.
    /// </summary>
    /// <value>A fraction in [0, 1]. The default is 0.8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Segments are placed from the most to the least confident. A segment is kept only if
    /// at least this share of its own mask is still free, which removes duplicates.</para>
    /// </remarks>
    public double OverlapThreshold { get; set; } = 0.8;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
