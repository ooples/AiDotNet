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
    public string[]? ClassNames { get; set; }

    /// <summary>
    /// Gets or sets the class names seen in training (ODISE's in-vocabulary categories). A test class whose name
    /// is in this list fuses its scores with <see cref="InVocabularyClipWeight"/>, any other class with
    /// <see cref="OutOfVocabularyClipWeight"/>. When null, every class counts as in-vocabulary.
    /// </summary>
    public string[]? InVocabularyClassNames { get; set; }

    /// <summary>
    /// Gets or sets the initial temperature <c>tau</c> of the diffusion classifier: its logits are cosine
    /// similarities times a learned scale initialized to <c>1 / tau</c> and clamped at 100 (ODISE's learnable
    /// <c>logit_scale</c>). Default: 0.07 (ODISE's initial temperature).
    /// </summary>
    public double Temperature { get; set; } = 0.07;

    /// <summary>
    /// Gets or sets the weight <c>alpha</c> of the frozen CLIP image encoder's prediction for in-vocabulary classes:
    /// <c>p = p_diffusion^(1 - alpha) * p_clip^alpha</c>. Default: 0.35 (Xu et al. 2023; lambda = 0.65 on the
    /// diffusion prediction).
    /// </summary>
    public double InVocabularyClipWeight { get; set; } = 0.35;

    /// <summary>
    /// Gets or sets the weight <c>beta</c> of the frozen CLIP image encoder's prediction for classes not seen in
    /// training.
    /// Default: 0.65.
    /// </summary>
    public double OutOfVocabularyClipWeight { get; set; } = 0.65;

    /// <summary>Gets or sets the number of mask queries. Default: 100 (Mask2Former, as ODISE uses).</summary>
    public int NumQueries { get; set; } = 100;

    /// <summary>Gets or sets the width of the pixel embedding the queries attend to. Default: 256.</summary>
    public int PixelEmbeddingDimension { get; set; } = 256;

    /// <summary>Gets or sets the number of masked-attention decoder layers. Default: 9 (Mask2Former).</summary>
    public int NumDecoderLayers { get; set; } = 9;

    /// <summary>
    /// Gets or sets the minimum class probability of a query kept by the panoptic merge. Default: 0.8 (Mask2Former).
    /// </summary>
    public double ObjectMaskThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the minimum share of its own mask a query must win to become a segment. Default: 0.8.
    /// </summary>
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
