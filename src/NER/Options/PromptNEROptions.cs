using AiDotNet.Enums;

namespace AiDotNet.NER.Options;

/// <summary>
/// Configuration for PromptNER's paper architecture and training recipe.
/// </summary>
/// <remarks>
/// The reference configuration uses BERT-large, Adam with a peak learning rate of
/// <c>2e-5</c>, and a linear warmup followed by linear decay over 50-100 epochs.
/// The step counts remain public because the exact number of optimizer steps depends
/// on the caller's dataset and batch size.
/// </remarks>
public sealed class PromptNEROptions : TransformerNEROptions
{
    /// <summary>Creates the reference-paper configuration.</summary>
    public PromptNEROptions()
    {
        Variant = NERModelVariant.Large;
        HiddenDimension = 1024;
        NumAttentionHeads = 16;
        NumTransformerLayers = 24;
        IntermediateDimension = 4096;
        MaxSequenceLength = 512;
        LearningRate = 2e-5;
        WarmupSteps = 10;
        TotalTrainingSteps = 100;
    }

    /// <summary>Creates an independent copy of another PromptNER configuration.</summary>
    public PromptNEROptions(PromptNEROptions other)
        : base(other)
    {
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        AdamEpsilon = other.AdamEpsilon;
        EnableGradientClipping = other.EnableGradientClipping;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>Gets or sets Adam's first-moment coefficient.</summary>
    public double AdamBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's second-moment coefficient.</summary>
    public double AdamBeta2 { get; set; } = 0.999;

    /// <summary>Gets or sets Adam's numerical-stability epsilon.</summary>
    public double AdamEpsilon { get; set; } = 1e-8;

    /// <summary>Gets or sets whether global-norm gradient clipping is enabled.</summary>
    public bool EnableGradientClipping { get; set; } = false;

    /// <summary>Gets or sets the global gradient-norm limit when clipping is enabled.</summary>
    public double MaxGradientNorm { get; set; } = 1.0;
}
