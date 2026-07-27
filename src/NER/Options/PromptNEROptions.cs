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
    }

    /// <summary>Creates an independent copy of another PromptNER configuration.</summary>
    public PromptNEROptions(PromptNEROptions other)
        : base(other)
    {
        WarmupSteps = other.WarmupSteps;
        TotalTrainingSteps = other.TotalTrainingSteps;
        WarmupInitialLearningRate = other.WarmupInitialLearningRate;
        EndLearningRate = other.EndLearningRate;
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        AdamEpsilon = other.AdamEpsilon;
        EnableGradientClipping = other.EnableGradientClipping;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>
    /// Gets or sets the linear-warmup length in optimizer steps.
    /// </summary>
    /// <remarks>
    /// The paper specifies linear warmup but not a dataset-independent step count.
    /// Ten steps is the 10% warmup for the default 100-step/epoch schedule.
    /// </remarks>
    public int WarmupSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the total warmup-plus-decay schedule length.
    /// </summary>
    /// <remarks>The default selects the upper end of the paper's 50-100 epoch range.</remarks>
    public int TotalTrainingSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the first positive warmup learning rate, or <see langword="null"/>
    /// to derive <c>LearningRate / WarmupSteps</c>.
    /// </summary>
    /// <remarks>
    /// Optimizer steps are one-based in the training recipe. Starting the first real
    /// update at exactly zero would silently turn a caller's first <c>Train</c> call
    /// into a no-op rather than a warmup update.
    /// </remarks>
    public double? WarmupInitialLearningRate { get; set; }

    /// <summary>Gets or sets the learning rate reached at the end of linear decay.</summary>
    public double EndLearningRate { get; set; } = 0.0;

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
