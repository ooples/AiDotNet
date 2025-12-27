namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Fine-tuning strategies for SSL pretrained encoders.
/// </summary>
public enum FineTuningStrategy
{
    /// <summary>Update all parameters including encoder.</summary>
    FullFineTuning,
    /// <summary>Freeze encoder, only train classifier.</summary>
    LinearProbing,
    /// <summary>Progressively unfreeze encoder layers.</summary>
    GradualUnfreezing
}
