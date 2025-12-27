namespace AiDotNet.SelfSupervisedLearning.Integration;

/// <summary>
/// Configuration for fine-tuning SSL pretrained encoders.
/// </summary>
public class FineTuningConfig
{
    /// <summary>Fine-tuning strategy.</summary>
    public FineTuningStrategy Strategy { get; set; } = FineTuningStrategy.FullFineTuning;

    /// <summary>Number of epochs.</summary>
    public int? Epochs { get; set; }

    /// <summary>Batch size.</summary>
    public int? BatchSize { get; set; }

    /// <summary>Base learning rate.</summary>
    public double? LearningRate { get; set; }

    /// <summary>Learning rate multiplier for encoder (default: 0.1).</summary>
    public double EncoderLRMultiplier { get; set; } = 0.1;
}
