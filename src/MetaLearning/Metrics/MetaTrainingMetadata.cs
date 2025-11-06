namespace AiDotNet.MetaLearning.Metrics;

/// <summary>
/// Metadata returned from a complete meta-training run.
/// </summary>
/// <remarks>
/// <para>
/// This class aggregates information from multiple meta-training iterations,
/// providing a complete picture of the training process.
/// </para>
/// <para><b>For Beginners:</b> Think of this as the "training report" that tells you:
/// - How many iterations you ran
/// - How the loss changed over time (learning curve)
/// - How long the entire training took
/// - Final performance metrics
/// </para>
/// </remarks>
public class MetaTrainingMetadata
{
    /// <summary>
    /// Gets or sets the total number of meta-training iterations completed.
    /// </summary>
    public int Iterations { get; set; }

    /// <summary>
    /// Gets or sets the loss history across all iterations.
    /// </summary>
    /// <remarks>
    /// Each element corresponds to the average meta-loss for that iteration.
    /// Useful for plotting learning curves and diagnosing training issues.
    /// </remarks>
    public List<double> LossHistory { get; set; } = new();

    /// <summary>
    /// Gets or sets the accuracy history across all iterations.
    /// </summary>
    public List<double> AccuracyHistory { get; set; } = new();

    /// <summary>
    /// Gets or sets the total training time.
    /// </summary>
    public TimeSpan TrainingTime { get; set; }

    /// <summary>
    /// Gets or sets the final meta-loss after training.
    /// </summary>
    public double FinalLoss => LossHistory.Count > 0 ? LossHistory[^1] : 0.0;

    /// <summary>
    /// Gets or sets the final accuracy after training.
    /// </summary>
    public double FinalAccuracy => AccuracyHistory.Count > 0 ? AccuracyHistory[^1] : 0.0;

    /// <summary>
    /// Gets additional metrics collected during training.
    /// </summary>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();

    /// <summary>
    /// Returns a formatted string representation of the training metadata.
    /// </summary>
    public override string ToString()
    {
        return $"Meta-Training Results:\n" +
               $"  Iterations: {Iterations}\n" +
               $"  Final Loss: {FinalLoss:F4}\n" +
               $"  Final Accuracy: {FinalAccuracy:F4}\n" +
               $"  Training Time: {TrainingTime.TotalSeconds:F2}s";
    }
}
