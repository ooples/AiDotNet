namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Result from a single benchmark evaluation.
/// </summary>
public class BenchmarkResult<T>
{
    /// <summary>
    /// The evaluation protocol used.
    /// </summary>
    public BenchmarkProtocol Protocol { get; set; }

    /// <summary>
    /// Top-1 classification accuracy.
    /// </summary>
    public double Top1Accuracy { get; set; }

    /// <summary>
    /// Top-5 classification accuracy.
    /// </summary>
    public double Top5Accuracy { get; set; }

    /// <summary>
    /// Percentage of training data used (for few-shot).
    /// </summary>
    public double SamplePercentage { get; set; } = 1.0;

    /// <summary>
    /// Training accuracy history.
    /// </summary>
    public List<double> TrainingHistory { get; set; } = [];

    /// <summary>
    /// Validation accuracy history.
    /// </summary>
    public List<double> ValidationHistory { get; set; } = [];

    /// <summary>
    /// Additional metrics (e.g., different k values for k-NN).
    /// </summary>
    public Dictionary<string, T> AdditionalMetrics { get; set; } = [];
}
