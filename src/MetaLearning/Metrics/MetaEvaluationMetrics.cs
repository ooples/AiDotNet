namespace AiDotNet.MetaLearning.Metrics;

/// <summary>
/// Comprehensive metrics for evaluating meta-learning performance across multiple tasks.
/// </summary>
/// <remarks>
/// <para>
/// Meta-evaluation assesses how well a meta-trained model can adapt to new, unseen tasks.
/// These metrics provide statistical measures of adaptation performance with confidence intervals.
/// </para>
/// <para><b>For Beginners:</b> After meta-training, you test on completely new tasks to see
/// how well the model learned to adapt. These metrics tell you:
///
/// - <b>Mean Accuracy:</b> Average success rate across all test tasks
/// - <b>Confidence Interval:</b> Statistical range where true accuracy likely falls
/// - <b>Standard Deviation:</b> How consistent performance is across tasks
///
/// For example, "Accuracy: 85% ± 3%" means the model adapted successfully on most tasks,
/// with consistent performance (small standard deviation).
/// </para>
/// </remarks>
public class MetaEvaluationMetrics
{
    /// <summary>
    /// Gets or sets the mean accuracy across all evaluated tasks.
    /// </summary>
    /// <value>
    /// Average accuracy on query sets after adapting to each task using support sets.
    /// Range: [0, 1] where 0 = 0% and 1 = 100% accuracy.
    /// </value>
    public double Accuracy { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation of accuracy across tasks.
    /// </summary>
    /// <value>
    /// Measures consistency of adaptation performance.
    /// Lower values indicate more reliable, consistent adaptation.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> High standard deviation suggests the model adapts
    /// well to some tasks but poorly to others. This may indicate:
    /// - Need for more diverse meta-training tasks
    /// - Task-specific biases in the meta-learned initialization
    /// - Insufficient inner loop adaptation steps
    /// </para>
    /// </remarks>
    public double AccuracyStd { get; set; }

    /// <summary>
    /// Gets or sets the 95% confidence interval for accuracy.
    /// </summary>
    /// <value>
    /// A tuple (lower, upper) defining the statistical confidence interval.
    /// With 95% confidence, the true mean accuracy lies within this range.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If your confidence interval is [0.82, 0.88], you can be
    /// 95% confident that the model's true accuracy on similar tasks is between 82% and 88%.
    ///
    /// Narrower intervals mean more reliable estimates (need more evaluation tasks for narrower intervals).
    /// </para>
    /// </remarks>
    public (double Lower, double Upper) ConfidenceInterval { get; set; }

    /// <summary>
    /// Gets or sets the mean loss across all evaluated tasks.
    /// </summary>
    /// <value>
    /// Average loss on query sets after task adaptation.
    /// Lower values indicate better predictions.
    /// </value>
    public double Loss { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation of loss across tasks.
    /// </summary>
    /// <value>
    /// Measures consistency of loss values across different tasks.
    /// </value>
    public double LossStd { get; set; }

    /// <summary>
    /// Gets or sets the number of tasks used for evaluation.
    /// </summary>
    /// <value>
    /// More evaluation tasks provide more reliable statistics.
    /// Typical: 100-1000 tasks for robust evaluation.
    /// </value>
    public int NumTasks { get; set; }

    /// <summary>
    /// Gets or sets the mean time per task adaptation in milliseconds.
    /// </summary>
    /// <value>
    /// Average time to adapt to a single task (support set training + query set evaluation).
    /// Important for real-time deployment considerations.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Track this metric to ensure adaptation is fast enough
    /// for your deployment scenario. Few-shot learning should adapt in seconds, not minutes.
    /// </para>
    /// </remarks>
    public double MeanAdaptationTimeMs { get; set; }

    /// <summary>
    /// Gets or sets per-task accuracy values for detailed analysis.
    /// </summary>
    /// <value>
    /// Individual accuracy for each evaluated task.
    /// Useful for identifying outliers or failure modes.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Analyze this distribution to identify:
    /// - Tasks where adaptation fails (accuracy near 0)
    /// - Consistently easy vs. hard task types
    /// - Whether performance is bimodal (some tasks work, others don't)
    /// </para>
    /// </remarks>
    public List<double> PerTaskAccuracies { get; set; } = new();

    /// <summary>
    /// Gets or sets algorithm-specific evaluation metrics.
    /// </summary>
    /// <value>
    /// Custom metrics that don't fit standard categories.
    /// </value>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();

    /// <summary>
    /// Returns a formatted string summary of evaluation results.
    /// </summary>
    /// <returns>Human-readable summary of meta-evaluation metrics.</returns>
    public override string ToString()
    {
        return $"Meta-Evaluation Results:\n" +
               $"  Accuracy: {Accuracy:P2} ± {AccuracyStd:P2}\n" +
               $"  95% CI: [{ConfidenceInterval.Lower:P2}, {ConfidenceInterval.Upper:P2}]\n" +
               $"  Loss: {Loss:F4} ± {LossStd:F4}\n" +
               $"  Tasks: {NumTasks}\n" +
               $"  Avg Adaptation Time: {MeanAdaptationTimeMs:F2}ms";
    }
}
