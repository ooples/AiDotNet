namespace AiDotNet.MetaLearning.Metrics;

/// <summary>
/// Detailed metrics for a single task adaptation and evaluation.
/// </summary>
/// <remarks>
/// <para>
/// Tracks the complete adaptation process for one task: how well the model learned
/// from the support set and how accurately it performed on the query set.
/// </para>
/// <para><b>For Beginners:</b> When adapting to a new task:
///
/// 1. <b>Support Set:</b> Few examples used to quickly teach the model about this task
/// 2. <b>Adaptation:</b> Model updates its parameters based on support set
/// 3. <b>Query Set:</b> New examples from the same task used to test adaptation
///
/// Good meta-learning shows:
/// - High support accuracy (model learned the examples)
/// - High query accuracy (model generalized to new examples)
/// - Fast adaptation time (few seconds, not minutes)
/// </para>
/// </remarks>
public class AdaptationMetrics
{
    /// <summary>
    /// Gets or sets the accuracy on the query set after adaptation.
    /// </summary>
    /// <value>
    /// The primary metric: how well the model performs on unseen examples
    /// after adapting to the support set. Range: [0, 1]
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> This is the key metric for few-shot learning.
    /// It measures whether the model can generalize from few support examples
    /// to new query examples. Target: >70% for 5-shot, >80% for 10-shot tasks.
    /// </para>
    /// </remarks>
    public double QueryAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the loss on the query set after adaptation.
    /// </summary>
    /// <value>
    /// Continuous measure of prediction quality on query set.
    /// Lower is better.
    /// </value>
    public double QueryLoss { get; set; }

    /// <summary>
    /// Gets or sets the accuracy on the support set after adaptation.
    /// </summary>
    /// <value>
    /// Measures how well the model fit the support examples.
    /// Should typically be very high (>90%) for successful adaptation.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Low support accuracy indicates adaptation failed.
    /// Possible causes:
    /// - Insufficient inner loop steps
    /// - Inner learning rate too low
    /// - Model capacity insufficient for task
    ///
    /// High support but low query accuracy indicates overfitting to support set.
    /// </para>
    /// </remarks>
    public double SupportAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the loss on the support set after adaptation.
    /// </summary>
    /// <value>
    /// Should be very low after successful adaptation to support examples.
    /// </value>
    public double SupportLoss { get; set; }

    /// <summary>
    /// Gets or sets the number of adaptation steps actually performed.
    /// </summary>
    /// <value>
    /// May differ from configured steps if early stopping or adaptive mechanisms are used.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> If using adaptive inner loop termination,
    /// this shows how many steps were actually needed. Fewer steps = faster adaptation.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; }

    /// <summary>
    /// Gets or sets the time taken for adaptation in milliseconds.
    /// </summary>
    /// <value>
    /// Total time for support set training + query set evaluation.
    /// Critical for real-time deployment feasibility.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Target adaptation times:
    /// - Real-time systems: &lt;100ms
    /// - Interactive systems: &lt;1s
    /// - Batch processing: &lt;10s per task
    ///
    /// Longer times may require:
    /// - Model compression
    /// - Reduced inner steps
    /// - Hardware acceleration (GPU)
    /// </para>
    /// </remarks>
    public double AdaptationTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the task identifier or description.
    /// </summary>
    /// <value>
    /// Human-readable identifier for tracking which task these metrics correspond to.
    /// </value>
    public string TaskId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets per-step metrics during adaptation.
    /// </summary>
    /// <value>
    /// Tracks support/query loss and accuracy after each inner loop step.
    /// Useful for debugging convergence issues.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Enable this for development and debugging,
    /// but disable in production for performance (avoid per-step overhead).
    /// </para>
    /// </remarks>
    public List<(int Step, double SupportLoss, double QueryLoss)> PerStepMetrics { get; set; } = new();

    /// <summary>
    /// Gets or sets algorithm-specific adaptation metrics.
    /// </summary>
    /// <value>
    /// Custom metrics for specific meta-learning algorithms.
    /// </value>
    /// <remarks>
    /// <para><b>Examples:</b>
    /// - "gradient_norm_initial": Gradient magnitude before adaptation
    /// - "gradient_norm_final": Gradient magnitude after adaptation
    /// - "parameter_change_norm": How much parameters changed during adaptation
    /// - "loss_improvement": Ratio of final to initial loss
    /// </para>
    /// </remarks>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();

    /// <summary>
    /// Returns a formatted string summary of adaptation results.
    /// </summary>
    /// <returns>Human-readable summary of adaptation metrics.</returns>
    public override string ToString()
    {
        return $"Adaptation Metrics for Task '{TaskId}':\n" +
               $"  Query Accuracy: {QueryAccuracy:P2} (Loss: {QueryLoss:F4})\n" +
               $"  Support Accuracy: {SupportAccuracy:P2} (Loss: {SupportLoss:F4})\n" +
               $"  Steps: {AdaptationSteps}, Time: {AdaptationTimeMs:F2}ms";
    }
}
