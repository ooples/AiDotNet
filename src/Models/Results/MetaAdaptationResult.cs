
using System.Text;

namespace AiDotNet.Models.Results;

/// <summary>
/// Results from adapting a meta-learner to a single task.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// This class captures detailed metrics from the inner loop adaptation process for a single task.
/// It tracks both support set performance (where the model adapts) and query set performance
/// (where we measure true generalization), along with adaptation details.
/// </para>
/// <para><b>For Beginners:</b> Meta-learning has two phases for each task:
///
/// 1. <b>Adaptation (Inner Loop):</b> The model quickly learns from the support set (a few examples)
/// 2. <b>Evaluation:</b> We test on the query set (held-out examples) to see if it really learned
///
/// This result tracks:
/// - <b>Support metrics:</b> How well the model fits the training examples (should be very good)
/// - <b>Query metrics:</b> How well it generalizes (the real test!)
/// - <b>Adaptation details:</b> How many steps, how long it took, etc.
///
/// For example, in 5-way 1-shot learning:
/// - Support set: 5 examples (1 per class) - model adapts on these
/// - Query set: 15 examples (3 per class) - model is evaluated on these
/// - Good meta-learning: High query accuracy even with tiny support set
/// </para>
/// </remarks>
public class MetaAdaptationResult<T>
{
    /// <summary>
    /// Gets the accuracy on the query set after adaptation.
    /// </summary>
    /// <value>
    /// The proportion of correct predictions on query examples, after adapting on the support set.
    /// This is the key metric for meta-learning quality - it measures true generalization.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the most important number - how well the model performs
    /// on new examples after learning from just a few examples (the support set).
    ///
    /// Higher query accuracy means better meta-learning. In research papers, this is typically
    /// reported as "test accuracy" or "query accuracy" for N-way K-shot tasks.
    /// </para>
    /// </remarks>
    public T QueryAccuracy { get; }

    /// <summary>
    /// Gets the loss on the query set after adaptation.
    /// </summary>
    /// <value>
    /// The average loss computed on query examples after inner loop adaptation.
    /// </value>
    public T QueryLoss { get; }

    /// <summary>
    /// Gets the accuracy on the support set after adaptation.
    /// </summary>
    /// <value>
    /// The proportion of correct predictions on support examples.
    /// This should typically be very high (close to 1.0 or 100%).
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Support accuracy tells you if the model successfully fit
    /// the training examples. It should be very high (often 100%) since these are the examples
    /// the model adapted on.
    ///
    /// If support accuracy is low, it indicates:
    /// - The inner loop learning rate might be too small
    /// - Not enough adaptation steps
    /// - The task might be too difficult
    /// </para>
    /// </remarks>
    public T SupportAccuracy { get; }

    /// <summary>
    /// Gets the loss on the support set after adaptation.
    /// </summary>
    /// <value>
    /// The average loss computed on support examples after inner loop adaptation.
    /// </value>
    public T SupportLoss { get; }

    /// <summary>
    /// Gets the number of gradient steps taken during adaptation.
    /// </summary>
    /// <value>
    /// The count of inner loop optimization steps performed.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> This is useful for:
    /// - Verifying your inner loop configuration
    /// - Debugging convergence issues
    /// - Comparing different adaptation strategies
    ///
    /// Typical values: 1-10 steps for simple tasks, 5-50 for complex tasks
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; }

    /// <summary>
    /// Gets the time taken for adaptation in milliseconds.
    /// </summary>
    /// <value>
    /// The elapsed time for the inner loop adaptation process.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Use this for:
    /// - Performance profiling and optimization
    /// - Setting realistic timeout values
    /// - Comparing algorithm efficiency
    ///
    /// Typical values vary widely based on:
    /// - Model complexity
    /// - Number of adaptation steps
    /// - Hardware (GPU vs CPU)
    /// </para>
    /// </remarks>
    public double AdaptationTimeMs { get; }

    /// <summary>
    /// Gets the loss values recorded at each adaptation step (optional).
    /// </summary>
    /// <value>
    /// A list of loss values, one per adaptation step, showing convergence progress.
    /// Empty if not tracked during adaptation.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Use this for:
    /// - Visualizing convergence curves
    /// - Debugging non-convergent adaptations
    /// - Tuning inner loop hyperparameters
    /// - Early stopping strategies
    /// </para>
    /// </remarks>
    public List<T> PerStepLosses { get; }

    /// <summary>
    /// Gets algorithm-specific metrics for this adaptation.
    /// </summary>
    /// <value>
    /// A dictionary of custom metrics with generic T values.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Common additional metrics include:
    /// - "initial_loss": Loss before any adaptation
    /// - "loss_reduction": How much loss decreased during adaptation
    /// - "gradient_norm": Magnitude of gradients during adaptation
    /// - "parameter_change": How much parameters moved
    /// </para>
    /// </remarks>
    public Dictionary<string, T> AdditionalMetrics { get; }

    /// <summary>
    /// Initializes a new instance with adaptation metrics.
    /// </summary>
    /// <param name="queryAccuracy">Accuracy on query set after adaptation.</param>
    /// <param name="queryLoss">Loss on query set after adaptation.</param>
    /// <param name="supportAccuracy">Accuracy on support set after adaptation.</param>
    /// <param name="supportLoss">Loss on support set after adaptation.</param>
    /// <param name="adaptationSteps">Number of gradient steps taken.</param>
    /// <param name="adaptationTimeMs">Time taken for adaptation in milliseconds.</param>
    /// <param name="perStepLosses">Optional per-step loss values for convergence analysis.</param>
    /// <param name="additionalMetrics">Optional algorithm-specific metrics.</param>
    /// <remarks>
    /// <para>
    /// This constructor captures a complete snapshot of adaptation performance for a single task.
    /// Unlike MetaEvaluationResult which aggregates across tasks, this represents one task's
    /// detailed adaptation results.
    /// </para>
    /// <para><b>For Beginners:</b> Call this after adapting to a single task to package all
    /// the results together. The most important values are queryAccuracy (how well you generalized)
    /// and queryLoss (the error on new examples).
    /// </para>
    /// </remarks>
    public MetaAdaptationResult(
        T queryAccuracy,
        T queryLoss,
        T supportAccuracy,
        T supportLoss,
        int adaptationSteps,
        double adaptationTimeMs,
        List<T>? perStepLosses = null,
        Dictionary<string, T>? additionalMetrics = null)
    {
        QueryAccuracy = queryAccuracy;
        QueryLoss = queryLoss;
        SupportAccuracy = supportAccuracy;
        SupportLoss = supportLoss;
        AdaptationSteps = adaptationSteps;
        AdaptationTimeMs = adaptationTimeMs;
        PerStepLosses = perStepLosses != null ? new List<T>(perStepLosses) : new List<T>();
        AdditionalMetrics = additionalMetrics != null ? new Dictionary<string, T>(additionalMetrics) : new Dictionary<string, T>();
    }

    /// <summary>
    /// Calculates the overfitting gap between support and query performance.
    /// </summary>
    /// <returns>The difference between support and query accuracy (positive means overfitting).</returns>
    /// <remarks>
    /// <para>
    /// A large overfitting gap indicates the model memorized the support set without learning
    /// generalizable patterns. This can happen with:
    /// - Too many adaptation steps
    /// - Too high inner learning rate
    /// - Support set too small or unrepresentative
    /// </para>
    /// <para><b>For Beginners:</b> The overfitting gap tells you if the model "memorized" the
    /// training examples instead of truly learning.
    ///
    /// - Small gap (< 5-10%): Good generalization
    /// - Medium gap (10-20%): Some overfitting, may need tuning
    /// - Large gap (> 20%): Significant overfitting problem
    ///
    /// For example:
    /// - Support accuracy: 95%
    /// - Query accuracy: 85%
    /// - Overfitting gap: 10% (moderate)
    /// </para>
    /// </remarks>
    public T CalculateOverfittingGap()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Subtract(SupportAccuracy, QueryAccuracy);
    }

    /// <summary>
    /// Checks if the adaptation converged based on loss reduction.
    /// </summary>
    /// <param name="convergenceThreshold">Minimum loss reduction to consider converged (default: 0.01).</param>
    /// <returns>True if loss decreased sufficiently, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Production:</b> Use this to:
    /// - Detect failed adaptations early
    /// - Implement adaptive inner loop step counts
    /// - Flag problematic tasks for analysis
    ///
    /// Requires PerStepLosses to be populated.
    /// </para>
    /// </remarks>
    public bool DidConverge(double convergenceThreshold = 0.01)
    {
        if (PerStepLosses.Count < 2)
            return false;

        var numOps = MathHelper.GetNumericOperations<T>();
        T initialLoss = PerStepLosses[0];
        T finalLoss = PerStepLosses[PerStepLosses.Count - 1];
        T lossReduction = numOps.Subtract(initialLoss, finalLoss);

        return Convert.ToDouble(lossReduction) >= convergenceThreshold;
    }

    /// <summary>
    /// Generates a formatted summary of adaptation results.
    /// </summary>
    /// <returns>A human-readable string with key adaptation metrics.</returns>
    public string GenerateReport()
    {
        var report = new StringBuilder();
        var overfittingGap = CalculateOverfittingGap();

        report.AppendLine("Task Adaptation Report");
        report.AppendLine("=====================");
        report.AppendLine($"Adaptation Steps: {AdaptationSteps}");
        report.AppendLine($"Adaptation Time: {AdaptationTimeMs:F2} ms");
        report.AppendLine();

        report.AppendLine("Query Set Performance (Key Metric):");
        report.AppendLine($"  Accuracy: {QueryAccuracy}");
        report.AppendLine($"  Loss: {QueryLoss}");
        report.AppendLine();

        report.AppendLine("Support Set Performance:");
        report.AppendLine($"  Accuracy: {SupportAccuracy}");
        report.AppendLine($"  Loss: {SupportLoss}");
        report.AppendLine();

        report.AppendLine($"Overfitting Gap: {overfittingGap}");

        if (PerStepLosses.Count > 0)
        {
            report.AppendLine($"Converged: {DidConverge()}");
            report.AppendLine($"Initial Loss: {PerStepLosses[0]}");
            report.AppendLine($"Final Loss: {PerStepLosses[PerStepLosses.Count - 1]}");
        }

        if (AdditionalMetrics.Count > 0)
        {
            report.AppendLine();
            report.AppendLine("Additional Metrics:");
            foreach (var kvp in AdditionalMetrics.OrderBy(x => x.Key))
            {
                report.AppendLine($"  {kvp.Key}: {kvp.Value}");
            }
        }

        return report.ToString();
    }
}
