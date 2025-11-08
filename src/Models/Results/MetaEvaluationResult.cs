namespace AiDotNet.Models.Results;

/// <summary>
/// Results from evaluating a meta-learner across multiple tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// This class aggregates evaluation metrics from multiple tasks to assess meta-learning performance.
/// It uses the existing BasicStats infrastructure to provide comprehensive statistical analysis
/// of how well the meta-learner adapts to new tasks.
/// </para>
/// <para><b>For Beginners:</b> Meta-learning evaluation tests how well your model can quickly learn new tasks.
///
/// This result tells you:
/// - <b>Average accuracy:</b> How well the model performs on new tasks after quick adaptation
/// - <b>Consistency:</b> How much performance varies across different tasks (standard deviation)
/// - <b>Confidence:</b> Statistical confidence intervals for the results
/// - <b>Per-task details:</b> Individual task results for deep analysis
///
/// For example, if you're doing 5-way 1-shot classification:
/// - You sample many test tasks (e.g., 100 tasks)
/// - For each task, the model sees 1 example per class and must classify new examples
/// - This result aggregates accuracy and loss across all those tasks
/// - Higher mean accuracy and lower standard deviation indicate better meta-learning
/// </para>
/// </remarks>
public class MetaEvaluationResult<T>
{
    /// <summary>
    /// Gets comprehensive statistics for accuracy across all evaluated tasks.
    /// </summary>
    /// <value>
    /// BasicStats containing mean, standard deviation, median, quartiles, and distribution metrics
    /// for accuracy values across tasks. Accuracy is measured on query sets after adaptation.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This contains all the statistical measures for accuracy:
    /// - Mean: The average accuracy across all tasks
    /// - StandardDeviation: How much accuracy varies between tasks
    /// - Median: The middle accuracy value (less affected by outliers)
    /// - Min/Max: The worst and best task accuracies
    ///
    /// Lower standard deviation means the model consistently performs well across different tasks.
    /// </para>
    /// </remarks>
    public BasicStats<T> AccuracyStats { get; }

    /// <summary>
    /// Gets comprehensive statistics for loss across all evaluated tasks.
    /// </summary>
    /// <value>
    /// BasicStats containing mean, standard deviation, median, quartiles, and distribution metrics
    /// for loss values across tasks. Loss is measured on query sets after adaptation.
    /// </value>
    public BasicStats<T> LossStats { get; }

    /// <summary>
    /// Gets the number of tasks used for evaluation.
    /// </summary>
    /// <value>
    /// The total count of tasks sampled and evaluated.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> More tasks give more reliable statistics.
    ///
    /// Typical values:
    /// - Research papers: 600-1000 tasks for final evaluation
    /// - Quick validation: 100-200 tasks
    /// - Debugging: 10-50 tasks
    /// </para>
    /// </remarks>
    public int NumTasks { get; }

    /// <summary>
    /// Gets the individual accuracy values for each evaluated task.
    /// </summary>
    /// <value>
    /// A vector containing one accuracy value per task, in the order tasks were evaluated.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Use this for:
    /// - Detailed analysis of task-level performance
    /// - Identifying problematic task types
    /// - Creating histograms or other visualizations
    /// - Statistical hypothesis testing
    /// </para>
    /// </remarks>
    public Vector<T> PerTaskAccuracies { get; }

    /// <summary>
    /// Gets the individual loss values for each evaluated task.
    /// </summary>
    /// <value>
    /// A vector containing one loss value per task, in the order tasks were evaluated.
    /// </value>
    public Vector<T> PerTaskLosses { get; }

    /// <summary>
    /// Gets algorithm-specific metrics that don't fit standard categories.
    /// </summary>
    /// <value>
    /// A dictionary of custom metrics with generic T values.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Common additional metrics include:
    /// - "support_accuracy": Accuracy on support sets (should be very high)
    /// - "adaptation_time_ms": Average time per task adaptation
    /// - "convergence_rate": How quickly inner loop converged
    /// - "gradient_magnitude": Size of adaptation gradients
    /// </para>
    /// </remarks>
    public Dictionary<string, T> AdditionalMetrics { get; }

    /// <summary>
    /// Gets the total time taken for evaluation.
    /// </summary>
    /// <value>
    /// The elapsed time for evaluating all tasks.
    /// </value>
    public TimeSpan EvaluationTime { get; }

    /// <summary>
    /// Initializes a new instance with task results and calculates all statistics.
    /// </summary>
    /// <param name="taskAccuracies">Accuracy values from all evaluated tasks.</param>
    /// <param name="taskLosses">Loss values from all evaluated tasks.</param>
    /// <param name="evaluationTime">Time taken for evaluation.</param>
    /// <param name="additionalMetrics">Optional algorithm-specific metrics.</param>
    /// <exception cref="ArgumentNullException">Thrown when taskAccuracies or taskLosses is null.</exception>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths or are empty.</exception>
    /// <remarks>
    /// <para>
    /// This constructor follows the established pattern of calculating statistics in the constructor.
    /// It uses the existing BasicStats infrastructure to compute comprehensive statistics from the
    /// raw per-task values, ensuring consistency with other result classes in the codebase.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor takes your raw task results and automatically
    /// calculates all the statistical summaries (mean, std dev, etc.).
    ///
    /// You typically call this from meta-learning evaluation code after collecting results
    /// from many tasks. The constructor does all the statistical work for you.
    /// </para>
    /// </remarks>
    public MetaEvaluationResult(
        Vector<T> taskAccuracies,
        Vector<T> taskLosses,
        TimeSpan evaluationTime,
        Dictionary<string, T>? additionalMetrics = null)
    {
        if (taskAccuracies == null)
            throw new ArgumentNullException(nameof(taskAccuracies));
        if (taskLosses == null)
            throw new ArgumentNullException(nameof(taskLosses));
        if (taskAccuracies.Length != taskLosses.Length)
            throw new ArgumentException("Accuracy and loss vectors must have the same length");
        if (taskAccuracies.Length == 0)
            throw new ArgumentException("Must provide at least one task result", nameof(taskAccuracies));

        NumTasks = taskAccuracies.Length;
        PerTaskAccuracies = taskAccuracies;
        PerTaskLosses = taskLosses;
        EvaluationTime = evaluationTime;
        AdditionalMetrics = additionalMetrics != null ? new Dictionary<string, T>(additionalMetrics) : new Dictionary<string, T>();

        // Calculate statistics using existing infrastructure
        AccuracyStats = new BasicStats<T>(new BasicStatsInputs<T> { Values = taskAccuracies });
        LossStats = new BasicStats<T>(new BasicStatsInputs<T> { Values = taskLosses });
    }

    /// <summary>
    /// Calculates the 95% confidence interval for mean accuracy.
    /// </summary>
    /// <returns>A tuple containing the lower and upper bounds of the confidence interval.</returns>
    /// <remarks>
    /// <para>
    /// Uses the standard error of the mean with a z-score of 1.96 (95% confidence level).
    /// Assumes the distribution of task accuracies is approximately normal, which is
    /// reasonable for large sample sizes (Central Limit Theorem).
    /// </para>
    /// <para><b>For Beginners:</b> The confidence interval tells you the range where the true
    /// average accuracy likely falls.
    ///
    /// For example, if you get (0.82, 0.88) with mean accuracy 0.85:
    /// - You can be 95% confident the true mean accuracy is between 82% and 88%
    /// - Narrower intervals mean more precise estimates
    /// - More tasks (larger NumTasks) give narrower intervals
    /// </para>
    /// </remarks>
    public (T Lower, T Upper) GetAccuracyConfidenceInterval()
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Standard error = std_dev / sqrt(n)
        T sqrtN = numOps.Sqrt(numOps.FromDouble(NumTasks));
        T standardError = numOps.Divide(AccuracyStats.StandardDeviation, sqrtN);

        // 95% confidence: mean ± 1.96 * SE
        T marginOfError = numOps.Multiply(numOps.FromDouble(1.96), standardError);

        return (
            numOps.Subtract(AccuracyStats.Mean, marginOfError),
            numOps.Add(AccuracyStats.Mean, marginOfError)
        );
    }

    /// <summary>
    /// Calculates the 95% confidence interval for mean loss.
    /// </summary>
    /// <returns>A tuple containing the lower and upper bounds of the confidence interval.</returns>
    public (T Lower, T Upper) GetLossConfidenceInterval()
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        T sqrtN = numOps.Sqrt(numOps.FromDouble(NumTasks));
        T standardError = numOps.Divide(LossStats.StandardDeviation, sqrtN);
        T marginOfError = numOps.Multiply(numOps.FromDouble(1.96), standardError);

        return (
            numOps.Subtract(LossStats.Mean, marginOfError),
            numOps.Add(LossStats.Mean, marginOfError)
        );
    }

    /// <summary>
    /// Generates a formatted summary report of the evaluation results.
    /// </summary>
    /// <returns>A human-readable string containing key evaluation metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a text summary you can print or log
    /// to quickly see how well your meta-learner performed.
    ///
    /// It includes the most important metrics:
    /// - Mean accuracy ± standard deviation
    /// - 95% confidence interval
    /// - Loss statistics
    /// - Number of tasks evaluated
    /// - Evaluation time
    /// </para>
    /// </remarks>
    public string GenerateReport()
    {
        var report = new StringBuilder();
        var (accLower, accUpper) = GetAccuracyConfidenceInterval();
        var (lossLower, lossUpper) = GetLossConfidenceInterval();

        report.AppendLine("Meta-Learning Evaluation Report");
        report.AppendLine("==============================");
        report.AppendLine($"Tasks Evaluated: {NumTasks}");
        report.AppendLine($"Evaluation Time: {EvaluationTime.TotalSeconds:F2} seconds");
        report.AppendLine();

        report.AppendLine("Accuracy Metrics:");
        report.AppendLine($"  Mean: {AccuracyStats.Mean}");
        report.AppendLine($"  Std Dev: {AccuracyStats.StandardDeviation}");
        report.AppendLine($"  95% CI: [{accLower}, {accUpper}]");
        report.AppendLine($"  Median: {AccuracyStats.Median}");
        report.AppendLine($"  Range: [{AccuracyStats.Min}, {AccuracyStats.Max}]");
        report.AppendLine();

        report.AppendLine("Loss Metrics:");
        report.AppendLine($"  Mean: {LossStats.Mean}");
        report.AppendLine($"  Std Dev: {LossStats.StandardDeviation}");
        report.AppendLine($"  95% CI: [{lossLower}, {lossUpper}]");
        report.AppendLine($"  Median: {LossStats.Median}");
        report.AppendLine();

        if (AdditionalMetrics.Count > 0)
        {
            report.AppendLine("Additional Metrics:");
            foreach (var kvp in AdditionalMetrics.OrderBy(x => x.Key))
            {
                report.AppendLine($"  {kvp.Key}: {kvp.Value}");
            }
        }

        return report.ToString();
    }
}
