namespace AiDotNet.Models.Results;

/// <summary>
/// Results from a complete meta-training run with history tracking.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// This class aggregates metrics across an entire meta-training session, tracking how performance
/// evolves over many meta-iterations. It combines the functionality of what were previously separate
/// "Metrics" and "Metadata" classes into a unified Result pattern consistent with the codebase.
/// </para>
/// <para><b>For Beginners:</b> Meta-training is the process of training your model to be good at
/// learning new tasks quickly. This happens over many iterations:
///
/// 1. Sample a batch of tasks
/// 2. Adapt to each task (inner loop)
/// 3. Update meta-parameters based on how well adaptations worked (outer loop)
/// 4. Repeat for many iterations
///
/// This result tracks:
/// - <b>Learning curves:</b> How loss and accuracy change over iterations
/// - <b>Final performance:</b> The end results after training
/// - <b>Training time:</b> How long it took
/// - <b>Convergence:</b> Whether training successfully improved the model
///
/// Use this to:
/// - Monitor training progress
/// - Diagnose training issues
/// - Compare different meta-learning configurations
/// - Report results in papers or documentation
/// </para>
/// </remarks>
public class MetaTrainingResult<T>
{
    /// <summary>
    /// Gets the total number of meta-training iterations completed.
    /// </summary>
    /// <value>
    /// The count of outer loop updates performed during training.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each iteration represents one complete cycle of:
    /// - Sample tasks → Adapt to each → Update meta-parameters
    ///
    /// More iterations generally lead to better meta-learning, but with diminishing returns.
    /// Typical values: 10,000-60,000 for research, 1,000-10,000 for practice.
    /// </para>
    /// </remarks>
    public int TotalIterations { get; }

    /// <summary>
    /// Gets the meta-loss history across all iterations.
    /// </summary>
    /// <value>
    /// A vector where each element is the meta-loss for that iteration.
    /// Lower values indicate better meta-learning performance.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Use this for:
    /// - Plotting learning curves
    /// - Detecting convergence or divergence
    /// - Implementing early stopping
    /// - Comparing training runs
    ///
    /// The meta-loss measures how well the model adapts across tasks in the outer loop.
    /// It should generally decrease over training.
    /// </para>
    /// </remarks>
    public Vector<T> LossHistory { get; }

    /// <summary>
    /// Gets the accuracy history across all iterations.
    /// </summary>
    /// <value>
    /// A vector where each element is the average accuracy for that iteration.
    /// Higher values indicate better meta-learning performance.
    /// </value>
    public Vector<T> AccuracyHistory { get; }

    /// <summary>
    /// Gets the total time taken for meta-training.
    /// </summary>
    /// <value>
    /// The elapsed time from start to finish of training.
    /// </value>
    public TimeSpan TrainingTime { get; }

    /// <summary>
    /// Gets the final meta-loss after training.
    /// </summary>
    /// <value>
    /// The meta-loss from the last iteration, representing final training performance.
    /// </value>
    public T FinalLoss => LossHistory.Length > 0 ? LossHistory[^1] : MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets the final accuracy after training.
    /// </summary>
    /// <value>
    /// The accuracy from the last iteration, representing final training performance.
    /// </value>
    public T FinalAccuracy => AccuracyHistory.Length > 0 ? AccuracyHistory[^1] : MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets the initial meta-loss before training.
    /// </summary>
    /// <value>
    /// The meta-loss from the first iteration, representing baseline performance.
    /// </value>
    public T InitialLoss => LossHistory.Length > 0 ? LossHistory[0] : MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets the initial accuracy before training.
    /// </summary>
    /// <value>
    /// The accuracy from the first iteration, representing baseline performance.
    /// </value>
    public T InitialAccuracy => AccuracyHistory.Length > 0 ? AccuracyHistory[0] : MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets algorithm-specific metrics collected during training.
    /// </summary>
    /// <value>
    /// A dictionary of custom metrics with generic T values.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Common additional metrics include:
    /// - "best_loss": Lowest loss achieved during training
    /// - "best_accuracy": Highest accuracy achieved during training
    /// - "gradient_norm_avg": Average gradient magnitude
    /// - "tasks_per_second": Training throughput
    /// - "convergence_iteration": When loss stabilized
    /// </para>
    /// </remarks>
    public Dictionary<string, T> AdditionalMetrics { get; }

    /// <summary>
    /// Initializes a new instance with complete training history.
    /// </summary>
    /// <param name="lossHistory">Meta-loss values from each training iteration.</param>
    /// <param name="accuracyHistory">Accuracy values from each training iteration.</param>
    /// <param name="trainingTime">Total time taken for training.</param>
    /// <param name="additionalMetrics">Optional algorithm-specific metrics.</param>
    /// <exception cref="ArgumentNullException">Thrown when lossHistory or accuracyHistory is null.</exception>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths or are empty.</exception>
    /// <remarks>
    /// <para>
    /// This constructor follows the established pattern of accepting raw data and deriving
    /// computed properties from it. The history vectors should contain one value per training
    /// iteration, in chronological order.
    /// </para>
    /// <para><b>For Beginners:</b> Call this at the end of training to package all your
    /// training history together. The constructor automatically calculates derived metrics
    /// like FinalLoss, InitialLoss, etc.
    /// </para>
    /// </remarks>
    public MetaTrainingResult(
        Vector<T> lossHistory,
        Vector<T> accuracyHistory,
        TimeSpan trainingTime,
        Dictionary<string, T>? additionalMetrics = null)
    {
        if (lossHistory == null)
            throw new ArgumentNullException(nameof(lossHistory));
        if (accuracyHistory == null)
            throw new ArgumentNullException(nameof(accuracyHistory));
        if (lossHistory.Length != accuracyHistory.Length)
            throw new ArgumentException("Loss and accuracy histories must have the same length");
        if (lossHistory.Length == 0)
            throw new ArgumentException("Must provide at least one iteration of history", nameof(lossHistory));

        TotalIterations = lossHistory.Length;
        LossHistory = lossHistory;
        AccuracyHistory = accuracyHistory;
        TrainingTime = trainingTime;
        AdditionalMetrics = additionalMetrics ?? new Dictionary<string, T>();
    }

    /// <summary>
    /// Calculates the total improvement in loss from start to finish.
    /// </summary>
    /// <returns>The difference between initial and final loss (positive means improvement).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how much the model improved during training.
    ///
    /// - Positive value: Loss decreased (good!)
    /// - Zero: No improvement (needs investigation)
    /// - Negative value: Loss increased (training problem)
    ///
    /// For example:
    /// - Initial loss: 2.5
    /// - Final loss: 0.8
    /// - Improvement: 1.7 (68% reduction)
    /// </para>
    /// </remarks>
    public T CalculateLossImprovement()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Subtract(InitialLoss, FinalLoss);
    }

    /// <summary>
    /// Calculates the total improvement in accuracy from start to finish.
    /// </summary>
    /// <returns>The difference between final and initial accuracy (positive means improvement).</returns>
    public T CalculateAccuracyImprovement()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Subtract(FinalAccuracy, InitialAccuracy);
    }

    /// <summary>
    /// Checks if training converged based on loss stabilization.
    /// </summary>
    /// <param name="windowSize">Number of recent iterations to analyze (default: 100).</param>
    /// <param name="varianceThreshold">Maximum variance to consider converged (default: 0.001).</param>
    /// <returns>True if loss variance in recent window is below threshold.</returns>
    /// <remarks>
    /// <para><b>For Production:</b> Use this to:
    /// - Implement automatic early stopping
    /// - Validate training completion
    /// - Diagnose non-convergent runs
    ///
    /// Convergence means the loss has stabilized and further training is unlikely to help.
    /// </para>
    /// </remarks>
    public bool HasConverged(int windowSize = 100, double varianceThreshold = 0.001)
    {
        if (LossHistory.Length < windowSize)
            return false;

        var recentLosses = LossHistory.ToArray()
            .Skip(LossHistory.Length - windowSize)
            .ToArray();

        var recentVector = new Vector<T>(recentLosses);
        var variance = StatisticsHelper<T>.CalculateVariance(recentVector);

        return MathHelper.GetNumericOperations<T>().ToDouble(variance) < varianceThreshold;
    }

    /// <summary>
    /// Finds the best (lowest) loss achieved during training.
    /// </summary>
    /// <returns>A tuple containing the best loss value and the iteration it occurred at.</returns>
    /// <remarks>
    /// <para><b>For Production:</b> The best loss might occur before the final iteration,
    /// especially if:
    /// - Learning rate is too high (oscillation)
    /// - Training ran too long (overfitting to training tasks)
    /// - Need early stopping or learning rate decay
    /// </para>
    /// </remarks>
    public (T BestLoss, int Iteration) FindBestLoss()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T bestLoss = LossHistory[0];
        int bestIteration = 0;

        for (int i = 1; i < LossHistory.Length; i++)
        {
            if (numOps.ToDouble(LossHistory[i]) < numOps.ToDouble(bestLoss))
            {
                bestLoss = LossHistory[i];
                bestIteration = i;
            }
        }

        return (bestLoss, bestIteration);
    }

    /// <summary>
    /// Finds the best (highest) accuracy achieved during training.
    /// </summary>
    /// <returns>A tuple containing the best accuracy value and the iteration it occurred at.</returns>
    public (T BestAccuracy, int Iteration) FindBestAccuracy()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T bestAccuracy = AccuracyHistory[0];
        int bestIteration = 0;

        for (int i = 1; i < AccuracyHistory.Length; i++)
        {
            if (numOps.ToDouble(AccuracyHistory[i]) > numOps.ToDouble(bestAccuracy))
            {
                bestAccuracy = AccuracyHistory[i];
                bestIteration = i;
            }
        }

        return (bestAccuracy, bestIteration);
    }

    /// <summary>
    /// Generates a comprehensive training report.
    /// </summary>
    /// <returns>A formatted string summarizing training results.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a human-readable summary of your training run
    /// that you can print, log, or include in documentation.
    /// </para>
    /// </remarks>
    public string GenerateReport()
    {
        var report = new StringBuilder();
        var lossImprovement = CalculateLossImprovement();
        var accuracyImprovement = CalculateAccuracyImprovement();
        var (bestLoss, bestLossIter) = FindBestLoss();
        var (bestAcc, bestAccIter) = FindBestAccuracy();

        report.AppendLine("Meta-Training Results");
        report.AppendLine("====================");
        report.AppendLine($"Total Iterations: {TotalIterations}");
        report.AppendLine($"Training Time: {TrainingTime.TotalSeconds:F2} seconds");
        report.AppendLine($"Iterations per Second: {TotalIterations / TrainingTime.TotalSeconds:F2}");
        report.AppendLine();

        report.AppendLine("Loss Metrics:");
        report.AppendLine($"  Initial: {InitialLoss}");
        report.AppendLine($"  Final: {FinalLoss}");
        report.AppendLine($"  Best: {bestLoss} (iteration {bestLossIter})");
        report.AppendLine($"  Total Improvement: {lossImprovement}");
        report.AppendLine();

        report.AppendLine("Accuracy Metrics:");
        report.AppendLine($"  Initial: {InitialAccuracy}");
        report.AppendLine($"  Final: {FinalAccuracy}");
        report.AppendLine($"  Best: {bestAcc} (iteration {bestAccIter})");
        report.AppendLine($"  Total Improvement: {accuracyImprovement}");
        report.AppendLine();

        report.AppendLine($"Converged: {HasConverged()}");

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
