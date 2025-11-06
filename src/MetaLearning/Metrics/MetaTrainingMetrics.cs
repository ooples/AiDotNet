namespace AiDotNet.MetaLearning.Metrics;

/// <summary>
/// Metrics collected during a single meta-training step (outer loop update).
/// </summary>
/// <remarks>
/// <para>
/// Meta-training metrics track performance during the outer loop optimization,
/// where the meta-learner updates its parameters based on adaptation performance
/// across a batch of tasks.
/// </para>
/// <para><b>For Beginners:</b> Think of meta-training as learning how to learn:
///
/// - <b>MetaLoss:</b> How well the model adapts across all tasks in this batch
/// - <b>TaskLoss:</b> Average loss on individual tasks after adaptation
/// - <b>Accuracy:</b> How accurate predictions are after quick adaptation
/// - <b>NumTasks:</b> How many tasks were used in this training step
///
/// Lower losses and higher accuracy indicate better meta-learning performance.
/// </para>
/// </remarks>
public class MetaTrainingMetrics
{
    /// <summary>
    /// Gets or sets the meta-training loss (outer loop optimization objective).
    /// </summary>
    /// <value>
    /// The loss computed across all tasks after their inner loop adaptations.
    /// This is what the meta-learner tries to minimize through outer loop updates.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main metric the meta-learner optimizes.
    /// It measures how well the initial parameters enable quick task adaptation.
    /// Lower values mean the model is getting better at learning new tasks quickly.
    /// </para>
    /// </remarks>
    public double MetaLoss { get; set; }

    /// <summary>
    /// Gets or sets the average task-specific loss after inner loop adaptation.
    /// </summary>
    /// <value>
    /// The mean loss across all tasks after they've been adapted using their support sets.
    /// This shows how well the model performs on each individual task.
    /// </value>
    public double TaskLoss { get; set; }

    /// <summary>
    /// Gets or sets the average accuracy on query sets after adaptation.
    /// </summary>
    /// <value>
    /// The mean accuracy across all task query sets, measured after adapting on support sets.
    /// Expressed as a value between 0 (0%) and 1 (100%).
    /// </value>
    public double Accuracy { get; set; }

    /// <summary>
    /// Gets or sets the number of tasks used in this meta-training step.
    /// </summary>
    /// <value>
    /// The meta-batch size (number of tasks sampled for this outer loop update).
    /// </value>
    public int NumTasks { get; set; }

    /// <summary>
    /// Gets or sets the current meta-training iteration number.
    /// </summary>
    /// <value>
    /// The iteration count for tracking training progress.
    /// </value>
    public int Iteration { get; set; }

    /// <summary>
    /// Gets or sets algorithm-specific metrics that don't fit standard categories.
    /// </summary>
    /// <value>
    /// A dictionary of custom metrics specific to the meta-learning algorithm.
    /// Examples: gradient norms, adaptation step counts, inner loop convergence metrics.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Use this for algorithm-specific monitoring and debugging.
    /// Common examples include:
    /// - "gradient_norm": Magnitude of meta-gradients
    /// - "inner_steps_actual": Average inner steps used per task
    /// - "support_accuracy": Accuracy on support sets (should be high)
    /// - "query_accuracy": Accuracy on query sets (the key metric)
    /// </para>
    /// </remarks>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();

    /// <summary>
    /// Gets or sets the time elapsed for this meta-training step in milliseconds.
    /// </summary>
    /// <value>
    /// Training time for performance monitoring and optimization.
    /// </value>
    public double TimeMs { get; set; }
}
