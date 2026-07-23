namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Interface for federated continual learning strategies that prevent catastrophic forgetting.
/// </summary>
/// <remarks>
/// <para>
/// In production FL, the data distribution changes over time (new tasks, seasonal shifts,
/// evolving user behavior). Without continual learning, the global model catastrophically
/// forgets previously learned knowledge when trained on new data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine a spam filter that learns about new types of spam each month.
/// Without continual learning, training on November's spam makes it forget how to catch
/// October's spam. Federated continual learning lets the global model learn new patterns
/// while remembering old ones, even though different clients see different evolving data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IFederatedContinualLearningStrategy<T>
{
    /// <summary>
    /// Computes the importance (Fisher information) of each parameter based on current task data.
    /// </summary>
    /// <param name="modelParameters">Current model parameters.</param>
    /// <param name="taskData">Training data for the current task.</param>
    /// <returns>Per-parameter importance scores.</returns>
    Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData);

    /// <summary>
    /// Applies the continual learning regularization to prevent forgetting.
    /// </summary>
    /// <param name="currentParameters">Current model parameters being updated.</param>
    /// <param name="referenceParameters">Parameters from the previous task (anchor).</param>
    /// <param name="importanceWeights">Per-parameter importance from previous tasks.</param>
    /// <param name="regularizationStrength">Lambda — controls forgetting prevention strength.</param>
    /// <returns>Regularization penalty to add to the loss.</returns>
    T ComputeRegularizationPenalty(Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength);

    /// <summary>
    /// Projects the gradient to be orthogonal to previously important directions.
    /// </summary>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="importanceWeights">Per-parameter importance from previous tasks.</param>
    /// <returns>The projected gradient that doesn't interfere with previous knowledge.</returns>
    Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights);

    /// <summary>
    /// Merges importance weights from multiple clients into a global importance estimate.
    /// </summary>
    /// <param name="clientImportances">Per-client importance weight vectors.</param>
    /// <param name="clientWeights">Optional per-client aggregation weights.</param>
    /// <returns>The aggregated global importance weights.</returns>
    Vector<T> AggregateImportance(Dictionary<int, Vector<T>> clientImportances, Dictionary<int, double>? clientWeights);
}
