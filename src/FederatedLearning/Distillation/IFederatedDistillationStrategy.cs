namespace AiDotNet.FederatedLearning.Distillation;

/// <summary>
/// Interface for federated knowledge distillation strategies that enable model-heterogeneous FL.
/// </summary>
/// <remarks>
/// <para>
/// In standard FL, all clients must use the same model architecture. Federated knowledge
/// distillation removes this constraint by exchanging knowledge (logits, prototypes, or
/// generated samples) instead of raw model parameters.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine different hospitals using different AI models (some simple,
/// some complex). Knowledge distillation lets them share what they've learned without requiring
/// everyone to use the same model. The "knowledge" is transferred through predictions or
/// summaries rather than model weights.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IFederatedDistillationStrategy<T>
{
    /// <summary>
    /// Generates the knowledge representation to send to the server (logits, prototypes, etc.).
    /// </summary>
    /// <param name="localModelParameters">The client's local model parameters.</param>
    /// <param name="publicData">Optional public dataset for generating logits.</param>
    /// <returns>The knowledge representation as a matrix (rows = samples, cols = output dims).</returns>
    Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData);

    /// <summary>
    /// Aggregates knowledge from multiple clients into a global knowledge representation.
    /// </summary>
    /// <param name="clientKnowledge">Dictionary of client ID to knowledge matrices.</param>
    /// <param name="clientWeights">Optional per-client weights for weighted aggregation.</param>
    /// <returns>The aggregated global knowledge representation.</returns>
    Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights);

    /// <summary>
    /// Applies the aggregated global knowledge to update a local model.
    /// </summary>
    /// <param name="localModelParameters">The client's current model parameters.</param>
    /// <param name="globalKnowledge">The aggregated global knowledge from the server.</param>
    /// <param name="publicData">Optional public dataset used for distillation.</param>
    /// <param name="temperature">Softmax temperature for soft label generation.</param>
    /// <returns>The updated local model parameters.</returns>
    Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge, Matrix<T>? publicData, double temperature);
}
