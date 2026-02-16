namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Interface for federated adapter strategies that enable parameter-efficient fine-tuning (PEFT) in FL.
/// </summary>
/// <remarks>
/// <para>
/// Federated adapter strategies allow clients to fine-tune foundation models by training only
/// small adapter modules (LoRA, prompt tuning) rather than full model parameters. This dramatically
/// reduces communication costs and enables FL for large language models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Large AI models (like GPT or LLaMA) have billions of parameters.
/// Sending all these between clients and server is impractical. Adapters are tiny "add-on" modules
/// (often &lt;1% of total parameters) that customize the model. Only these adapters are shared
/// in federated learning, making it practical to fine-tune massive models collaboratively.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IFederatedAdapterStrategy<T>
{
    /// <summary>
    /// Extracts adapter parameters from a full model parameter vector.
    /// </summary>
    /// <param name="fullModelParameters">The complete model parameters.</param>
    /// <returns>Only the adapter (trainable) parameters.</returns>
    Vector<T> ExtractAdapterParameters(Vector<T> fullModelParameters);

    /// <summary>
    /// Merges aggregated adapter parameters back into the full model.
    /// </summary>
    /// <param name="fullModelParameters">The client's full model parameters (frozen base + local adapters).</param>
    /// <param name="aggregatedAdapters">The globally aggregated adapter parameters from the server.</param>
    /// <returns>Updated full model parameters with new adapters applied.</returns>
    Vector<T> MergeAdapterParameters(Vector<T> fullModelParameters, Vector<T> aggregatedAdapters);

    /// <summary>
    /// Aggregates adapter parameters from multiple clients.
    /// </summary>
    /// <param name="clientAdapters">Dictionary of client ID to adapter parameter vectors.</param>
    /// <param name="clientWeights">Optional per-client weights.</param>
    /// <returns>The aggregated global adapter parameters.</returns>
    Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights);

    /// <summary>
    /// Gets the total number of trainable adapter parameters.
    /// </summary>
    int AdapterParameterCount { get; }

    /// <summary>
    /// Gets the compression ratio (adapter params / total model params).
    /// </summary>
    double CompressionRatio { get; }
}
