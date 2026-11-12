using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Federated Prompt Tuning — soft prompt aggregation for foundation model personalization.
/// </summary>
/// <remarks>
/// <para>
/// Prompt tuning (Lester et al., 2021) prepends learnable "soft prompt" tokens to the input.
/// In federated settings, only these prompt embeddings (typically 10-100 tokens × embedding dim)
/// are communicated, offering even higher compression than LoRA for very large models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of modifying the model itself, prompt tuning adds a few
/// learned "instructions" in front of each input. These instructions tell the model how
/// to adapt to the specific task. In federated prompt tuning, each device learns its own
/// instructions and they're combined at the server — only a few thousand parameters need
/// to be shared, even for billion-parameter models.
/// </para>
/// <para>
/// References:
/// Lester et al. (2021), "The Power of Scale for Parameter-Efficient Prompt Tuning".
/// Zhao et al. (2024), "FedPSF-LLM: Dual Prompt Personalization for Federated Foundation Models".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FederatedPromptTuning<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly int _numPromptTokens;
    private readonly int _embeddingDim;
    private readonly int _modelDim;

    /// <inheritdoc/>
    public int AdapterParameterCount => _numPromptTokens * _embeddingDim;

    /// <inheritdoc/>
    public double CompressionRatio => _modelDim > 0 ? (double)AdapterParameterCount / _modelDim : 0;

    /// <summary>
    /// Creates a new federated prompt tuning strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="numPromptTokens">Number of soft prompt tokens. Default: 20.</param>
    /// <param name="embeddingDim">Embedding dimension per token. Default: 768.</param>
    public FederatedPromptTuning(int modelDim, int numPromptTokens = 20, int embeddingDim = 768)
    {
        _modelDim = modelDim;
        _numPromptTokens = numPromptTokens;
        _embeddingDim = embeddingDim;
    }

    /// <inheritdoc/>
    public Vector<T> ExtractAdapterParameters(Vector<T> fullModelParameters)
    {
        // Prompt parameters stored at the beginning of the parameter vector
        int promptSize = AdapterParameterCount;
        int actualSize = Math.Min(promptSize, fullModelParameters.Length);

        var promptParams = new Vector<T>(actualSize);
        fullModelParameters.AsSpan().Slice(0, actualSize).CopyTo(promptParams.AsWritableSpan());
        return promptParams;
    }

    /// <inheritdoc/>
    public Vector<T> MergeAdapterParameters(Vector<T> fullModelParameters, Vector<T> aggregatedAdapters)
    {
        int promptSize = aggregatedAdapters.Length;

        // Replace prompt parameters at the start; keep base model parameters unchanged.
        var merged = fullModelParameters.Clone();
        aggregatedAdapters.AsSpan().CopyTo(merged.AsWritableSpan().Slice(0, promptSize));
        return merged;
    }

    /// <inheritdoc/>
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        if (clientAdapters.Count == 0)
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));

        int promptSize = clientAdapters.Values.First().Length;
        var aggregated = new Vector<T>(promptSize);
        double totalWeight = 0;

        foreach (var (clientId, prompts) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            // Only the first promptSize entries of a longer prompt vector take part.
            var clientPrompt = prompts.Length == promptSize ? prompts : prompts.GetSubVector(0, promptSize);
            aggregated = Engine.Add(aggregated, Engine.Multiply(clientPrompt, NumOps.FromDouble(w)));
        }

        return Engine.Multiply(aggregated, NumOps.FromDouble(1.0 / totalWeight));
    }
}
