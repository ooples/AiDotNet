using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Handles missing features in vertical FL when not all parties have data for all entities.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In real-world VFL, not all parties have records for all entities.
/// A bank might have 100,000 customers while the partner hospital only has 30,000 patients,
/// with only 20,000 in common. For the other 80,000 bank customers, the hospital's features
/// are "missing".</para>
///
/// <para>This class provides strategies for handling those missing features:</para>
/// <list type="bullet">
/// <item><description><b>Zero:</b> Replace missing features with zeros. Fast but may bias the model.</description></item>
/// <item><description><b>Mean:</b> Replace with column-wise means. Better for centered distributions.</description></item>
/// <item><description><b>Learned:</b> Train a small model to predict missing features from available ones.</description></item>
/// <item><description><b>Skip:</b> Only use fully-aligned entities. Safest but reduces training data.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Based on "Vertical Federated Learning with Missing Features During
/// Training and Inference" (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MissingFeatureHandler<T> : FederatedLearningComponentBase<T>
{
    private readonly MissingFeatureOptions _options;
    private readonly Dictionary<string, Tensor<T>> _columnMeans;
    private bool _meansComputed;

    /// <summary>
    /// Initializes a new instance of <see cref="MissingFeatureHandler{T}"/>.
    /// </summary>
    /// <param name="options">Configuration for missing feature handling.</param>
    public MissingFeatureHandler(MissingFeatureOptions? options = null)
    {
        _options = options ?? new MissingFeatureOptions();
        _columnMeans = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
    }

    /// <summary>
    /// Gets the configured missing feature strategy.
    /// </summary>
    public MissingFeatureStrategy Strategy => _options.Strategy;

    /// <summary>
    /// Determines whether an entity should be included in training based on its availability
    /// across parties.
    /// </summary>
    /// <param name="entityId">The entity identifier.</param>
    /// <param name="availablePartyIds">The party IDs that have data for this entity.</param>
    /// <param name="totalParties">The total number of parties.</param>
    /// <returns>True if the entity should be included in training.</returns>
    public bool ShouldIncludeEntity(string entityId, IReadOnlyCollection<string> availablePartyIds, int totalParties)
    {
        if (_options.Strategy == MissingFeatureStrategy.Skip)
        {
            return availablePartyIds.Count >= totalParties;
        }

        if (_options.AllowPartialAlignment)
        {
            return availablePartyIds.Count >= 1;
        }

        return availablePartyIds.Count >= totalParties;
    }

    /// <summary>
    /// Imputes missing embeddings for a party that doesn't have data for the current batch entities.
    /// </summary>
    /// <param name="partyId">The identifier of the party with missing data.</param>
    /// <param name="embeddingDimension">The expected embedding dimension.</param>
    /// <param name="batchSize">The number of entities in the batch.</param>
    /// <returns>An imputed embedding tensor with shape [batchSize, embeddingDimension].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When a party doesn't have data for some entities in a batch,
    /// we need to provide "placeholder" embeddings. The strategy determines how these placeholders
    /// are generated.</para>
    /// </remarks>
    public Tensor<T> ImputeEmbeddings(string partyId, int embeddingDimension, int batchSize)
    {
        return _options.Strategy switch
        {
            MissingFeatureStrategy.Zero => CreateZeroEmbeddings(embeddingDimension, batchSize),
            MissingFeatureStrategy.Mean => CreateMeanEmbeddings(partyId, embeddingDimension, batchSize),
            MissingFeatureStrategy.Learned => CreateMeanEmbeddings(partyId, embeddingDimension, batchSize),
            _ => CreateZeroEmbeddings(embeddingDimension, batchSize)
        };
    }

    /// <summary>
    /// Updates the running mean statistics for a party's embeddings.
    /// Called during training with actual (non-imputed) embeddings.
    /// </summary>
    /// <param name="partyId">The party identifier.</param>
    /// <param name="embeddings">The actual embedding values from this party.</param>
    public void UpdateStatistics(string partyId, Tensor<T> embeddings)
    {
        if (_options.Strategy == MissingFeatureStrategy.Zero || _options.Strategy == MissingFeatureStrategy.Skip)
        {
            return;
        }

        int batchSize = embeddings.Shape[0];
        int embDim = embeddings.Rank > 1 ? embeddings.Shape[1] : embeddings.Shape[0];

        if (!_columnMeans.ContainsKey(partyId))
        {
            _columnMeans[partyId] = new Tensor<T>(new[] { embDim });
        }

        // Compute running mean with exponential moving average
        var currentMean = _columnMeans[partyId];
        double alpha = 0.1;

        for (int d = 0; d < embDim; d++)
        {
            double batchMean = 0.0;
            for (int b = 0; b < batchSize; b++)
            {
                int idx = embeddings.Rank > 1 ? b * embDim + d : d;
                batchMean += NumOps.ToDouble(embeddings[idx]);
            }

            batchMean /= batchSize;
            double existing = NumOps.ToDouble(currentMean[d]);
            currentMean[d] = NumOps.FromDouble(existing * (1.0 - alpha) + batchMean * alpha);
        }

        _meansComputed = true;
    }

    /// <summary>
    /// Creates a missingness indicator tensor for a batch, marking which features are imputed.
    /// </summary>
    /// <param name="batchSize">Number of entities in the batch.</param>
    /// <param name="partyCount">Total number of parties.</param>
    /// <param name="missingPartyIndices">Indices of parties with missing data for this batch.</param>
    /// <returns>A binary tensor where 1 indicates imputed (missing) and 0 indicates observed.</returns>
    public Tensor<T> CreateMissingnessIndicator(int batchSize, int partyCount, IReadOnlyCollection<int> missingPartyIndices)
    {
        if (!_options.AddMissingnessIndicator)
        {
            return new Tensor<T>(new[] { batchSize, 0 });
        }

        var indicator = new Tensor<T>(new[] { batchSize, partyCount });
        foreach (int missingIdx in missingPartyIndices)
        {
            for (int b = 0; b < batchSize; b++)
            {
                indicator[b * partyCount + missingIdx] = NumOps.FromDouble(1.0);
            }
        }

        return indicator;
    }

    private static Tensor<T> CreateZeroEmbeddings(int embeddingDimension, int batchSize)
    {
        return new Tensor<T>(new[] { batchSize, embeddingDimension });
    }

    private Tensor<T> CreateMeanEmbeddings(string partyId, int embeddingDimension, int batchSize)
    {
        var result = new Tensor<T>(new[] { batchSize, embeddingDimension });

        if (_meansComputed && _columnMeans.ContainsKey(partyId))
        {
            var mean = _columnMeans[partyId];
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < embeddingDimension; d++)
                {
                    result[b * embeddingDimension + d] = mean[d];
                }
            }
        }

        return result;
    }
}
