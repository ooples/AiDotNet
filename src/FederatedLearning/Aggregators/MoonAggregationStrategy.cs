namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements MOON (Model-COntrastive Learning) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MOON corrects "local drift" by adding a contrastive loss
/// during client training. The contrastive loss pulls the local model's representation closer
/// to the global model and pushes it away from the previous local model, reducing divergence
/// caused by non-IID data.</para>
///
/// <para>During aggregation, MOON uses standard weighted averaging (same as FedAvg). The key
/// innovation is in the local training objective, which includes:</para>
/// <code>L_total = L_task + mu * L_contrastive(z_local, z_global, z_prev_local)</code>
///
/// <para>The contrastive loss uses cosine similarity scaled by temperature:</para>
/// <code>
/// L_con = -log( exp(sim(z_local, z_global) / tau) /
///              (exp(sim(z_local, z_global) / tau) + exp(sim(z_local, z_prev) / tau)) )
/// </code>
///
/// <para>Reference: Li, Q., He, B., and Song, D. (2021). "Model-Contrastive Federated Learning."
/// CVPR 2021.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class MoonAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _contrastiveWeight;
    private readonly double _temperature;
    private readonly Dictionary<int, Vector<T>> _previousRepresentations = [];

    /// <summary>
    /// Initializes a new instance of the <see cref="MoonAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="contrastiveWeight">Weight of the contrastive loss term (mu). Default: 1.0 per paper.</param>
    /// <param name="temperature">Temperature for contrastive similarity. Default: 0.5 per paper.</param>
    public MoonAggregationStrategy(double contrastiveWeight = 1.0, double temperature = 0.5)
    {
        if (contrastiveWeight < 0)
        {
            throw new ArgumentException("Contrastive weight must be non-negative.", nameof(contrastiveWeight));
        }

        if (temperature <= 0)
        {
            throw new ArgumentException("Temperature must be positive.", nameof(temperature));
        }

        _contrastiveWeight = contrastiveWeight;
        _temperature = temperature;
    }

    /// <summary>
    /// Aggregates client models using weighted averaging.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The server-side aggregation in MOON is identical to FedAvg.
    /// The contrastive loss is applied during local training on each client via
    /// <see cref="ComputeContrastiveLoss"/>.</para>
    /// </remarks>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Computes the MOON contrastive loss for a client during local training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This loss encourages the local model's representations to stay
    /// close to the global model (positive pair) while staying different from the previous
    /// round's local model (negative pair). This prevents local models from drifting too far
    /// from the global consensus.</para>
    /// </remarks>
    /// <param name="localRepresentation">Current local model's feature representation z_local.</param>
    /// <param name="globalRepresentation">Global model's feature representation z_global (positive pair).</param>
    /// <param name="previousLocalRepresentation">Previous round's local representation z_prev (negative pair).
    /// If null, only the positive similarity is used (first round behavior).</param>
    /// <returns>The contrastive loss value, scaled by <see cref="ContrastiveWeight"/>.</returns>
    public T ComputeContrastiveLoss(
        Vector<T> localRepresentation,
        Vector<T> globalRepresentation,
        Vector<T>? previousLocalRepresentation)
    {
        double simPositive = CosineSimilarity(localRepresentation, globalRepresentation);
        double logitPositive = simPositive / _temperature;

        double loss;

        if (previousLocalRepresentation != null)
        {
            double simNegative = CosineSimilarity(localRepresentation, previousLocalRepresentation);
            double logitNegative = simNegative / _temperature;

            // Numerically stable log-sum-exp: L = -log(exp(pos) / (exp(pos) + exp(neg)))
            //   = -pos + log(exp(pos) + exp(neg))
            //   = -pos + max(pos,neg) + log(exp(pos-max) + exp(neg-max))
            double maxLogit = Math.Max(logitPositive, logitNegative);
            double logSumExp = maxLogit + Math.Log(
                Math.Exp(logitPositive - maxLogit) + Math.Exp(logitNegative - maxLogit));

            loss = -logitPositive + logSumExp;
        }
        else
        {
            // First round: no previous representation. Loss is zero (no negative pair).
            loss = 0.0;
        }

        return NumOps.FromDouble(_contrastiveWeight * loss);
    }

    /// <summary>
    /// Stores the current local representation for a client to be used as the negative pair
    /// in the next round.
    /// </summary>
    /// <param name="clientId">The client identifier.</param>
    /// <param name="representation">The client's current local representation vector.</param>
    public void StoreRepresentation(int clientId, Vector<T> representation)
    {
        _previousRepresentations[clientId] = representation.Clone();
    }

    /// <summary>
    /// Retrieves the previous round's local representation for a client.
    /// </summary>
    /// <param name="clientId">The client identifier.</param>
    /// <returns>The stored representation, or null if this is the first round for this client.</returns>
    public Vector<T>? GetPreviousRepresentation(int clientId)
    {
        return _previousRepresentations.TryGetValue(clientId, out var rep) ? rep : null;
    }

    /// <summary>
    /// Computes the complete local training loss including both task loss and contrastive loss.
    /// </summary>
    /// <param name="taskLoss">The base task loss (e.g., cross-entropy).</param>
    /// <param name="localRepresentation">Current local model's feature representation.</param>
    /// <param name="globalRepresentation">Global model's feature representation.</param>
    /// <param name="clientId">The client identifier (to look up stored previous representation).</param>
    /// <returns>L_total = L_task + mu * L_contrastive.</returns>
    public T ComputeTotalLoss(
        T taskLoss,
        Vector<T> localRepresentation,
        Vector<T> globalRepresentation,
        int clientId)
    {
        var previousRep = GetPreviousRepresentation(clientId);
        var contrastiveLoss = ComputeContrastiveLoss(localRepresentation, globalRepresentation, previousRep);
        return NumOps.Add(taskLoss, contrastiveLoss);
    }

    private static double CosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vectors must have the same length for cosine similarity.");
        }

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double va = NumOps.ToDouble(a[i]);
            double vb = NumOps.ToDouble(b[i]);
            dot += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 1e-10 ? dot / denom : 0.0;
    }

    /// <summary>Gets the contrastive loss weight (mu).</summary>
    public double ContrastiveWeight => _contrastiveWeight;

    /// <summary>Gets the contrastive temperature parameter.</summary>
    public double Temperature => _temperature;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"MOON(\u03bc={_contrastiveWeight},\u03c4={_temperature})";
}
