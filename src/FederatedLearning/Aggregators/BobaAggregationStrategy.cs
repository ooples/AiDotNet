namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements BOBA (Bayesian Optimal Byzantine-robust Aggregation) strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Most Byzantine defenses use fixed rules (e.g., remove outliers).
/// BOBA takes a probabilistic approach — it maintains a belief (probability) for each client
/// about whether they are honest or malicious. Each round, it updates these beliefs using
/// Bayesian inference based on how consistent each client's update is with the majority.
/// The aggregation weight for each client is then proportional to its probability of being honest.</para>
///
/// <para>Belief update:</para>
/// <code>
/// likelihood_k = exp(-||g_k - g_mean||² / (2 * sigma²))
/// posterior_k ∝ prior_k * likelihood_k
/// w_k = posterior_k / sum(posterior_j)
/// </code>
///
/// <para>Reference: BOBA: Bayesian Optimal Byzantine-robust Aggregation (2024).
/// https://arxiv.org/abs/2312.09672</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class BobaAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _priorHonest;
    private readonly double _likelihoodScale;
    private Dictionary<int, double>? _beliefs;

    /// <summary>
    /// Initializes a new instance of the <see cref="BobaAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="priorHonest">Prior probability that a client is honest. Default: 0.8.</param>
    /// <param name="likelihoodScale">Scale (sigma²) for the Gaussian likelihood.
    /// Smaller values make the filter more aggressive. Default: 1.0.</param>
    public BobaAggregationStrategy(double priorHonest = 0.8, double likelihoodScale = 1.0)
    {
        if (priorHonest <= 0 || priorHonest >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(priorHonest), "Prior must be in (0, 1).");
        }

        if (likelihoodScale <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(likelihoodScale), "Likelihood scale must be positive.");
        }

        _priorHonest = priorHonest;
        _likelihoodScale = likelihoodScale;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        if (clientModels.Count == 1)
        {
            return clientModels.First().Value;
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();
        var clientIds = clientModels.Keys.ToList();
        int n = clientIds.Count;

        // Initialize beliefs if first round or new clients appear.
        _beliefs ??= new Dictionary<int, double>();
        foreach (var clientId in clientIds)
        {
            if (!_beliefs.ContainsKey(clientId))
            {
                _beliefs[clientId] = _priorHonest;
            }
        }

        // Compute the mean update (unweighted, as a reference direction).
        int totalParams = layerNames.Sum(ln => referenceModel[ln].Length);
        var meanFlat = new double[totalParams];

        for (int c = 0; c < n; c++)
        {
            int offset = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModels[clientIds[c]][layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    meanFlat[offset++] += NumOps.ToDouble(cp[i]) / n;
                }
            }
        }

        // Compute squared distance of each client from the mean and update beliefs.
        var distances = new double[n];
        for (int c = 0; c < n; c++)
        {
            int offset = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModels[clientIds[c]][layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    double diff = NumOps.ToDouble(cp[i]) - meanFlat[offset++];
                    distances[c] += diff * diff;
                }
            }
        }

        // Bayesian update: posterior ∝ prior * likelihood.
        var posteriors = new double[n];
        for (int c = 0; c < n; c++)
        {
            double prior = _beliefs[clientIds[c]];
            double likelihood = Math.Exp(-distances[c] / (2.0 * _likelihoodScale));
            posteriors[c] = prior * likelihood;
        }

        // Normalize posteriors.
        double posteriorSum = posteriors.Sum();
        if (posteriorSum <= 0)
        {
            // Fallback to uniform.
            for (int c = 0; c < n; c++)
            {
                posteriors[c] = 1.0 / n;
            }

            posteriorSum = 1.0;
        }
        else
        {
            for (int c = 0; c < n; c++)
            {
                posteriors[c] /= posteriorSum;
            }
        }

        // Store updated beliefs for next round.
        for (int c = 0; c < n; c++)
        {
            _beliefs[clientIds[c]] = posteriors[c] * n; // Re-scale to ~prior range for next round.
            _beliefs[clientIds[c]] = Math.Max(0.01, Math.Min(0.99, _beliefs[clientIds[c]]));
        }

        // Aggregate with posterior weights.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        for (int c = 0; c < n; c++)
        {
            var pw = NumOps.FromDouble(posteriors[c]);
            var clientModel = clientModels[clientIds[c]];

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], pw));
                }
            }
        }

        return result;
    }

    /// <summary>Gets the prior probability of a client being honest.</summary>
    public double PriorHonest => _priorHonest;

    /// <summary>Gets the Gaussian likelihood scale parameter.</summary>
    public double LikelihoodScale => _likelihoodScale;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"BOBA(prior={_priorHonest},σ²={_likelihoodScale})";
}
