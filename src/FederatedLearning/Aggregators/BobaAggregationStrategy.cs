namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements BOBA (Bayesian Optimal Byzantine-robust Aggregation) strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Most Byzantine defenses use fixed rules (e.g., remove outliers).
/// BOBA takes a probabilistic approach using a two-class mixture model — it models client updates
/// as coming from either an "honest" distribution (tight cluster) or a "Byzantine" distribution
/// (diffuse/adversarial). It uses Expectation-Maximization (EM) to estimate which class each
/// client belongs to, then aggregates using only the honest-class posterior probabilities.</para>
///
/// <para>Two-class mixture model:</para>
/// <code>
/// p(g_k) = pi_H * N(g_k | mu_H, sigma_H^2) + pi_B * N(g_k | mu_B, sigma_B^2)
///
/// where:
///   pi_H, pi_B = mixing weights (honest vs Byzantine prior)
///   mu_H, mu_B = cluster centers
///   sigma_H, sigma_B = cluster spreads
/// </code>
///
/// <para>EM algorithm per round:</para>
/// <list type="number">
/// <item><b>E-step:</b> Compute responsibility r_k = P(honest|g_k) for each client k</item>
/// <item><b>M-step:</b> Update mixture parameters (mu, sigma, pi) from responsibilities</item>
/// <item>Repeat until convergence or max iterations</item>
/// <item>Aggregate using normalized honest responsibilities as weights</item>
/// </list>
///
/// <para>Cross-round belief propagation: posteriors from round t become priors for round t+1,
/// so persistent attackers accumulate low trust over multiple rounds.</para>
///
/// <para>Reference: BOBA: Bayesian Optimal Byzantine-robust Aggregation (2024).
/// https://arxiv.org/abs/2312.09672</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class BobaAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _priorHonest;
    private readonly int _emIterations;
    private readonly double _emConvergenceTol;
    private readonly double _minResponsibility;
    private Dictionary<int, double>? _beliefs;

    /// <summary>
    /// Initializes a new instance of the <see cref="BobaAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="priorHonest">Prior probability that a client is honest. Default: 0.8.
    /// Used to initialize the mixing weight pi_H for new clients.</param>
    /// <param name="emIterations">Maximum EM iterations per round. Default: 10.</param>
    /// <param name="emConvergenceTol">Convergence tolerance for EM log-likelihood change. Default: 1e-4.</param>
    /// <param name="minResponsibility">Minimum responsibility to avoid complete exclusion. Default: 0.01.</param>
    public BobaAggregationStrategy(
        double priorHonest = 0.8,
        int emIterations = 10,
        double emConvergenceTol = 1e-4,
        double minResponsibility = 0.01)
    {
        if (priorHonest <= 0 || priorHonest >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(priorHonest), "Prior must be in (0, 1).");
        }

        if (emIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(emIterations), "EM iterations must be at least 1.");
        }

        if (emConvergenceTol <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(emConvergenceTol), "Convergence tolerance must be positive.");
        }

        _priorHonest = priorHonest;
        _emIterations = emIterations;
        _emConvergenceTol = emConvergenceTol;
        _minResponsibility = minResponsibility;
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
        int totalParams = layerNames.Sum(ln => referenceModel[ln].Length);

        // Flatten all client models into vectors.
        var flatVectors = new double[n][];
        for (int c = 0; c < n; c++)
        {
            flatVectors[c] = new double[totalParams];
            int offset = 0;
            foreach (var layerName in layerNames)
            {
                var cp = clientModels[clientIds[c]][layerName];
                for (int i = 0; i < cp.Length; i++)
                {
                    flatVectors[c][offset++] = NumOps.ToDouble(cp[i]);
                }
            }
        }

        // Initialize beliefs for new clients.
        _beliefs ??= new Dictionary<int, double>();
        foreach (var clientId in clientIds)
        {
            if (!_beliefs.ContainsKey(clientId))
            {
                _beliefs[clientId] = _priorHonest;
            }
        }

        // Run EM to estimate honest responsibilities.
        var responsibilities = RunEM(flatVectors, clientIds, n, totalParams);

        // Store updated beliefs for cross-round propagation.
        for (int c = 0; c < n; c++)
        {
            _beliefs[clientIds[c]] = Math.Max(_minResponsibility, Math.Min(1.0 - _minResponsibility, responsibilities[c]));
        }

        // Normalize responsibilities to sum to 1 for aggregation weights.
        double respSum = 0;
        for (int c = 0; c < n; c++)
        {
            respSum += responsibilities[c];
        }

        if (respSum <= 0)
        {
            respSum = n;
            for (int c = 0; c < n; c++)
            {
                responsibilities[c] = 1.0;
            }
        }

        // Aggregate with posterior-weighted averaging.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            result[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
        }

        for (int c = 0; c < n; c++)
        {
            double w = responsibilities[c] / respSum;
            var wT = NumOps.FromDouble(w);
            var clientModel = clientModels[clientIds[c]];

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = result[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], wT));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Runs the EM algorithm to estimate honest-class responsibilities for each client.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> EM alternates between two steps:</para>
    /// <list type="bullet">
    /// <item><b>E-step:</b> Given current estimates of the honest and Byzantine distributions,
    /// compute the probability that each client is honest.</item>
    /// <item><b>M-step:</b> Given these probabilities, re-estimate the distribution parameters
    /// (center and spread of each class, and the mixing weight).</item>
    /// </list>
    /// <para>After convergence, the honest responsibilities are the aggregation weights.</para>
    /// </remarks>
    private double[] RunEM(double[][] vectors, List<int> clientIds, int n, int dim)
    {
        // Work with squared distances from the mean for efficiency.
        // This is equivalent to a 1D projection (distance to centroid) for mixture modeling.
        var mean = new double[dim];
        for (int c = 0; c < n; c++)
        {
            for (int i = 0; i < dim; i++)
            {
                mean[i] += vectors[c][i] / n;
            }
        }

        var sqDistances = new double[n];
        for (int c = 0; c < n; c++)
        {
            double sumSq = 0;
            for (int i = 0; i < dim; i++)
            {
                double diff = vectors[c][i] - mean[i];
                sumSq += diff * diff;
            }

            sqDistances[c] = sumSq;
        }

        // Initialize EM parameters using cross-round beliefs as priors.
        double piH = 0;
        for (int c = 0; c < n; c++)
        {
            piH += _beliefs![clientIds[c]];
        }

        piH /= n;
        piH = Math.Max(0.1, Math.Min(0.99, piH));
        double piB = 1.0 - piH;

        // Initialize honest variance from the closest half of clients.
        var sortedDists = sqDistances.OrderBy(d => d).ToArray();
        int halfN = Math.Max(1, n / 2);
        double sigmaHSq = sortedDists.Take(halfN).Average() + 1e-10;

        // Byzantine variance: from the farthest half (wider distribution).
        double sigmaBSq = sortedDists.Skip(halfN).DefaultIfEmpty(sigmaHSq * 4).Average() + 1e-10;
        sigmaBSq = Math.Max(sigmaBSq, sigmaHSq * 2.0); // Ensure Byzantine is wider.

        var responsibilities = new double[n];
        double prevLogLikelihood = double.NegativeInfinity;

        for (int iter = 0; iter < _emIterations; iter++)
        {
            // E-step: compute responsibilities r_k = P(honest | g_k).
            for (int c = 0; c < n; c++)
            {
                double d = sqDistances[c];

                // Log-likelihoods for numerical stability.
                double logLikH = -d / (2.0 * sigmaHSq) - 0.5 * Math.Log(sigmaHSq);
                double logLikB = -d / (2.0 * sigmaBSq) - 0.5 * Math.Log(sigmaBSq);

                // Incorporate cross-round belief as a prior multiplier.
                double beliefPrior = _beliefs![clientIds[c]];
                double logPostH = Math.Log(piH) + logLikH + Math.Log(Math.Max(beliefPrior, 1e-15));
                double logPostB = Math.Log(piB) + logLikB + Math.Log(Math.Max(1.0 - beliefPrior, 1e-15));

                // Numerically stable softmax for responsibilities.
                double maxLog = Math.Max(logPostH, logPostB);
                double expH = Math.Exp(logPostH - maxLog);
                double expB = Math.Exp(logPostB - maxLog);
                responsibilities[c] = expH / (expH + expB);

                // Clamp to avoid complete exclusion.
                responsibilities[c] = Math.Max(_minResponsibility, Math.Min(1.0 - _minResponsibility, responsibilities[c]));
            }

            // Check convergence via log-likelihood.
            double logLikelihood = 0;
            for (int c = 0; c < n; c++)
            {
                double d = sqDistances[c];
                double likH = piH * Math.Exp(-d / (2.0 * sigmaHSq)) / Math.Sqrt(sigmaHSq);
                double likB = piB * Math.Exp(-d / (2.0 * sigmaBSq)) / Math.Sqrt(sigmaBSq);
                logLikelihood += Math.Log(Math.Max(likH + likB, 1e-300));
            }

            if (Math.Abs(logLikelihood - prevLogLikelihood) < _emConvergenceTol)
            {
                break;
            }

            prevLogLikelihood = logLikelihood;

            // M-step: update parameters from responsibilities.
            double sumR = 0;
            double weightedSqH = 0;
            double weightedSqB = 0;

            for (int c = 0; c < n; c++)
            {
                sumR += responsibilities[c];
                weightedSqH += responsibilities[c] * sqDistances[c];
                weightedSqB += (1.0 - responsibilities[c]) * sqDistances[c];
            }

            piH = sumR / n;
            piH = Math.Max(0.1, Math.Min(0.99, piH));
            piB = 1.0 - piH;

            if (sumR > 1e-10)
            {
                sigmaHSq = weightedSqH / sumR + 1e-10;
            }

            double sumRB = n - sumR;
            if (sumRB > 1e-10)
            {
                sigmaBSq = weightedSqB / sumRB + 1e-10;
            }

            // Ensure Byzantine variance stays wider than honest.
            sigmaBSq = Math.Max(sigmaBSq, sigmaHSq * 1.5);
        }

        return responsibilities;
    }

    /// <summary>Gets the prior probability of a client being honest.</summary>
    public double PriorHonest => _priorHonest;

    /// <summary>Gets the maximum number of EM iterations per round.</summary>
    public int EmIterations => _emIterations;

    /// <summary>Gets the EM convergence tolerance.</summary>
    public double EmConvergenceTol => _emConvergenceTol;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"BOBA(prior={_priorHonest},EM={_emIterations})";
}
