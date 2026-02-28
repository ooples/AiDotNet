namespace AiDotNet.FederatedLearning.Privacy;

/// <summary>
/// Implements Shuffle Model Differential Privacy for federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In standard local DP, each client adds a lot of noise to their
/// update before sending it (because the server sees individual updates). In shuffle model DP,
/// a trusted shuffler randomly permutes the updates before the server sees them. Because the
/// server can't link updates to specific clients, each client needs to add much less noise —
/// achieving central-DP-level accuracy with local-DP trust assumptions.</para>
///
/// <para>Privacy amplification by shuffling:</para>
/// <code>
/// Local DP: epsilon_local per client
/// After shuffling n clients:
///   epsilon_central ≈ (1 - 1/n) * ln(1 + (e^epsilon_local - 1) / n) + epsilon_local / n
/// (Balle-Bell-Gascon 2019 tighter bound)
///
/// For large n and moderate epsilon_local:
///   epsilon_central ≈ epsilon_local * sqrt(ln(1/delta) / n)
/// </code>
///
/// <para>Protocol:</para>
/// <list type="number">
/// <item>Each client applies local DP noise (Gaussian mechanism) to their update</item>
/// <item>The shuffler collects all noisy updates and randomly permutes them (Fisher-Yates)</item>
/// <item>The server receives anonymized, shuffled updates — cannot link to clients</item>
/// <item>Privacy amplification gives central-DP guarantees from local-DP noise levels</item>
/// </list>
///
/// <para>Reference: Balle, B., Bell, J., &amp; Gascon, A. (2019). "The Privacy Blanket of the
/// Shuffle Model." Crypto 2019.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class ShuffleModelDP<T> : PrivacyMechanismBase<Dictionary<string, T[]>, T>
{
    private readonly double _localEpsilon;
    private readonly int _seed;
    private int _roundCounter;
    private double _totalBudgetConsumed;

    /// <summary>
    /// Creates a new Shuffle Model DP mechanism.
    /// </summary>
    /// <param name="localEpsilon">Per-client local DP epsilon. Default: 8.0 (high local, amplified by shuffle).</param>
    /// <param name="seed">Random seed. Default: 42.</param>
    public ShuffleModelDP(double localEpsilon = 8.0, int seed = 42)
    {
        if (localEpsilon <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(localEpsilon), "Epsilon must be positive.");
        }

        _localEpsilon = localEpsilon;
        _seed = seed;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Applies local DP noise to a single client's model update. This is the per-client step.
    /// The noise scale is calibrated for the local epsilon, and the privacy amplification
    /// comes from the shuffle step (see <see cref="ShuffleAndAnonymize"/>).
    /// </remarks>
    public override Dictionary<string, T[]> ApplyPrivacy(
        Dictionary<string, T[]> model, double epsilon, double delta)
    {
        Guard.NotNull(model);
        if (epsilon <= 0 || double.IsNaN(epsilon) || double.IsInfinity(epsilon))
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be a positive finite value.");
        }

        if (delta < 0 || delta >= 1 || double.IsNaN(delta))
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in [0, 1).");
        }

        var rng = new Random(_seed + _roundCounter++);
        double effectiveEpsilon = Math.Min(epsilon, _localEpsilon);

        // Gaussian mechanism: sigma = sqrt(2 * ln(1.25/delta)) / epsilon.
        double sigma = delta > 0
            ? Math.Sqrt(2.0 * Math.Log(1.25 / delta)) / effectiveEpsilon
            : 1.0 / effectiveEpsilon;

        // Compute global sensitivity (L2 norm of the update) for calibration.
        double l2Norm = 0;
        foreach (var kvp in model)
        {
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                double v = NumOps.ToDouble(kvp.Value[i]);
                l2Norm += v * v;
            }
        }

        l2Norm = Math.Sqrt(l2Norm);
        double noiseScale = sigma * Math.Max(l2Norm, 1e-10);

        var privatized = new Dictionary<string, T[]>(model.Count);
        foreach (var kvp in model)
        {
            var noisy = new T[kvp.Value.Length];
            for (int i = 0; i < noisy.Length; i++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                double noise = noiseScale * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                noisy[i] = NumOps.Add(kvp.Value[i], NumOps.FromDouble(noise));
            }

            privatized[kvp.Key] = noisy;
        }

        _totalBudgetConsumed += effectiveEpsilon;
        return privatized;
    }

    /// <summary>
    /// Shuffles a collection of client updates, removing the association between client IDs
    /// and their updates. This is the core of the shuffle model — after shuffling, the server
    /// sees a bag of anonymized updates and cannot determine which client sent which update.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Think of this like putting everyone's answers in a hat and
    /// drawing them out in random order. Nobody can tell who wrote which answer. This anonymity
    /// is what lets each person add less noise — the crowd provides privacy protection.</para>
    ///
    /// <para>Uses Fisher-Yates shuffle for uniform random permutation with O(n) time.</para>
    /// </remarks>
    /// <param name="clientUpdates">Dictionary mapping client IDs to their (already locally noised) updates.</param>
    /// <returns>A list of anonymized updates in random order (client IDs removed).</returns>
    public List<Dictionary<string, T[]>> ShuffleAndAnonymize(
        Dictionary<int, Dictionary<string, T[]>> clientUpdates)
    {
        if (clientUpdates == null || clientUpdates.Count == 0)
        {
            throw new ArgumentException("Client updates cannot be null or empty.", nameof(clientUpdates));
        }

        // Collect all updates into a list (stripping client IDs).
        var updates = clientUpdates.Values.ToList();
        int n = updates.Count;

        // Fisher-Yates shuffle for uniform random permutation.
        var rng = new Random(_seed + _roundCounter);
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (updates[i], updates[j]) = (updates[j], updates[i]);
        }

        return updates;
    }

    /// <summary>
    /// Applies local DP noise to each client's update and then shuffles the collection.
    /// This is the complete shuffle-model DP pipeline for a single round.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method does both steps: (1) each client adds noise to
    /// their update for local DP protection, then (2) all the noisy updates are shuffled so
    /// the server can't link them back to clients. The combination gives privacy amplification —
    /// much better privacy than either step alone.</para>
    /// </remarks>
    /// <param name="clientUpdates">Dictionary mapping client IDs to their raw (un-noised) updates.</param>
    /// <param name="delta">Target delta for the (epsilon, delta)-DP guarantee.</param>
    /// <returns>A shuffled list of anonymized, locally-noised updates.</returns>
    public List<Dictionary<string, T[]>> ApplyLocalDPAndShuffle(
        Dictionary<int, Dictionary<string, T[]>> clientUpdates,
        double delta = 1e-5)
    {
        Guard.NotNull(clientUpdates);
        if (clientUpdates.Count == 0)
        {
            throw new ArgumentException("Client updates cannot be empty.", nameof(clientUpdates));
        }

        if (delta < 0 || delta >= 1 || double.IsNaN(delta))
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in [0, 1).");
        }

        // Step 1: Apply local DP noise to each client's update independently.
        var noisedUpdates = new Dictionary<int, Dictionary<string, T[]>>(clientUpdates.Count);
        foreach (var (clientId, update) in clientUpdates)
        {
            noisedUpdates[clientId] = ApplyPrivacy(update, _localEpsilon, delta);
        }

        // Step 2: Shuffle and anonymize.
        return ShuffleAndAnonymize(noisedUpdates);
    }

    /// <summary>
    /// Aggregates shuffled, anonymized updates by simple averaging.
    /// </summary>
    /// <remarks>
    /// <para>The server receives a list of anonymized updates (no client IDs) and averages them.
    /// This is the standard aggregation step after the shuffle model DP pipeline.</para>
    /// </remarks>
    /// <param name="shuffledUpdates">List of anonymized updates from <see cref="ShuffleAndAnonymize"/>.</param>
    /// <returns>Averaged model update.</returns>
    public Dictionary<string, T[]> AggregateShuffled(List<Dictionary<string, T[]>> shuffledUpdates)
    {
        if (shuffledUpdates == null || shuffledUpdates.Count == 0)
        {
            throw new ArgumentException("Shuffled updates cannot be null or empty.", nameof(shuffledUpdates));
        }

        var reference = shuffledUpdates[0];
        var result = new Dictionary<string, T[]>(reference.Count);
        int n = shuffledUpdates.Count;
        var weight = NumOps.FromDouble(1.0 / n);

        foreach (var layerName in reference.Keys)
        {
            int layerLen = reference[layerName].Length;
            var aggregated = new T[layerLen];

            for (int i = 0; i < layerLen; i++)
            {
                aggregated[i] = NumOps.Zero;
            }

            foreach (var update in shuffledUpdates)
            {
                if (!update.TryGetValue(layerName, out var layerData))
                {
                    continue;
                }

                for (int i = 0; i < Math.Min(layerLen, layerData.Length); i++)
                {
                    aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(layerData[i], weight));
                }
            }

            result[layerName] = aggregated;
        }

        return result;
    }

    /// <summary>
    /// Computes the effective central epsilon after shuffling n clients using the
    /// tight bound from Balle, Bell, and Gascon (2019).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you the actual privacy level the server gets.
    /// Even though each client uses epsilon_local (which might be large like 8.0), the shuffle
    /// gives the server a much smaller effective epsilon (better privacy). More clients = more
    /// amplification = better privacy.</para>
    ///
    /// <para>Tight bound: epsilon_central = (1 - 1/n) * ln(1 + (e^eps_local - 1)/n) + eps_local/n</para>
    /// <para>Asymptotic: epsilon_central ~ epsilon_local * sqrt(ln(1/delta) / n) for large n</para>
    /// </remarks>
    /// <param name="numClients">Number of clients participating in the round.</param>
    /// <param name="delta">Target delta for DP guarantee.</param>
    /// <returns>Effective central epsilon after shuffling.</returns>
    public double ComputeEffectiveEpsilon(int numClients, double delta)
    {
        if (numClients < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numClients), "Must have at least 1 client.");
        }

        if (delta < 0 || delta >= 1 || double.IsNaN(delta))
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in [0, 1).");
        }

        if (numClients <= 1)
        {
            return _localEpsilon;
        }

        double n = numClients;

        // Tight bound from Balle-Bell-Gascon 2019 (pure DP case).
        double expEps = Math.Exp(_localEpsilon);
        double tightBound = (1.0 - 1.0 / n) * Math.Log(1.0 + (expEps - 1.0) / n) + _localEpsilon / n;

        // Asymptotic bound (approximate DP case, includes delta).
        double asymptotic = delta > 0
            ? _localEpsilon * Math.Sqrt(Math.Log(1.0 / delta) / n)
            : tightBound;

        // Return the tighter (smaller) of the two bounds.
        return Math.Min(tightBound, asymptotic);
    }

    /// <summary>
    /// Computes the minimum number of clients needed to achieve a target central epsilon.
    /// </summary>
    /// <param name="targetEpsilon">Desired central epsilon after amplification.</param>
    /// <param name="delta">Target delta for DP guarantee.</param>
    /// <returns>Minimum number of clients required.</returns>
    public int ComputeMinClientsNeeded(double targetEpsilon, double delta)
    {
        if (targetEpsilon <= 0 || double.IsNaN(targetEpsilon) || double.IsInfinity(targetEpsilon))
        {
            throw new ArgumentOutOfRangeException(nameof(targetEpsilon), "Target epsilon must be positive and finite.");
        }

        if (delta <= 0 || delta >= 1 || double.IsNaN(delta))
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in (0, 1).");
        }

        if (targetEpsilon >= _localEpsilon)
        {
            return 1;
        }

        // From asymptotic bound: n >= epsilon_local^2 * ln(1/delta) / target_epsilon^2.
        double n = _localEpsilon * _localEpsilon * Math.Log(1.0 / delta) / (targetEpsilon * targetEpsilon);
        return (int)Math.Ceiling(Math.Max(n, 2));
    }

    /// <inheritdoc/>
    public override double GetPrivacyBudgetConsumed() => _totalBudgetConsumed;

    /// <inheritdoc/>
    public override string GetMechanismName() => $"ShuffleDP(\u03b5_local={_localEpsilon})";

    /// <summary>Gets the local epsilon value.</summary>
    public double LocalEpsilon => _localEpsilon;
}
