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
/// <para>Privacy amplification:</para>
/// <code>
/// Local DP: epsilon_local per client
/// After shuffling n clients: epsilon_central ≈ epsilon_local * sqrt(ln(1/delta) / n)
/// </code>
///
/// <para>Reference: Balle, B., et al. (2019). "Privacy Amplification by Shuffling."
/// Crypto 2019.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class ShuffleModelDP<T> : PrivacyMechanismBase<Dictionary<string, T[]>, T>
{
    private readonly double _localEpsilon;
    private readonly int _seed;
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
    public override Dictionary<string, T[]> ApplyPrivacy(
        Dictionary<string, T[]> model, double epsilon, double delta)
    {
        var rng = new Random(_seed + (int)(_totalBudgetConsumed * 1000));
        double noiseScale = 1.0 / Math.Min(epsilon, _localEpsilon);

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

        _totalBudgetConsumed += epsilon;
        return privatized;
    }

    /// <summary>
    /// Computes the effective central epsilon after shuffling n clients.
    /// </summary>
    /// <param name="numClients">Number of clients participating in the round.</param>
    /// <param name="delta">Target delta for DP guarantee.</param>
    /// <returns>Effective central epsilon.</returns>
    public double ComputeEffectiveEpsilon(int numClients, double delta)
    {
        if (numClients <= 1)
        {
            return _localEpsilon;
        }

        return _localEpsilon * Math.Sqrt(Math.Log(1.0 / delta) / numClients);
    }

    /// <inheritdoc/>
    public override double GetPrivacyBudgetConsumed() => _totalBudgetConsumed;

    /// <inheritdoc/>
    public override string GetMechanismName() => $"ShuffleDP(ε_local={_localEpsilon})";

    /// <summary>Gets the local epsilon value.</summary>
    public double LocalEpsilon => _localEpsilon;
}
