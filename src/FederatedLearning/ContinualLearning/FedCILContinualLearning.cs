namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Implements FedCIL — Federated Class-Incremental Learning with prototype consolidation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In class-incremental learning, new classes appear over time
/// (e.g., a spam filter encountering new spam categories). FedCIL handles this in FL by:
/// (1) maintaining class prototypes (average feature vectors per class) that are shared instead
/// of raw data, (2) using these prototypes to generate synthetic features for old classes
/// during training, preventing forgetting. This is especially important when different clients
/// see different new classes at different times.</para>
///
/// <para>Algorithm:</para>
/// <list type="number">
/// <item>When new classes appear, compute prototypes from real data</item>
/// <item>Share prototypes with server (privacy-preserving, no raw data)</item>
/// <item>Server maintains global prototype bank for all seen classes</item>
/// <item>During training, generate synthetic features from old-class prototypes</item>
/// <item>Train on both real new-class data and synthetic old-class data</item>
/// </list>
///
/// <para>Reference: Qi, D., et al. (2023). "Better Generative Replay for Continual Federated
/// Learning." CVPR 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedCILContinualLearning<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly double _prototypeDecay;
    private readonly Dictionary<int, T[]> _globalPrototypes;
    private readonly int _seed;

    /// <summary>
    /// Creates a new FedCIL strategy.
    /// </summary>
    /// <param name="prototypeDecay">EMA decay for prototype updates. Default: 0.9.</param>
    /// <param name="seed">Random seed for synthetic feature generation. Default: 42.</param>
    public FedCILContinualLearning(double prototypeDecay = 0.9, int seed = 42)
    {
        if (prototypeDecay < 0 || prototypeDecay > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(prototypeDecay), "Decay must be in [0, 1].");
        }

        _prototypeDecay = prototypeDecay;
        _globalPrototypes = new Dictionary<int, T[]>();
        _seed = seed;
    }

    /// <summary>
    /// Updates the global prototype for a class.
    /// </summary>
    /// <param name="classLabel">Class label.</param>
    /// <param name="newPrototype">Newly computed prototype from client data.</param>
    public void UpdatePrototype(int classLabel, T[] newPrototype)
    {
        Guard.NotNull(newPrototype);
        if (newPrototype.Length == 0)
        {
            throw new ArgumentException("Prototype cannot be empty.", nameof(newPrototype));
        }

        if (_globalPrototypes.TryGetValue(classLabel, out var existing))
        {
            // EMA update.
            var updated = new T[existing.Length];
            var decay = NumOps.FromDouble(_prototypeDecay);
            var oneMinusDecay = NumOps.FromDouble(1.0 - _prototypeDecay);
            for (int i = 0; i < existing.Length; i++)
            {
                updated[i] = NumOps.Add(
                    NumOps.Multiply(existing[i], decay),
                    NumOps.Multiply(newPrototype[i], oneMinusDecay));
            }

            _globalPrototypes[classLabel] = updated;
        }
        else
        {
            _globalPrototypes[classLabel] = newPrototype;
        }
    }

    /// <summary>
    /// Generates synthetic features for a class from its prototype (with noise).
    /// </summary>
    /// <param name="classLabel">Class to generate features for.</param>
    /// <param name="numSamples">Number of synthetic samples.</param>
    /// <param name="noiseScale">Scale of Gaussian noise added to prototype. Default: 0.1.</param>
    /// <returns>Synthetic feature vectors.</returns>
    public T[][] GenerateSyntheticFeatures(int classLabel, int numSamples, double noiseScale = 0.1)
    {
        if (!_globalPrototypes.TryGetValue(classLabel, out var prototype))
        {
            throw new ArgumentException($"No prototype for class {classLabel}.", nameof(classLabel));
        }

        var rng = new Random(_seed + classLabel);
        var samples = new T[numSamples][];

        for (int s = 0; s < numSamples; s++)
        {
            samples[s] = new T[prototype.Length];
            for (int i = 0; i < prototype.Length; i++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                double noise = noiseScale * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                samples[s][i] = NumOps.Add(prototype[i], NumOps.FromDouble(noise));
            }
        }

        return samples;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// FedCIL uses prototype-based replay rather than parameter importance weighting.
    /// Returns uniform importance (1.0) for all parameters. The <paramref name="taskData"/>
    /// is intentionally unused because class prototypes are maintained separately via
    /// <see cref="UpdatePrototype"/>.
    /// </remarks>
    public Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData)
    {
        Guard.NotNull(modelParameters);
        var importance = new T[modelParameters.Length];
        for (int i = 0; i < importance.Length; i++)
        {
            importance[i] = NumOps.FromDouble(1.0);
        }

        return new Vector<T>(importance);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// FedCIL uses prototype replay for anti-forgetting. The <paramref name="importanceWeights"/>
    /// are applied as uniform weights, making this a simple L2 penalty rather than
    /// importance-weighted like EWC.
    /// </remarks>
    public T ComputeRegularizationPenalty(
        Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength)
    {
        Guard.NotNull(currentParameters);
        Guard.NotNull(referenceParameters);
        Guard.NotNull(importanceWeights);

        T penalty = NumOps.Zero;
        int len = Math.Min(currentParameters.Length, referenceParameters.Length);
        for (int i = 0; i < len; i++)
        {
            var diff = NumOps.Subtract(currentParameters[i], referenceParameters[i]);
            penalty = NumOps.Add(penalty, NumOps.Multiply(diff, diff));
        }

        return NumOps.Multiply(penalty, NumOps.FromDouble(regularizationStrength * 0.5));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// FedCIL uses prototype-based synthetic replay for anti-forgetting, not gradient projection.
    /// The gradient is passed through unchanged. Use <see cref="GenerateSyntheticFeatures"/>
    /// to create replay data from class prototypes.
    /// </remarks>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        return gradient;
    }

    /// <inheritdoc/>
    public Vector<T> AggregateImportance(
        Dictionary<int, Vector<T>> clientImportances,
        Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientImportances);
        if (clientImportances.Count == 0)
        {
            throw new ArgumentException("Client importances cannot be empty.", nameof(clientImportances));
        }

        int d = clientImportances.Values.First().Length;
        var aggregated = new T[d];
        double totalWeight = clientWeights?.Values.Sum() ?? clientImportances.Count;
        if (totalWeight <= 0)
        {
            totalWeight = clientImportances.Count;
        }

        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            var wT = NumOps.FromDouble(w / totalWeight);
            for (int i = 0; i < d; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(importance[i], wT));
            }
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>Gets the known classes as a snapshot (not a live view).</summary>
    public IReadOnlyCollection<int> KnownClasses => _globalPrototypes.Keys.ToList().AsReadOnly();

    /// <summary>Gets the prototype decay rate.</summary>
    public double PrototypeDecay => _prototypeDecay;
}
