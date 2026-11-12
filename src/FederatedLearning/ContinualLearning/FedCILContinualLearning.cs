using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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
            if (existing.Length != newPrototype.Length)
            {
                throw new ArgumentException(
                    $"Prototype length {newPrototype.Length} does not match the stored length {existing.Length} for class {classLabel}.",
                    nameof(newPrototype));
            }

            // EMA update.
            var updated = Engine.Add(
                Engine.Multiply(new Vector<T>(existing), NumOps.FromDouble(_prototypeDecay)),
                Engine.Multiply(new Vector<T>(newPrototype), NumOps.FromDouble(1.0 - _prototypeDecay)));
            _globalPrototypes[classLabel] = updated.ToArray();
        }
        else
        {
            // Store a copy so later caller edits cannot change the global prototype.
            _globalPrototypes[classLabel] = (T[])newPrototype.Clone();
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
        if (numSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSamples), "Number of samples must be positive.");
        }

        if (noiseScale < 0 || double.IsNaN(noiseScale) || double.IsInfinity(noiseScale))
        {
            throw new ArgumentOutOfRangeException(nameof(noiseScale), "Noise scale must be non-negative and finite.");
        }

        if (!_globalPrototypes.TryGetValue(classLabel, out var prototype))
        {
            throw new ArgumentException($"No prototype for class {classLabel}.", nameof(classLabel));
        }

        var rng = RandomHelper.CreateSeededRandom(_seed + classLabel);
        int dim = prototype.Length;

        // Box-Muller draws stay in the original (sample, feature) order of the seeded generator.
        var noise = new double[numSamples * dim];
        for (int k = 0; k < noise.Length; k++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            noise[k] = noiseScale * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        var noiseTensor = new Tensor<double>(noise, new[] { numSamples, dim }).Cast<T>();
        var prototypeRow = new Tensor<T>((T[])prototype.Clone(), new[] { 1, dim });
        var batch = Engine.TensorAdd(noiseTensor, Engine.TensorBroadcastTo(prototypeRow, new[] { numSamples, dim })).ToArray();

        var samples = new T[numSamples][];
        for (int s = 0; s < numSamples; s++)
        {
            samples[s] = new T[dim];
            Array.Copy(batch, s * dim, samples[s], 0, dim);
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
        return Engine.TensorOnesLike(modelParameters.AsTensor()).ToVector();
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

        int len = Math.Min(currentParameters.Length, referenceParameters.Length);
        if (len == 0)
        {
            return NumOps.Zero;
        }

        var current = currentParameters.Length == len ? currentParameters : currentParameters.GetSubVector(0, len);
        var reference = referenceParameters.Length == len ? referenceParameters : referenceParameters.GetSubVector(0, len);
        var diff = Engine.Subtract(current, reference);
        T penalty = Engine.DotProduct(diff, diff);

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
        var aggregated = new Vector<T>(d);
        // Normalize over the weights actually applied: participating clients, with 1.0 for a missing entry.
        double totalWeight = clientImportances.Keys.Sum(id => clientWeights?.GetValueOrDefault(id, 1.0) ?? 1.0);
        bool uniform = totalWeight <= 0;
        if (uniform)
        {
            totalWeight = clientImportances.Count;
        }

        foreach (var (clientId, importance) in clientImportances)
        {
            double w = uniform ? 1.0 : clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            var clientImportance = importance.Length == d ? importance : importance.GetSubVector(0, d);
            aggregated = Engine.Add(aggregated, Engine.Multiply(clientImportance, NumOps.FromDouble(w / totalWeight)));
        }

        return aggregated;
    }

    /// <summary>Gets the known classes as a snapshot (not a live view).</summary>
    public IReadOnlyCollection<int> KnownClasses => _globalPrototypes.Keys.ToList().AsReadOnly();

    /// <summary>Gets the prototype decay rate.</summary>
    public double PrototypeDecay => _prototypeDecay;
}
