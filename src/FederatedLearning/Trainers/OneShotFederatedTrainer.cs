namespace AiDotNet.FederatedLearning.Trainers;

/// <summary>
/// Implements One-Shot Federated Learning — a single communication round from clients to server.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard FL requires many communication rounds between clients and
/// the server (often 100-1000). One-Shot FL reduces this to just one round: each client trains a
/// local model independently, sends it to the server, and the server combines them into a single
/// global model using ensemble distillation or model averaging. This is much cheaper in terms of
/// communication, but trades off some accuracy. It's ideal for cross-silo settings where clients
/// have enough data to train reasonable local models.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// 1. Server sends initial model to all clients
/// 2. Each client trains independently for many epochs (one-time)
/// 3. Clients send final models to server
/// 4. Server aggregates: either average, or distill an ensemble
/// </code>
///
/// <para>Reference: Guha, N., et al. (2019). "One-Shot Federated Learning."
/// arXiv:1902.11175. Practical variants: Li et al. (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
internal class OneShotFederatedTrainer<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _localEpochs;
    private readonly OneShotAggregationMode _aggregationMode;
    private readonly double _distillationTemperature;
    private readonly int _distillationSteps;

    /// <summary>
    /// Creates a new One-Shot FL trainer.
    /// </summary>
    /// <param name="localEpochs">Number of local training epochs per client. Default: 50.</param>
    /// <param name="aggregationMode">How to combine client models. Default: WeightedAverage.</param>
    /// <param name="distillationTemperature">Temperature for ensemble distillation. Default: 3.0.</param>
    /// <param name="distillationSteps">Number of distillation training steps. Default: 500.</param>
    public OneShotFederatedTrainer(
        int localEpochs = 50,
        OneShotAggregationMode aggregationMode = OneShotAggregationMode.WeightedAverage,
        double distillationTemperature = 3.0,
        int distillationSteps = 500)
    {
        if (localEpochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(localEpochs), "Must have at least 1 local epoch.");
        }

        if (distillationTemperature <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(distillationTemperature), "Temperature must be positive.");
        }

        if (distillationSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(distillationSteps), "Must have at least 1 distillation step.");
        }

        _localEpochs = localEpochs;
        _aggregationMode = aggregationMode;
        _distillationTemperature = distillationTemperature;
        _distillationSteps = distillationSteps;
    }

    /// <summary>
    /// Aggregates client models in one shot using the configured aggregation mode.
    /// </summary>
    /// <param name="clientModels">Locally-trained models from each client.</param>
    /// <param name="clientSampleCounts">Number of training samples per client.</param>
    /// <returns>The aggregated global model parameters.</returns>
    public Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, int> clientSampleCounts)
    {
        Guard.NotNull(clientModels);
        Guard.NotNull(clientSampleCounts);
        if (clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be empty.", nameof(clientModels));
        }

        LastEnsembleDiversity = 0;

        return _aggregationMode switch
        {
            OneShotAggregationMode.WeightedAverage => AggregateWeightedAverage(clientModels, clientSampleCounts),
            OneShotAggregationMode.UniformAverage => AggregateUniform(clientModels),
            OneShotAggregationMode.EnsembleDistillation => AggregateEnsembleDistillation(clientModels, clientSampleCounts),
            _ => AggregateWeightedAverage(clientModels, clientSampleCounts)
        };
    }

    private Dictionary<string, T[]> AggregateWeightedAverage(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, int> clientSampleCounts)
    {
        double totalSamples = clientSampleCounts.Values.Sum();
        var result = new Dictionary<string, T[]>();

        // Use first client as template for layer structure.
        var template = clientModels.Values.First();

        foreach (var (layerName, layerParams) in template)
        {
            var merged = new double[layerParams.Length];

            foreach (var (clientId, model) in clientModels)
            {
                double weight = totalSamples > 0
                    ? clientSampleCounts.GetValueOrDefault(clientId, 1) / totalSamples
                    : 1.0 / clientModels.Count;

                if (model.TryGetValue(layerName, out var clientLayer))
                {
                    if (clientLayer.Length != merged.Length)
                    {
                        throw new ArgumentException(
                            $"Client {clientId} layer '{layerName}' length {clientLayer.Length} differs from expected {merged.Length}.");
                    }

                    for (int i = 0; i < clientLayer.Length; i++)
                    {
                        merged[i] += weight * NumOps.ToDouble(clientLayer[i]);
                    }
                }
            }

            var mergedT = new T[layerParams.Length];
            for (int i = 0; i < mergedT.Length; i++)
            {
                mergedT[i] = NumOps.FromDouble(merged[i]);
            }

            result[layerName] = mergedT;
        }

        return result;
    }

    private Dictionary<string, T[]> AggregateUniform(
        Dictionary<int, Dictionary<string, T[]>> clientModels)
    {
        var sampleCounts = clientModels.ToDictionary(kvp => kvp.Key, _ => 1);
        return AggregateWeightedAverage(clientModels, sampleCounts);
    }

    /// <remarks>
    /// Returns a weighted average as the initialization point for ensemble distillation.
    /// The actual distillation loop (using <see cref="ComputeEnsembleSoftLabels"/> and
    /// <see cref="ComputeDistillationLoss"/>) is performed by the caller, since it
    /// requires a training dataset and student model that are not available at aggregation time.
    /// This method also computes ensemble diversity to assess client model disagreement.
    /// </remarks>
    private Dictionary<string, T[]> AggregateEnsembleDistillation(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, int> clientSampleCounts)
    {
        // Ensemble distillation initialization: weighted average as starting point.
        var initial = AggregateWeightedAverage(clientModels, clientSampleCounts);

        // Compute ensemble diversity metric.
        double totalDiversity = 0;
        int layerCount = 0;

        foreach (var (layerName, globalLayer) in initial)
        {
            double layerVariance = 0;
            int count = 0;

            foreach (var (_, model) in clientModels)
            {
                if (model.TryGetValue(layerName, out var clientLayer))
                {
                    int len = Math.Min(clientLayer.Length, globalLayer.Length);
                    for (int i = 0; i < len; i++)
                    {
                        double diff = NumOps.ToDouble(clientLayer[i]) - NumOps.ToDouble(globalLayer[i]);
                        layerVariance += diff * diff;
                        count++;
                    }
                }
            }

            if (count > 0)
            {
                totalDiversity += layerVariance / count;
                layerCount++;
            }
        }

        LastEnsembleDiversity = layerCount > 0 ? totalDiversity / layerCount : 0;

        return initial;
    }

    /// <summary>
    /// Computes the soft-label ensemble prediction by averaging softmax outputs from all client models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each client model produces a set of "logits" (raw prediction scores)
    /// for a given input. By softening these with temperature and averaging across all client models,
    /// we get a "soft label" that captures the collective knowledge of the entire ensemble. A student
    /// model can then be trained to match these soft labels, effectively distilling the ensemble's
    /// knowledge into a single model.</para>
    /// </remarks>
    /// <param name="clientLogits">Dictionary of client ID to logits array for a single sample.</param>
    /// <param name="clientWeights">Optional per-client weights (by sample count).</param>
    /// <returns>Soft-label probabilities averaged over the ensemble.</returns>
    public double[] ComputeEnsembleSoftLabels(
        Dictionary<int, double[]> clientLogits,
        Dictionary<int, double>? clientWeights = null)
    {
        Guard.NotNull(clientLogits);
        if (clientLogits.Count == 0)
        {
            throw new ArgumentException("Client logits cannot be empty.", nameof(clientLogits));
        }

        int numClasses = clientLogits.Values.First().Length;
        var ensembleProbs = new double[numClasses];
        double totalWeight = 0;

        foreach (var (clientId, logits) in clientLogits)
        {
            if (logits.Length != numClasses)
            {
                throw new ArgumentException(
                    $"Client {clientId} logits length {logits.Length} differs from expected {numClasses}.");
            }
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            // Compute temperature-scaled softmax for this client.
            double maxLogit = double.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
            {
                double scaled = logits[i] / _distillationTemperature;
                if (scaled > maxLogit) maxLogit = scaled;
            }

            double expSum = 0;
            var probs = new double[numClasses];
            for (int i = 0; i < numClasses; i++)
            {
                probs[i] = Math.Exp(logits[i] / _distillationTemperature - maxLogit);
                expSum += probs[i];
            }

            for (int i = 0; i < numClasses; i++)
            {
                ensembleProbs[i] += w * (probs[i] / expSum);
            }
        }

        // Normalize by total weight.
        if (totalWeight > 0)
        {
            for (int i = 0; i < numClasses; i++)
            {
                ensembleProbs[i] /= totalWeight;
            }
        }

        return ensembleProbs;
    }

    /// <summary>
    /// Computes the knowledge distillation loss: KL divergence between ensemble soft labels
    /// and student soft predictions, scaled by temperature squared.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The distillation loss measures how well the student model
    /// matches the ensemble's predictions. Temperature scaling makes the distributions softer,
    /// revealing more information about inter-class relationships. The T² scaling ensures the
    /// gradient magnitude is independent of temperature.</para>
    /// </remarks>
    /// <param name="ensembleSoftLabels">Soft labels from the ensemble (probabilities).</param>
    /// <param name="studentLogits">Raw logits from the student model.</param>
    /// <returns>Distillation loss: T² * KL(ensemble || student).</returns>
    public double ComputeDistillationLoss(double[] ensembleSoftLabels, double[] studentLogits)
    {
        int n = ensembleSoftLabels.Length;
        if (n != studentLogits.Length)
        {
            throw new ArgumentException("Ensemble and student must have same number of classes.");
        }

        // Compute temperature-scaled softmax for student.
        double maxLogit = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            double scaled = studentLogits[i] / _distillationTemperature;
            if (scaled > maxLogit) maxLogit = scaled;
        }

        double expSum = 0;
        var studentProbs = new double[n];
        for (int i = 0; i < n; i++)
        {
            studentProbs[i] = Math.Exp(studentLogits[i] / _distillationTemperature - maxLogit);
            expSum += studentProbs[i];
        }

        for (int i = 0; i < n; i++)
        {
            studentProbs[i] /= expSum;
        }

        // KL(ensemble || student).
        double kl = 0;
        for (int i = 0; i < n; i++)
        {
            if (ensembleSoftLabels[i] > 1e-10)
            {
                kl += ensembleSoftLabels[i] * Math.Log(ensembleSoftLabels[i] / Math.Max(studentProbs[i], 1e-10));
            }
        }

        return kl * _distillationTemperature * _distillationTemperature;
    }

    /// <summary>
    /// Computes the combined one-shot distillation loss: alpha * KD_loss + (1-alpha) * task_loss.
    /// </summary>
    /// <param name="taskLoss">Hard-label task loss (cross-entropy on real data).</param>
    /// <param name="ensembleSoftLabels">Soft labels from ensemble.</param>
    /// <param name="studentLogits">Student model logits.</param>
    /// <param name="alpha">Interpolation weight between task and distillation loss. Default: 0.7.</param>
    /// <returns>Combined loss for the distillation step.</returns>
    public double ComputeCombinedLoss(
        double taskLoss,
        double[] ensembleSoftLabels,
        double[] studentLogits,
        double alpha = 0.7)
    {
        if (alpha < 0 || alpha > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be in [0, 1].");
        }

        double kdLoss = ComputeDistillationLoss(ensembleSoftLabels, studentLogits);
        return alpha * kdLoss + (1.0 - alpha) * taskLoss;
    }

    /// <summary>
    /// Gets the ensemble diversity from the last distillation aggregation.
    /// Higher values indicate more disagreement between client models.
    /// </summary>
    public double LastEnsembleDiversity { get; private set; }

    /// <summary>Gets the configured number of local epochs.</summary>
    public int LocalEpochs => _localEpochs;

    /// <summary>Gets the aggregation mode.</summary>
    public OneShotAggregationMode AggregationMode => _aggregationMode;

    /// <summary>Gets the distillation temperature.</summary>
    public double DistillationTemperature => _distillationTemperature;

    /// <summary>Gets the number of distillation steps.</summary>
    public int DistillationSteps => _distillationSteps;
}

/// <summary>
/// Aggregation modes for One-Shot Federated Learning.
/// </summary>
public enum OneShotAggregationMode
{
    /// <summary>Weighted average by sample count (FedAvg-style).</summary>
    WeightedAverage = 0,

    /// <summary>Uniform average (equal weight per client).</summary>
    UniformAverage = 1,

    /// <summary>Ensemble distillation (average + diversity tracking).</summary>
    EnsembleDistillation = 2
}
