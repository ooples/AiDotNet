namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Implements FedKD — Knowledge Distillation-based communication for federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of sending model parameters (which can be huge for LLMs),
/// clients send soft predictions (logits) on a shared public dataset. The server trains a global
/// model by distilling knowledge from these aggregated soft labels. This enables FL even when
/// clients have different model architectures (heterogeneous FL), since predictions are
/// architecture-agnostic.</para>
///
/// <para>Algorithm:</para>
/// <list type="number">
/// <item>Server broadcasts a small public unlabeled dataset to all clients</item>
/// <item>Each client runs inference on the public data and sends logits</item>
/// <item>Server aggregates logits (weighted average)</item>
/// <item>Server trains global model on aggregated soft labels (KD loss)</item>
/// </list>
///
/// <para>Reference: Wu, C., et al. (2022). "Communication-Efficient Federated Learning via
/// Knowledge Distillation." NeurIPS 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedKDCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _temperature;
    private readonly double _kdWeight;

    /// <summary>
    /// Creates a new FedKD compressor.
    /// </summary>
    /// <param name="temperature">Softmax temperature for knowledge distillation. Default: 3.0.</param>
    /// <param name="kdWeight">Weight of KD loss vs. hard label loss. Default: 0.7.</param>
    public FedKDCompressor(double temperature = 3.0, double kdWeight = 0.7)
    {
        if (temperature <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive.");
        }

        if (kdWeight < 0 || kdWeight > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(kdWeight), "KD weight must be in [0, 1].");
        }

        _temperature = temperature;
        _kdWeight = kdWeight;
    }

    /// <summary>
    /// Aggregates soft predictions from multiple clients.
    /// </summary>
    /// <param name="clientLogits">Client logits: clientId → array of prediction vectors per sample.</param>
    /// <param name="clientWeights">Per-client weights.</param>
    /// <returns>Aggregated soft labels per sample.</returns>
    public T[][] AggregateLogits(
        Dictionary<int, T[][]> clientLogits,
        Dictionary<int, double> clientWeights)
    {
        Guard.NotNull(clientLogits);
        Guard.NotNull(clientWeights);
        if (clientLogits.Count == 0)
        {
            throw new ArgumentException("No client logits provided.", nameof(clientLogits));
        }

        var firstClient = clientLogits.First().Value;
        int numSamples = firstClient.Length;
        int numClasses = firstClient[0].Length;
        double totalWeight = clientWeights.Values.Sum();

        var aggregated = new T[numSamples][];
        for (int s = 0; s < numSamples; s++)
        {
            aggregated[s] = new T[numClasses];
            for (int c = 0; c < numClasses; c++)
            {
                aggregated[s][c] = NumOps.Zero;
            }

            foreach (var (clientId, logits) in clientLogits)
            {
                double w = clientWeights.GetValueOrDefault(clientId, 1.0) / totalWeight;
                var wT = NumOps.FromDouble(w);

                for (int c = 0; c < numClasses; c++)
                {
                    aggregated[s][c] = NumOps.Add(
                        aggregated[s][c],
                        NumOps.Multiply(logits[s][c], wT));
                }
            }
        }

        return aggregated;
    }

    /// <summary>
    /// Computes KD loss between student logits and teacher soft labels.
    /// </summary>
    /// <param name="studentLogits">Student model's logits.</param>
    /// <param name="teacherLogits">Aggregated teacher soft labels.</param>
    /// <returns>KL divergence loss scaled by T².</returns>
    public T ComputeKDLoss(T[] studentLogits, T[] teacherLogits)
    {
        Guard.NotNull(studentLogits);
        Guard.NotNull(teacherLogits);
        if (studentLogits.Length != teacherLogits.Length)
        {
            throw new ArgumentException(
                $"Logit arrays must have equal length. Got student={studentLogits.Length}, teacher={teacherLogits.Length}.");
        }

        int n = studentLogits.Length;
        var studentSoft = Softmax(studentLogits, _temperature);
        var teacherSoft = Softmax(teacherLogits, _temperature);

        double kl = 0;
        for (int i = 0; i < n; i++)
        {
            if (teacherSoft[i] > 1e-10)
            {
                kl += teacherSoft[i] * Math.Log(teacherSoft[i] / Math.Max(studentSoft[i], 1e-10));
            }
        }

        return NumOps.FromDouble(kl * _temperature * _temperature * _kdWeight);
    }

    private double[] Softmax(T[] logits, double temperature)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > max) max = v;
        }

        var exps = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            exps[i] = Math.Exp((NumOps.ToDouble(logits[i]) - max) / temperature);
            sum += exps[i];
        }

        for (int i = 0; i < exps.Length; i++)
        {
            exps[i] /= sum;
        }

        return exps;
    }

    /// <summary>
    /// Performs one server-side distillation step: updates the student model parameters
    /// to match the aggregated ensemble soft labels via gradient descent on the KD loss.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After collecting logits from all clients, the server has
    /// "soft labels" — probability distributions that capture the collective knowledge of all
    /// client models. The server then trains its own model to match these soft labels using
    /// gradient descent. This is how the global model learns without ever seeing the raw data.</para>
    /// <para><b>Implementation Note:</b> Gradients are currently approximated via finite differences
    /// (perturbing each sampled parameter by epsilon and measuring loss change). This is a
    /// reference implementation suitable for small models and testing. Production systems should
    /// replace this with backpropagation through the student model for O(1) gradient computation
    /// per parameter instead of the current O(n) forward passes per sampled parameter.</para>
    /// </remarks>
    /// <param name="studentParams">Current student model parameters (will be updated).</param>
    /// <param name="aggregatedSoftLabels">Soft labels from AggregateLogits.</param>
    /// <param name="studentLogitsFn">Function that computes student logits from parameters and sample index.</param>
    /// <param name="learningRate">Server-side learning rate. Default: 0.01.</param>
    /// <param name="steps">Number of gradient descent steps. Default: 10.</param>
    /// <returns>Updated student parameters and the final average KD loss.</returns>
    public (T[] UpdatedParams, double FinalLoss) ServerDistillationStep(
        T[] studentParams,
        T[][] aggregatedSoftLabels,
        Func<T[], int, T[]> studentLogitsFn,
        double learningRate = 0.01,
        int steps = 10)
    {
        var currentParams = (T[])studentParams.Clone();
        double lastLoss = 0;

        for (int step = 0; step < steps; step++)
        {
            var totalGrad = new double[currentParams.Length];
            double totalLoss = 0;

            for (int s = 0; s < aggregatedSoftLabels.Length; s++)
            {
                var studentLogits = studentLogitsFn(currentParams, s);
                var loss = ComputeKDLoss(studentLogits, aggregatedSoftLabels[s]);
                totalLoss += NumOps.ToDouble(loss);

                // Approximate gradient via finite differences on each parameter.
                // In a real system, this would use backpropagation.
                double epsilon = 1e-5;
                int gradSamples = Math.Min(currentParams.Length, 50); // Subsample for efficiency.
                var rng = new Random(step * aggregatedSoftLabels.Length + s);

                for (int g = 0; g < gradSamples; g++)
                {
                    int idx = rng.Next(currentParams.Length);
                    var saved = currentParams[idx];

                    currentParams[idx] = NumOps.Add(saved, NumOps.FromDouble(epsilon));
                    var lossPlus = NumOps.ToDouble(ComputeKDLoss(studentLogitsFn(currentParams, s), aggregatedSoftLabels[s]));

                    currentParams[idx] = saved;
                    totalGrad[idx] += (lossPlus - NumOps.ToDouble(loss)) / epsilon;
                }
            }

            // Apply gradient descent.
            double invSamples = 1.0 / aggregatedSoftLabels.Length;
            for (int i = 0; i < currentParams.Length; i++)
            {
                double grad = totalGrad[i] * invSamples;
                currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.FromDouble(learningRate * grad));
            }

            lastLoss = totalLoss / aggregatedSoftLabels.Length;
        }

        return (currentParams, lastLoss);
    }

    /// <summary>
    /// Handles heterogeneous client architectures by padding/truncating logits to a common dimension.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When clients have different model architectures, they may produce
    /// different numbers of output classes. This method normalizes all client logits to the same
    /// dimension by padding smaller outputs with zeros or truncating larger ones.</para>
    /// </remarks>
    /// <param name="clientLogits">Raw logits from heterogeneous clients (may have different class counts).</param>
    /// <param name="targetClasses">Target number of output classes for aggregation.</param>
    /// <returns>Normalized logits with uniform dimensions.</returns>
    public Dictionary<int, T[][]> NormalizeHeterogeneousLogits(
        Dictionary<int, T[][]> clientLogits,
        int targetClasses)
    {
        Guard.NotNull(clientLogits);
        if (targetClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetClasses), "Target classes must be positive.");
        }

        var normalized = new Dictionary<int, T[][]>();

        foreach (var (clientId, logits) in clientLogits)
        {
            var clientNorm = new T[logits.Length][];
            for (int s = 0; s < logits.Length; s++)
            {
                var sample = new T[targetClasses];
                int copyLen = Math.Min(logits[s].Length, targetClasses);
                for (int c = 0; c < copyLen; c++)
                {
                    sample[c] = logits[s][c];
                }

                // Remaining positions are zero (default).
                clientNorm[s] = sample;
            }

            normalized[clientId] = clientNorm;
        }

        return normalized;
    }

    /// <summary>Gets the KD temperature.</summary>
    public double Temperature => _temperature;

    /// <summary>Gets the KD loss weight.</summary>
    public double KDWeight => _kdWeight;
}
