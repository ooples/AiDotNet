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
    public double ComputeKDLoss(double[] studentLogits, double[] teacherLogits)
    {
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

        return kl * _temperature * _temperature * _kdWeight;
    }

    private static double[] Softmax(double[] logits, double temperature)
    {
        double max = logits.Max();
        var exps = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            exps[i] = Math.Exp((logits[i] - max) / temperature);
            sum += exps[i];
        }

        for (int i = 0; i < exps.Length; i++)
        {
            exps[i] /= sum;
        }

        return exps;
    }

    /// <summary>Gets the KD temperature.</summary>
    public double Temperature => _temperature;

    /// <summary>Gets the KD loss weight.</summary>
    public double KDWeight => _kdWeight;
}
