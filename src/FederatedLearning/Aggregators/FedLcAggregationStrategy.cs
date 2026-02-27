namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedLC (Logit Calibration) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When different clients have different class distributions
/// (e.g., Hospital A sees mostly flu, Hospital B sees mostly cold), local models develop
/// biased predictions. FedLC fixes this by adjusting each client's logits based on its
/// local class frequency before aggregation.</para>
///
/// <para>During local training, logits are calibrated:</para>
/// <code>z_calibrated[c] = z[c] - tau * log(p_local[c])</code>
/// <para>where p_local[c] is the local class frequency for class c. This counteracts the bias
/// introduced by the imbalanced local data distribution.</para>
///
/// <para>Reference: Zhang, J., et al. (2022). "Federated Learning with Label Distribution Skew
/// via Logits Calibration." ICML 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedLcAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _calibrationTemperature;
    private readonly Dictionary<int, Vector<T>> _clientClassDistributions = [];

    /// <summary>
    /// Initializes a new instance of the <see cref="FedLcAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="calibrationTemperature">Temperature for logit calibration (tau). Default: 1.0 per paper.</param>
    public FedLcAggregationStrategy(double calibrationTemperature = 1.0)
    {
        if (calibrationTemperature <= 0)
        {
            throw new ArgumentException("Calibration temperature must be positive.", nameof(calibrationTemperature));
        }

        _calibrationTemperature = calibrationTemperature;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return AggregateWeightedAverage(clientModels, clientWeights);
    }

    /// <summary>
    /// Registers a client's local class distribution for use in logit calibration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before local training, each client reports how many samples
    /// it has per class (as a probability distribution). This is used to calibrate the logits
    /// during training so the model isn't biased toward classes that are over-represented locally.</para>
    /// </remarks>
    /// <param name="clientId">The client identifier.</param>
    /// <param name="classDistribution">Probability distribution over classes (must sum to ~1.0).
    /// Each element p[c] is the fraction of local data belonging to class c.</param>
    public void RegisterClassDistribution(int clientId, Vector<T> classDistribution)
    {
        // Validate it's a probability distribution.
        double sum = 0;
        for (int i = 0; i < classDistribution.Length; i++)
        {
            double v = NumOps.ToDouble(classDistribution[i]);
            if (v < 0)
            {
                throw new ArgumentException(
                    $"Class distribution must be non-negative, but got {v} at index {i}.",
                    nameof(classDistribution));
            }

            sum += v;
        }

        if (Math.Abs(sum - 1.0) > 0.01)
        {
            throw new ArgumentException(
                $"Class distribution must sum to ~1.0, but sums to {sum}.",
                nameof(classDistribution));
        }

        _clientClassDistributions[clientId] = classDistribution.Clone();
    }

    /// <summary>
    /// Calibrates raw logits using the client's local class distribution.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If a client has 90% flu and 10% cold samples, the model
    /// naturally predicts flu more often. This method subtracts a correction term based on
    /// the class frequency, effectively removing the bias so the model makes predictions
    /// based on actual features rather than local data imbalance.</para>
    /// </remarks>
    /// <param name="logits">Raw model logits for a single sample (one per class).</param>
    /// <param name="clientId">The client identifier (to look up registered distribution).</param>
    /// <returns>Calibrated logits: z_cal[c] = z[c] - tau * log(p_local[c]).</returns>
    public Vector<T> CalibrateLogits(Vector<T> logits, int clientId)
    {
        if (!_clientClassDistributions.TryGetValue(clientId, out var distribution))
        {
            throw new InvalidOperationException(
                $"No class distribution registered for client {clientId}. Call RegisterClassDistribution first.");
        }

        if (logits.Length != distribution.Length)
        {
            throw new ArgumentException(
                $"Logits length ({logits.Length}) must match distribution length ({distribution.Length}).");
        }

        var calibrated = new T[logits.Length];
        for (int c = 0; c < logits.Length; c++)
        {
            double p = NumOps.ToDouble(distribution[c]);
            // Clamp to avoid log(0).
            double logP = Math.Log(Math.Max(p, 1e-10));
            double correction = _calibrationTemperature * logP;
            calibrated[c] = NumOps.Subtract(logits[c], NumOps.FromDouble(correction));
        }

        return new Vector<T>(calibrated);
    }

    /// <summary>
    /// Calibrates a batch of logits for a client.
    /// </summary>
    /// <param name="logitsBatch">Logits matrix (rows = samples, columns = classes).</param>
    /// <param name="clientId">The client identifier.</param>
    /// <returns>Calibrated logits matrix.</returns>
    public Matrix<T> CalibrateBatch(Matrix<T> logitsBatch, int clientId)
    {
        if (!_clientClassDistributions.TryGetValue(clientId, out var distribution))
        {
            throw new InvalidOperationException(
                $"No class distribution registered for client {clientId}.");
        }

        if (logitsBatch.Columns != distribution.Length)
        {
            throw new ArgumentException(
                $"Logits columns ({logitsBatch.Columns}) must match distribution length ({distribution.Length}).");
        }

        // Precompute the calibration correction vector (same for all samples).
        var correction = new T[distribution.Length];
        for (int c = 0; c < distribution.Length; c++)
        {
            double p = NumOps.ToDouble(distribution[c]);
            correction[c] = NumOps.FromDouble(_calibrationTemperature * Math.Log(Math.Max(p, 1e-10)));
        }

        var result = new Matrix<T>(logitsBatch.Rows, logitsBatch.Columns);
        for (int s = 0; s < logitsBatch.Rows; s++)
        {
            for (int c = 0; c < logitsBatch.Columns; c++)
            {
                result[s, c] = NumOps.Subtract(logitsBatch[s, c], correction[c]);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes class distribution from sample labels (convenience helper).
    /// </summary>
    /// <param name="labels">Class label indices for all local samples.</param>
    /// <param name="numClasses">Total number of classes.</param>
    /// <returns>Probability distribution vector of length numClasses.</returns>
    public static Vector<T> ComputeClassDistribution(int[] labels, int numClasses)
    {
        if (labels.Length == 0)
        {
            throw new ArgumentException("Labels array must not be empty.", nameof(labels));
        }

        var counts = new double[numClasses];
        foreach (int label in labels)
        {
            if (label < 0 || label >= numClasses)
            {
                throw new ArgumentOutOfRangeException(nameof(labels),
                    $"Label {label} is out of range [0, {numClasses}).");
            }

            counts[label]++;
        }

        var distribution = new T[numClasses];
        double total = labels.Length;
        for (int c = 0; c < numClasses; c++)
        {
            distribution[c] = NumOps.FromDouble(counts[c] / total);
        }

        return new Vector<T>(distribution);
    }

    /// <summary>Gets the logit calibration temperature (tau).</summary>
    public double CalibrationTemperature => _calibrationTemperature;

    /// <summary>Gets the number of clients with registered distributions.</summary>
    public int RegisteredClientCount => _clientClassDistributions.Count;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedLC(\u03c4={_calibrationTemperature})";
}
