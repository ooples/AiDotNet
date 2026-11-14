using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedLC (Logit Calibration) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When different clients have different class distributions
/// (e.g., Hospital A sees mostly flu, Hospital B sees mostly cold), local models develop
/// biased predictions. FedLC fixes this by adjusting each client's logits based on its
/// local class counts during local training.</para>
///
/// <para>During local training, logits are calibrated:</para>
/// <code>z_calibrated[c] = z[c] - tau * n_local[c]^(-1/4)</code>
/// <para>where n_local[c] is the number of local samples of class c. This counteracts the bias
/// introduced by the imbalanced local data distribution.</para>
///
/// <para>Reference: Zhang, J., et al. (2022). "Federated Learning with Label Distribution Skew
/// via Logits Calibration." ICML 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedLcAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _calibrationTemperature;
    private readonly object _distributionLock = new();
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
    /// Registers a client's local per-class sample counts for use in logit calibration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before local training, each client reports how many samples
    /// it has per class. This is used to calibrate the logits during training so the model
    /// isn't biased toward classes that are over-represented locally.</para>
    /// </remarks>
    /// <param name="clientId">The client identifier.</param>
    /// <param name="classCounts">Number of local samples n[c] in each class c (non-negative).</param>
    public void RegisterClassCounts(int clientId, Vector<T> classCounts)
    {
        Guard.NotNull(classCounts);
        if (classCounts.Length == 0)
        {
            throw new ArgumentException("Class counts must not be empty.", nameof(classCounts));
        }

        var smallest = Engine.TensorMinValue(classCounts.AsTensor());
        if (NumOps.IsNaN(smallest) || NumOps.LessThan(smallest, NumOps.Zero))
        {
            throw new ArgumentException(
                $"Class counts must be non-negative, but the smallest is {smallest}.",
                nameof(classCounts));
        }

        lock (_distributionLock)
        {
            _clientClassDistributions[clientId] = classCounts.Clone();
        }
    }

    /// <summary>
    /// Removes a client's registered class counts.
    /// </summary>
    /// <param name="clientId">The client identifier.</param>
    /// <returns>True if the client was found and removed; false if not registered.</returns>
    public bool UnregisterClassCounts(int clientId)
    {
        lock (_distributionLock)
        {
            return _clientClassDistributions.Remove(clientId);
        }
    }

    /// <summary>
    /// Removes all registered class counts.
    /// </summary>
    public void ClearAllClassCounts()
    {
        lock (_distributionLock)
        {
            _clientClassDistributions.Clear();
        }
    }

    /// <summary>
    /// Calibrates raw logits using the client's local class counts.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If a client has 90% flu and 10% cold samples, the model
    /// naturally predicts flu more often. This method subtracts a pairwise label margin that is
    /// larger for rare classes, effectively removing the bias so the model makes predictions
    /// based on actual features rather than local data imbalance.</para>
    /// </remarks>
    /// <param name="logits">Raw model logits for a single sample (one per class).</param>
    /// <param name="clientId">The client identifier (to look up registered counts).</param>
    /// <returns>Calibrated logits: z_cal[c] = z[c] - tau * n[c]^(-1/4) (Zhang et al. 2022, eq. 9).</returns>
    public Vector<T> CalibrateLogits(Vector<T> logits, int clientId)
    {
        var correction = GetCorrection(clientId);
        if (logits.Length != correction.Length)
        {
            throw new ArgumentException(
                $"Logits length ({logits.Length}) must match distribution length ({correction.Length}).");
        }

        return Engine.Subtract(logits, correction);
    }

    /// <summary>
    /// Calibrates a batch of logits for a client.
    /// </summary>
    /// <param name="logitsBatch">Logits matrix (rows = samples, columns = classes).</param>
    /// <param name="clientId">The client identifier.</param>
    /// <returns>Calibrated logits matrix.</returns>
    public Matrix<T> CalibrateBatch(Matrix<T> logitsBatch, int clientId)
    {
        // Precompute the calibration correction vector (same for all samples).
        var correction = GetCorrection(clientId);
        if (logitsBatch.Columns != correction.Length)
        {
            throw new ArgumentException(
                $"Logits columns ({logitsBatch.Columns}) must match distribution length ({correction.Length}).");
        }

        var rows = Engine.TensorBroadcastTo(
            correction.AsTensor().Reshape(1, correction.Length),
            new[] { logitsBatch.Rows, logitsBatch.Columns });
        return Engine.TensorSubtract(Tensor<T>.FromMatrix(logitsBatch), rows).ToMatrix();
    }

    /// <summary>
    /// Counts the local samples of each class (convenience helper).
    /// </summary>
    /// <param name="labels">Class label indices for all local samples.</param>
    /// <param name="numClasses">Total number of classes.</param>
    /// <returns>Count vector of length numClasses.</returns>
    public static Vector<T> ComputeClassCounts(int[] labels, int numClasses)
    {
        Guard.NotNull(labels);
        if (labels.Length == 0)
        {
            throw new ArgumentException("Labels array must not be empty.", nameof(labels));
        }

        if (numClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");
        }

        var counts = new int[numClasses];
        foreach (int label in labels)
        {
            if (label < 0 || label >= numClasses)
            {
                throw new ArgumentOutOfRangeException(nameof(labels),
                    $"Label {label} is out of range [0, {numClasses}).");
            }

            counts[label]++;
        }

        return Vector<int>.Wrap(counts).AsTensor().Cast<T>().ToVector();
    }

    /// <summary>
    /// tau * max(n[c], 1e-8)^(-1/4) for the client's registered counts (absent classes get the largest margin).
    /// </summary>
    private Vector<T> GetCorrection(int clientId)
    {
        Vector<T> counts;
        lock (_distributionLock)
        {
            if (!_clientClassDistributions.TryGetValue(clientId, out var registered))
            {
                throw new InvalidOperationException(
                    $"No class counts registered for client {clientId}. Call RegisterClassCounts first.");
            }

            counts = registered;
        }

        // Half evaluates the margin in float: its 1e-8 floor rounds to 0, which would give an infinite margin.
        return AiDotNet.FederatedLearning.Infrastructure.FederatedReductions.IsNarrowFloat<T>()
            ? Margin(counts.AsTensor().Cast<float>(), _calibrationTemperature).Cast<T>().ToVector()
            : Margin(counts.AsTensor(), _calibrationTemperature).ToVector();
    }

    private static Tensor<TV> Margin<TV>(Tensor<TV> counts, double temperature)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<TV>();
        var clamped = Engine.TensorClampMin(counts, ops.FromDouble(1e-8));
        var margin = Engine.TensorPower(clamped, ops.FromDouble(-0.25));
        return Engine.TensorMultiplyScalar(margin, ops.FromDouble(temperature));
    }

    /// <summary>Gets the logit calibration temperature (tau).</summary>
    public double CalibrationTemperature => _calibrationTemperature;

    /// <summary>Gets the number of clients with registered class counts.</summary>
    public int RegisteredClientCount
    {
        get
        {
            lock (_distributionLock)
            {
                return _clientClassDistributions.Count;
            }
        }
    }

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedLC(\u03c4={_calibrationTemperature})";
}
