using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Federated Elastic Weight Consolidation (EWC) — prevents forgetting by penalizing changes
/// to parameters that are important for previously learned tasks.
/// </summary>
/// <remarks>
/// <para>
/// EWC (Kirkpatrick et al., 2017) uses the Fisher information matrix to estimate which
/// parameters are important for each task. In federated EWC, the Fisher information is
/// computed locally at each client and aggregated at the server to form a global estimate
/// of parameter importance across all tasks and clients.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some model parameters are critical for recognizing cats, while
/// others are critical for recognizing dogs. EWC identifies which parameters matter for which
/// tasks and penalizes changing the important ones. In federated EWC, each client reports
/// which parameters are important for their data, and the server keeps a global importance map.
/// </para>
/// <para>
/// Reference: Kirkpatrick et al. (2017), "Overcoming Catastrophic Forgetting in Neural Networks".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FederatedEWC<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly int _fisherSamples;

    /// <summary>
    /// Creates a new Federated EWC strategy.
    /// </summary>
    /// <param name="fisherSamples">Number of data samples to estimate Fisher information. Default: 200.</param>
    public FederatedEWC(int fisherSamples = 200)
    {
        _fisherSamples = fisherSamples;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData)
    {
        Guard.NotNull(modelParameters);
        Guard.NotNull(taskData);
        int d = modelParameters.Length;
        int n = Math.Min(_fisherSamples, taskData.Rows);
        int columns = taskData.Columns;
        if (n <= 0 || columns == 0)
        {
            throw new ArgumentException(
                "Task data must contain at least one row and one column, and fisherSamples must be positive.", nameof(taskData));
        }

        // Approximate Fisher information: average squared gradient per parameter. The pseudo-gradient of
        // parameter j on sample i is theta_j * x_{i, j mod C}, so the average of its square factors into
        // theta_j^2 * mean_i(x_{i, j mod C}^2).
        var samples = Engine.TensorSlice(Tensor<T>.FromMatrix(taskData), new[] { 0, 0 }, new[] { n, columns });
        var columnMeanSquares = Engine.ReduceMean(Engine.TensorSquare(samples), new[] { 0 }, false); // [C]

        var featureIndex = new int[d];
        for (int j = 0; j < d; j++)
        {
            featureIndex[j] = j % columns;
        }

        var perParameter = Engine.TensorGather(columnMeanSquares, new Tensor<int>(featureIndex, new[] { d }), 0);
        return Engine.TensorMultiply(Engine.TensorSquare(modelParameters.AsTensor()), perParameter).ToVector();
    }

    /// <inheritdoc/>
    public T ComputeRegularizationPenalty(Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength)
    {
        Guard.NotNull(currentParameters);
        Guard.NotNull(referenceParameters);
        Guard.NotNull(importanceWeights);
        int d = currentParameters.Length;
        if (d == 0)
        {
            return NumOps.Zero;
        }

        var reference = referenceParameters.Length == d ? referenceParameters : referenceParameters.GetSubVector(0, d);
        var importance = importanceWeights.Length == d ? importanceWeights : importanceWeights.GetSubVector(0, d);
        var diff = Engine.Subtract(currentParameters, reference);
        T penalty = Engine.Sum(Engine.Multiply(importance, Engine.Multiply(diff, diff)));

        return NumOps.Multiply(penalty, NumOps.FromDouble(0.5 * regularizationStrength));
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        // EWC doesn't project gradients; it uses regularization instead
        // Return gradient unchanged — the penalty handles forgetting prevention
        return gradient;
    }

    /// <inheritdoc/>
    public Vector<T> AggregateImportance(Dictionary<int, Vector<T>> clientImportances,
        Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientImportances);
        if (clientImportances.Count == 0)
            throw new ArgumentException("No client importances provided.", nameof(clientImportances));

        int d = clientImportances.Values.First().Length;
        double totalWeight = 0;
        foreach (var clientId in clientImportances.Keys)
        {
            totalWeight += clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
        }

        if (totalWeight <= 0)
        {
            throw new ArgumentException("Client weights must sum to a positive value.", nameof(clientWeights));
        }

        // Weights are normalized by their total before scaling, so the running sum stays bounded for narrow T.
        var aggregated = new Vector<T>(d);
        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            if (importance.Length != d)
            {
                throw new ArgumentException(
                    $"Client {clientId} importance has length {importance.Length}; expected {d}.", nameof(clientImportances));
            }

            aggregated = Engine.Add(aggregated, Engine.Multiply(importance, NumOps.FromDouble(w / totalWeight)));
        }

        return aggregated;
    }
}
