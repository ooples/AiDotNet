using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Federated Orthogonal Projection — prevents forgetting by projecting gradients to be
/// orthogonal to the subspace of previously important parameter directions.
/// </summary>
/// <remarks>
/// <para>
/// Orthogonal gradient projection (Farajtabar et al., 2020; Zhang et al., ICCV 2025)
/// computes the subspace spanned by important parameter directions from previous tasks,
/// then projects current gradients to be orthogonal to this subspace. This guarantees
/// that new learning does not interfere with previously learned knowledge.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of it as building new knowledge on a "wall" that's perpendicular
/// to the old knowledge. If the old knowledge lives on the floor, new knowledge goes up the wall
/// — they never conflict. In federated settings, each client reports which directions are important,
/// and the global projection space is the union of all clients' important directions.
/// </para>
/// <para>
/// References:
/// Farajtabar et al. (2020), "Orthogonal Gradient Descent for Continual Learning".
/// Zhang et al. (2025), "FedAGC: Federated Continual Learning with Asymmetric Gradient Correction" (ICCV 2025).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FederatedOrthogonalProjection<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly double _projectionThreshold;

    /// <summary>
    /// Creates a new federated orthogonal projection strategy.
    /// </summary>
    /// <param name="projectionThreshold">Minimum importance to include a direction in the projection space. Default: 0.01.</param>
    public FederatedOrthogonalProjection(double projectionThreshold = 0.01)
    {
        _projectionThreshold = projectionThreshold;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData)
    {
        Guard.NotNull(modelParameters);
        Guard.NotNull(taskData);
        int d = modelParameters.Length;
        int n = taskData.Rows;
        int columns = taskData.Columns;
        if (n == 0 || columns == 0)
        {
            throw new ArgumentException("Task data must contain at least one row and one column.", nameof(taskData));
        }

        // Gradient covariance diagonal as importance estimate: the pseudo-gradient of parameter j on sample i
        // is theta_j * x_{i, j mod C}, so its mean square is theta_j^2 * mean_i(x_{i, j mod C}^2).
        var columnMeanSquares = Engine.ReduceMean(Engine.TensorSquare(Tensor<T>.FromMatrix(taskData)), new[] { 0 }, false);

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
        // Orthogonal projection doesn't use regularization — it uses gradient projection instead
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        Guard.NotNull(gradient);
        Guard.NotNull(importanceWeights);
        int d = gradient.Length;
        if (d == 0)
        {
            return new Vector<T>(0);
        }

        // Compute the projection: g_projected = g - (g·e)/(e·e) * e
        // where e is the importance vector restricted to entries above the projection threshold; the
        // remaining entries keep the gradient as-is because e is zero there.
        var importance = (importanceWeights.Length == d ? importanceWeights : importanceWeights.GetSubVector(0, d)).AsTensor();
        var mask = Engine.TensorGreaterThan(importance, NumOps.FromDouble(_projectionThreshold));
        var direction = Engine.TensorMultiply(importance, mask);

        // The projection is invariant to the scale of e; dividing by its largest entry keeps e·e finite for
        // narrow T.
        T maxDirection = Engine.TensorMaxValue(Engine.TensorAbs(direction));
        if (!NumOps.GreaterThan(maxDirection, NumOps.Zero))
        {
            return gradient.Clone();
        }

        var unit = Engine.TensorDivideScalar(direction, maxDirection).ToVector();
        T projScale = NumOps.Divide(Engine.DotProduct(gradient, unit), Engine.DotProduct(unit, unit));
        return Engine.Subtract(gradient, Engine.Multiply(unit, projScale));
    }

    /// <inheritdoc/>
    public Vector<T> AggregateImportance(Dictionary<int, Vector<T>> clientImportances,
        Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientImportances);
        if (clientImportances.Count == 0)
            throw new ArgumentException("No client importances provided.", nameof(clientImportances));

        int d = clientImportances.Values.First().Length;

        // Union of important directions: take max importance across clients (floored at zero).
        var aggregated = new Tensor<T>(new[] { d });
        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            var clientImportance = importance.Length == d ? importance : importance.GetSubVector(0, d);
            aggregated = Engine.TensorMax(aggregated, Engine.TensorMultiplyScalar(clientImportance.AsTensor(), NumOps.FromDouble(w)));
        }

        return aggregated.ToVector();
    }
}
