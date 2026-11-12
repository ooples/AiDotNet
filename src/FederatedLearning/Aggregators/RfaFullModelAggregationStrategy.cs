using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Robust Federated Aggregation (RFA) via geometric median (Weiszfeld iterations).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Instead of averaging client updates, the geometric median finds a point that
/// minimizes the sum of distances to all client updates. This is more robust when some clients are
/// outliers or adversarial.
/// </remarks>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class RfaFullModelAggregationStrategy<T, TInput, TOutput> :
    RobustFullModelAggregationStrategyBase<T, TInput, TOutput>
{
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly double _epsilon;
    private readonly bool _useClientWeights;

    public RfaFullModelAggregationStrategy(
        int maxIterations = 10,
        double tolerance = 1e-6,
        double epsilon = 1e-12,
        bool useClientWeights = false)
    {
        if (maxIterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "Max iterations must be positive.");
        }

        if (tolerance <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(tolerance), "Tolerance must be positive.");
        }

        if (epsilon <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _epsilon = epsilon;
        _useClientWeights = useClientWeights;
    }

    /// <inheritdoc />
    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var (reference, parameterCount) = GetReferenceModelOrThrow(clientModels);
        var clientParameters = GetClientParametersOrThrow(clientModels, parameterCount);
        var clientIds = clientParameters.Keys.OrderBy(id => id).ToArray();

        // Initialize with an unweighted average (stable starting point).
        var current = new Vector<T>(parameterCount);
        foreach (var clientId in clientIds)
        {
            current = Engine.Add(current, clientParameters[clientId]);
        }

        current = Engine.Divide(current, NumOps.FromDouble(clientIds.Length));

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var next = new Vector<T>(parameterCount);
            double weightSum = 0.0;

            foreach (var clientId in clientIds)
            {
                var p = clientParameters[clientId];
                var offset = Engine.Subtract(current, p);
                double dist = Math.Sqrt(NumOps.ToDouble(Engine.DotProduct(offset, offset)));
                dist = Math.Max(dist, _epsilon);

                double w = 1.0 / dist;
                if (_useClientWeights && clientWeights.TryGetValue(clientId, out var cw) && cw > 0.0)
                {
                    w *= cw;
                }

                weightSum += w;
                next = Engine.Add(next, Engine.Multiply(p, NumOps.FromDouble(w)));
            }

            if (weightSum <= 0.0)
            {
                break;
            }

            next = Engine.Divide(next, NumOps.FromDouble(weightSum));

            var step = Engine.Subtract(next, current);
            double delta = NumOps.ToDouble(Engine.DotProduct(step, step));
            current = next;
            if (Math.Sqrt(delta) <= _tolerance)
            {
                break;
            }
        }

        var aggregated = current;
        return InterfaceGuard.Parameterizable(reference).WithParameters(aggregated);
    }

    /// <inheritdoc />
    public override string GetStrategyName() => "RFA";

    public static RfaFullModelAggregationStrategy<T, TInput, TOutput> FromOptions(RobustAggregationOptions options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return new RfaFullModelAggregationStrategy<T, TInput, TOutput>(
            maxIterations: options.GeometricMedianMaxIterations,
            tolerance: options.GeometricMedianTolerance,
            epsilon: options.GeometricMedianEpsilon,
            useClientWeights: options.UseClientWeightsWhenAveragingSelectedUpdates);
    }
}

