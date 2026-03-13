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

    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        var (reference, parameterCount) = GetReferenceModelOrThrow(clientModels);
        var clientParameters = GetClientParametersOrThrow(clientModels, parameterCount);
        var clientIds = clientParameters.Keys.OrderBy(id => id).ToArray();

        // Initialize with an unweighted average (stable starting point).
        var current = new double[parameterCount];
        for (int i = 0; i < parameterCount; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < clientIds.Length; j++)
            {
                sum += NumOps.ToDouble(clientParameters[clientIds[j]][i]);
            }

            current[i] = sum / clientIds.Length;
        }

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var next = new double[parameterCount];
            double weightSum = 0.0;

            foreach (var clientId in clientIds)
            {
                var p = clientParameters[clientId];
                double dist = 0.0;
                for (int i = 0; i < parameterCount; i++)
                {
                    double diff = current[i] - NumOps.ToDouble(p[i]);
                    dist += diff * diff;
                }

                dist = Math.Sqrt(dist);
                dist = Math.Max(dist, _epsilon);

                double w = 1.0 / dist;
                if (_useClientWeights && clientWeights.TryGetValue(clientId, out var cw) && cw > 0.0)
                {
                    w *= cw;
                }

                weightSum += w;
                for (int i = 0; i < parameterCount; i++)
                {
                    next[i] += w * NumOps.ToDouble(p[i]);
                }
            }

            if (weightSum <= 0.0)
            {
                break;
            }

            for (int i = 0; i < parameterCount; i++)
            {
                next[i] /= weightSum;
            }

            double delta = 0.0;
            for (int i = 0; i < parameterCount; i++)
            {
                double diff = next[i] - current[i];
                delta += diff * diff;
            }

            if (Math.Sqrt(delta) <= _tolerance)
            {
                current = next;
                break;
            }

            current = next;
        }

        var aggregated = new Vector<T>(parameterCount);
        for (int i = 0; i < parameterCount; i++)
        {
            aggregated[i] = NumOps.FromDouble(current[i]);
        }

        return reference.WithParameters(aggregated);
    }

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

