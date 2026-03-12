using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// FedProx aggregation for <see cref="IFullModel{T,TInput,TOutput}"/>.
/// </summary>
/// <remarks>
/// The server-side aggregation step for FedProx is identical to FedAvg. The proximal term
/// affects local training, not aggregation.
/// <para><b>For Beginners:</b> FedProxFullModelAggregationStrategy provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public sealed class FedProxFullModelAggregationStrategy<T, TInput, TOutput> :
    AggregationStrategyBase<IFullModel<T, TInput, TOutput>, T>
{
    private readonly double _mu;
    private readonly FedAvgFullModelAggregationStrategy<T, TInput, TOutput> _fedAvg = new();

    public FedProxFullModelAggregationStrategy(double mu = 0.01)
    {
        if (mu < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(mu), "Mu must be non-negative.");
        }

        _mu = mu;
    }

    public override IFullModel<T, TInput, TOutput> Aggregate(
        Dictionary<int, IFullModel<T, TInput, TOutput>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        return _fedAvg.Aggregate(clientModels, clientWeights);
    }

    public override string GetStrategyName() => "FedProx";

    public double GetMu() => _mu;
}
