using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Trainers;

/// <summary>
/// Base class for federated learning trainers.
/// </summary>
/// <typeparam name="TModel">The type of the global model.</typeparam>
/// <typeparam name="TData">The type of per-client training data.</typeparam>
/// <typeparam name="TMetadata">The type of metadata returned by training.</typeparam>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class FederatedTrainerBase<TModel, TData, TMetadata, T> :
    AiDotNet.FederatedLearning.Infrastructure.FederatedLearningComponentBase<T>,
    IFederatedTrainer<TModel, TData, TMetadata>
{
    private IAggregationStrategy<TModel>? _aggregationStrategy;
    private TModel? _globalModel;
    private int _numberOfClients;

    public void Initialize(TModel globalModel, int numberOfClients)
    {
        if (numberOfClients <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfClients), "Number of clients must be positive.");
        }

        _globalModel = globalModel;
        _numberOfClients = numberOfClients;
    }

    public TModel GetGlobalModel()
    {
        if (_globalModel == null)
        {
            throw new InvalidOperationException("Federated trainer is not initialized.");
        }

        return _globalModel;
    }

    public void SetAggregationStrategy(IAggregationStrategy<TModel> strategy)
    {
        _aggregationStrategy = strategy ?? throw new ArgumentNullException(nameof(strategy));
    }

    public abstract TMetadata TrainRound(Dictionary<int, TData> clientData, double clientSelectionFraction = 1.0, int localEpochs = 1);

    public abstract TMetadata Train(Dictionary<int, TData> clientData, int rounds, double clientSelectionFraction = 1.0, int localEpochs = 1);

    protected IAggregationStrategy<TModel> GetAggregationStrategyOrThrow()
    {
        return _aggregationStrategy ?? throw new InvalidOperationException("Aggregation strategy is not set.");
    }

    protected int GetNumberOfClientsOrThrow()
    {
        if (_numberOfClients <= 0)
        {
            throw new InvalidOperationException("Federated trainer is not initialized.");
        }

        return _numberOfClients;
    }

    protected void SetGlobalModel(TModel globalModel)
    {
        _globalModel = globalModel;
    }
}

