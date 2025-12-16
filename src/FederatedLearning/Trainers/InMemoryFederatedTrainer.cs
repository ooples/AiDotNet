using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Results;

namespace AiDotNet.FederatedLearning.Trainers;

/// <summary>
/// In-memory federated learning trainer for local simulation and tests.
/// </summary>
/// <remarks>
/// This trainer runs federated learning rounds in-process by creating per-client model/optimizer instances,
/// running local optimization on each client's data, and aggregating client models into a global model.
/// </remarks>
public sealed class InMemoryFederatedTrainer<T, TInput, TOutput> :
    FederatedTrainerBase<IFullModel<T, TInput, TOutput>, FederatedClientDataset<TInput, TOutput>, FederatedLearningMetadata, T>
{
    private readonly IOptimizer<T, TInput, TOutput> _optimizerPrototype;
    private readonly double? _learningRateOverride;
    private readonly int? _randomSeed;
    private readonly double _convergenceThreshold;
    private readonly int _minRoundsBeforeConvergence;

    public InMemoryFederatedTrainer(
        IOptimizer<T, TInput, TOutput> optimizerPrototype,
        double? learningRateOverride = null,
        int? randomSeed = null,
        double convergenceThreshold = 0.001,
        int minRoundsBeforeConvergence = 10)
    {
        _optimizerPrototype = optimizerPrototype ?? throw new ArgumentNullException(nameof(optimizerPrototype));
        _learningRateOverride = learningRateOverride;

        if (convergenceThreshold < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(convergenceThreshold), "Convergence threshold must be non-negative.");
        }

        if (minRoundsBeforeConvergence < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minRoundsBeforeConvergence), "Min rounds before convergence must be non-negative.");
        }

        _randomSeed = randomSeed;
        _convergenceThreshold = convergenceThreshold;
        _minRoundsBeforeConvergence = minRoundsBeforeConvergence;
    }

    public override FederatedLearningMetadata TrainRound(
        Dictionary<int, FederatedClientDataset<TInput, TOutput>> clientData,
        double clientSelectionFraction = 1.0,
        int localEpochs = 1)
    {
        if (clientData == null || clientData.Count == 0)
        {
            throw new ArgumentException("Client data cannot be null or empty.", nameof(clientData));
        }

        if (clientSelectionFraction <= 0.0 || clientSelectionFraction > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(clientSelectionFraction), "Client selection fraction must be in (0, 1].");
        }

        if (localEpochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(localEpochs), "Local epochs must be positive.");
        }

        var globalModelBefore = GetGlobalModel();
        var selectedClientIds = SelectClients(clientData.Keys.ToList(), clientSelectionFraction);

        var clientModels = new Dictionary<int, IFullModel<T, TInput, TOutput>>();
        var clientWeights = new Dictionary<int, double>();

        foreach (var clientId in selectedClientIds)
        {
            if (!clientData.TryGetValue(clientId, out var dataset))
            {
                continue;
            }

            var localModel = CloneModelByParameters(globalModelBefore);
            var localOptimizer = CreateOptimizerForModel(localModel);
            ConfigureLocalOptimizer(localOptimizer, localEpochs);

            var inputData = CreateLocalOptimizationInputData(dataset, globalModelBefore);
            OptimizationResult<T, TInput, TOutput> localResult = localOptimizer.Optimize(inputData);

            var trainedModel = localResult.BestSolution ?? localModel;

            clientModels[clientId] = trainedModel;
            clientWeights[clientId] = Math.Max(1.0, dataset.SampleCount);
        }

        var aggregator = GetAggregationStrategyOrThrow();
        var newGlobalModel = aggregator.Aggregate(clientModels, clientWeights);
        SetGlobalModel(newGlobalModel);

        var metadata = new FederatedLearningMetadata
        {
            RoundsCompleted = 1,
            AggregationStrategyUsed = aggregator.GetStrategyName()
        };

        return metadata;
    }

    public override FederatedLearningMetadata Train(
        Dictionary<int, FederatedClientDataset<TInput, TOutput>> clientData,
        int rounds,
        double clientSelectionFraction = 1.0,
        int localEpochs = 1)
    {
        if (rounds <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rounds), "Rounds must be positive.");
        }

        var start = DateTime.UtcNow;
        var metadata = new FederatedLearningMetadata();
        var uniqueParticipants = new HashSet<int>();

        var previousGlobalParams = GetGlobalModel().GetParameters();

        for (int round = 0; round < rounds; round++)
        {
            var globalBefore = GetGlobalModel();
            var selectedClientIds = SelectClients(clientData.Keys.ToList(), clientSelectionFraction);

            foreach (var id in selectedClientIds)
            {
                uniqueParticipants.Add(id);
            }

            var clientModels = new Dictionary<int, IFullModel<T, TInput, TOutput>>();
            var clientWeights = new Dictionary<int, double>();

            foreach (var clientId in selectedClientIds)
            {
                if (!clientData.TryGetValue(clientId, out var dataset))
                {
                    continue;
                }

                var localModel = CloneModelByParameters(globalBefore);
                var localOptimizer = CreateOptimizerForModel(localModel);
                ConfigureLocalOptimizer(localOptimizer, localEpochs);

                var inputData = CreateLocalOptimizationInputData(dataset, globalBefore);
                OptimizationResult<T, TInput, TOutput> localResult = localOptimizer.Optimize(inputData);

                var trainedModel = localResult.BestSolution ?? localModel;
                clientModels[clientId] = trainedModel;
                clientWeights[clientId] = Math.Max(1.0, dataset.SampleCount);
            }

            var aggregator = GetAggregationStrategyOrThrow();
            var newGlobalModel = aggregator.Aggregate(clientModels, clientWeights);
            SetGlobalModel(newGlobalModel);

            metadata.RoundsCompleted = round + 1;
            metadata.AggregationStrategyUsed = aggregator.GetStrategyName();
            metadata.TotalClientsParticipated = uniqueParticipants.Count;

            var newParams = newGlobalModel.GetParameters();
            var deltaNorm = ComputeL2Distance(previousGlobalParams, newParams);
            metadata.RoundMetrics.Add(new RoundMetadata
            {
                RoundNumber = round,
                SelectedClientIds = selectedClientIds,
                RoundTimeSeconds = (DateTime.UtcNow - start).TotalSeconds,
                GlobalLoss = double.NaN,
                GlobalAccuracy = double.NaN,
                AverageLocalLoss = double.NaN,
                CommunicationMB = 0.0,
                PrivacyBudgetConsumed = 0.0
            });

            previousGlobalParams = newParams;

            if (round + 1 >= _minRoundsBeforeConvergence && deltaNorm <= _convergenceThreshold)
            {
                metadata.Converged = true;
                metadata.ConvergenceRound = round + 1;
                metadata.Notes = $"Converged by parameter delta L2 <= {_convergenceThreshold:0.########}.";
                break;
            }
        }

        var elapsed = DateTime.UtcNow - start;
        metadata.TotalTrainingTimeSeconds = elapsed.TotalSeconds;
        metadata.AverageRoundTimeSeconds = metadata.RoundsCompleted > 0 ? elapsed.TotalSeconds / metadata.RoundsCompleted : 0.0;
        metadata.AverageClientsPerRound = metadata.RoundsCompleted > 0 ? (double)metadata.RoundMetrics.Sum(r => r.SelectedClientIds.Count) / metadata.RoundsCompleted : 0.0;

        return metadata;
    }

    private List<int> SelectClients(List<int> allClientIds, double fraction)
    {
        allClientIds.Sort();

        int countToSelect = Math.Max(1, (int)Math.Ceiling(allClientIds.Count * fraction));
        if (countToSelect >= allClientIds.Count)
        {
            return allClientIds;
        }

        var rng = _randomSeed.HasValue ? new Random(_randomSeed.Value) : new Random();
        var shuffled = allClientIds.OrderBy(_ => rng.Next()).ToList();
        shuffled.Sort();
        return shuffled.Take(countToSelect).ToList();
    }

    private IOptimizer<T, TInput, TOutput> CreateOptimizerForModel(IFullModel<T, TInput, TOutput> model)
    {
        var optimizerType = _optimizerPrototype.GetType();
        var options = _optimizerPrototype.GetOptions();

        var constructors = optimizerType.GetConstructors();
        foreach (var ctor in constructors)
        {
            var parameters = ctor.GetParameters();
            if (parameters.Length == 2 &&
                parameters[0].ParameterType.IsInstanceOfType(model) &&
                parameters[1].ParameterType.IsInstanceOfType(options))
            {
                return (IOptimizer<T, TInput, TOutput>)ctor.Invoke([model, options]);
            }
        }

        foreach (var ctor in constructors)
        {
            var parameters = ctor.GetParameters();
            if (parameters.Length == 1 &&
                parameters[0].ParameterType.IsInstanceOfType(model))
            {
                return (IOptimizer<T, TInput, TOutput>)ctor.Invoke([model]);
            }
        }

        foreach (var ctor in constructors)
        {
            var parameters = ctor.GetParameters();
            if (parameters.Length == 2 &&
                parameters[0].ParameterType.IsInstanceOfType(model) &&
                parameters[1].IsOptional)
            {
                return (IOptimizer<T, TInput, TOutput>)ctor.Invoke([model, null]);
            }
        }

        throw new InvalidOperationException(
            $"Unable to construct optimizer of type '{optimizerType.FullName}' for model '{model.GetType().FullName}'. " +
            "Expected a constructor like (IFullModel<...> model, <Options> options) or (IFullModel<...> model).");
    }

    private void ConfigureLocalOptimizer(IOptimizer<T, TInput, TOutput> optimizer, int localEpochs)
    {
        var options = optimizer.GetOptions();
        options.MaxIterations = localEpochs;
        options.UseEarlyStopping = false;
        if (_learningRateOverride.HasValue)
        {
            options.InitialLearningRate = _learningRateOverride.Value;
        }
    }

    private OptimizationInputData<T, TInput, TOutput> CreateLocalOptimizationInputData(
        FederatedClientDataset<TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput> globalModel)
    {
        var input = OptimizerHelper<T, TInput, TOutput>.CreateOptimizationInputData(
            dataset.Features, dataset.Labels,
            dataset.Features, dataset.Labels,
            dataset.Features, dataset.Labels);

        input.InitialSolution = globalModel;
        return input;
    }

    private double ComputeL2Distance(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vector length mismatch when computing distance.");
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            var diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        var norm = NumOps.Sqrt(sum);
        return NumOps.ToDouble(norm);
    }

    private static IFullModel<T, TInput, TOutput> CloneModelByParameters(IFullModel<T, TInput, TOutput> model)
    {
        // Prefer parameter-based cloning to avoid relying on model-specific serialization in DeepCopy().
        var parameters = model.GetParameters();
        return model.WithParameters(parameters);
    }
}
