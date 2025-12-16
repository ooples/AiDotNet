using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.FederatedLearning.Privacy;
using AiDotNet.FederatedLearning.Privacy.Accounting;
using AiDotNet.FederatedLearning.Selection;
using AiDotNet.FederatedLearning.Infrastructure;

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
    private readonly FederatedLearningOptions? _federatedLearningOptions;
    private readonly IPrivacyMechanism<Vector<T>>? _differentialPrivacyMechanismOverride;
    private readonly IPrivacyAccountant? _privacyAccountantOverride;
    private readonly IClientSelectionStrategy? _clientSelectionStrategyOverride;
    private readonly Dictionary<int, double> _clientPerformanceScores = new();
    private readonly Dictionary<int, double[]> _clientEmbeddings = new();

    public InMemoryFederatedTrainer(
        IOptimizer<T, TInput, TOutput> optimizerPrototype,
        double? learningRateOverride = null,
        int? randomSeed = null,
        double convergenceThreshold = 0.001,
        int minRoundsBeforeConvergence = 10,
        FederatedLearningOptions? federatedLearningOptions = null,
        IPrivacyMechanism<Vector<T>>? differentialPrivacyMechanism = null,
        IPrivacyAccountant? privacyAccountant = null,
        IClientSelectionStrategy? clientSelectionStrategy = null)
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
        _federatedLearningOptions = federatedLearningOptions;
        _differentialPrivacyMechanismOverride = differentialPrivacyMechanism;
        _privacyAccountantOverride = privacyAccountant;
        _clientSelectionStrategyOverride = clientSelectionStrategy;
    }

    public override FederatedLearningMetadata TrainRound(
        Dictionary<int, FederatedClientDataset<TInput, TOutput>> clientData,
        double clientSelectionFraction = 1.0,
        int localEpochs = 1)
    {
        return Train(clientData, 1, clientSelectionFraction, localEpochs);
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

        var aggregator = GetAggregationStrategyOrThrow();
        var aggregationName = aggregator.GetStrategyName();

        var flOptions = _federatedLearningOptions;
        bool useDifferentialPrivacy = flOptions?.UseDifferentialPrivacy == true &&
                                      flOptions.DifferentialPrivacyMode != DifferentialPrivacyMode.None;
        bool useSecureAggregation = flOptions?.UseSecureAggregation == true;

        if (useSecureAggregation && !string.Equals(aggregationName, "FedAvg", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(aggregationName, "FedProx", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"Secure aggregation is currently only supported with FedAvg/FedProx. Requested strategy: '{aggregationName}'.");
        }

        var dpMode = flOptions?.DifferentialPrivacyMode ?? DifferentialPrivacyMode.None;
        double dpEpsilon = flOptions?.PrivacyEpsilon ?? 0.0;
        double dpDelta = flOptions?.PrivacyDelta ?? 0.0;
        double dpClipNorm = flOptions?.DifferentialPrivacyClipNorm ?? 1.0;
        string accountantName = flOptions?.PrivacyAccountant ?? "RDP";

        IPrivacyMechanism<Vector<T>>? dpMechanism = null;
        IPrivacyAccountant? privacyAccountant = null;

        if (useDifferentialPrivacy)
        {
            dpMechanism = _differentialPrivacyMechanismOverride ?? new GaussianDifferentialPrivacyVector<T>(dpClipNorm, _randomSeed);
            privacyAccountant = _privacyAccountantOverride ?? CreateDefaultPrivacyAccountant(accountantName, dpClipNorm);
        }

        metadata.DifferentialPrivacyEnabled = useDifferentialPrivacy;
        metadata.SecureAggregationEnabled = useSecureAggregation;
        metadata.AggregationStrategyUsed = aggregationName;
        if (privacyAccountant != null)
        {
            metadata.PrivacyAccountantUsed = privacyAccountant.GetAccountantName();
        }

        for (int round = 0; round < rounds; round++)
        {
            var roundStart = DateTime.UtcNow;
            var globalBefore = GetGlobalModel();
            var selectedClientIds = SelectClients(clientData, round, clientSelectionFraction);

            foreach (var id in selectedClientIds)
            {
                uniqueParticipants.Add(id);
            }

            var clientModels = new Dictionary<int, IFullModel<T, TInput, TOutput>>();
            var clientWeights = new Dictionary<int, double>();
            var maskedParameters = new Dictionary<int, Vector<T>>();

            SecureAggregationVector<T>? secureAggregation = null;
            if (useSecureAggregation)
            {
                secureAggregation = new SecureAggregationVector<T>(globalBefore.ParameterCount, _randomSeed);
                secureAggregation.GeneratePairwiseSecrets(selectedClientIds);
            }

            int privacyEventsThisRound = 0;

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

                double weight = Math.Max(1.0, dataset.SampleCount);
                clientWeights[clientId] = weight;

                var parameters = trainedModel.GetParameters();

                UpdateClientEmbedding(clientId, globalBefore.GetParameters(), parameters);

                if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Local || dpMode == DifferentialPrivacyMode.LocalAndCentral))
                {
                    parameters = dpMechanism!.ApplyPrivacy(parameters, dpEpsilon, dpDelta);
                }

                if (useSecureAggregation)
                {
                    maskedParameters[clientId] = secureAggregation!.MaskUpdate(clientId, parameters, weight);
                }
                else
                {
                    clientModels[clientId] = trainedModel.WithParameters(parameters);
                }
            }

            if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Local || dpMode == DifferentialPrivacyMode.LocalAndCentral))
            {
                privacyAccountant!.AddRound(dpEpsilon, dpDelta, samplingRate: (double)selectedClientIds.Count / GetNumberOfClientsOrThrow());
                privacyEventsThisRound++;
            }

            IFullModel<T, TInput, TOutput> newGlobalModel;
            if (useSecureAggregation)
            {
                var averagedParameters = secureAggregation!.AggregateSecurely(maskedParameters, clientWeights);
                secureAggregation.ClearSecrets();
                newGlobalModel = globalBefore.WithParameters(averagedParameters);
            }
            else
            {
                newGlobalModel = aggregator.Aggregate(clientModels, clientWeights);
            }

            if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Central || dpMode == DifferentialPrivacyMode.LocalAndCentral))
            {
                var globalParams = newGlobalModel.GetParameters();
                var privateGlobalParams = dpMechanism!.ApplyPrivacy(globalParams, dpEpsilon, dpDelta);
                privacyAccountant!.AddRound(dpEpsilon, dpDelta, samplingRate: (double)selectedClientIds.Count / GetNumberOfClientsOrThrow());
                privacyEventsThisRound++;
                newGlobalModel = newGlobalModel.WithParameters(privateGlobalParams);
            }

            SetGlobalModel(newGlobalModel);

            metadata.RoundsCompleted = round + 1;
            metadata.TotalClientsParticipated = uniqueParticipants.Count;
            if (privacyAccountant != null)
            {
                metadata.TotalPrivacyBudgetConsumed = privacyAccountant.GetTotalEpsilonConsumed();
                metadata.TotalPrivacyDeltaConsumed = privacyAccountant.GetTotalDeltaConsumed();
                metadata.ReportedDelta = dpDelta;
                metadata.ReportedEpsilonAtDelta = privacyAccountant.GetEpsilonAtDelta(dpDelta);
            }

            var newParams = newGlobalModel.GetParameters();
            var deltaNorm = ComputeL2Distance(previousGlobalParams, newParams);

            UpdateClientPerformanceScores(selectedClientIds, clientWeights);

            double roundCommunicationMB = EstimateRoundCommunicationMB(selectedClientIds.Count, globalBefore.ParameterCount);
            metadata.TotalCommunicationMB += roundCommunicationMB;
            metadata.RoundMetrics.Add(new RoundMetadata
            {
                RoundNumber = round,
                SelectedClientIds = selectedClientIds,
                RoundTimeSeconds = (DateTime.UtcNow - roundStart).TotalSeconds,
                GlobalLoss = double.NaN,
                GlobalAccuracy = double.NaN,
                AverageLocalLoss = double.NaN,
                CommunicationMB = roundCommunicationMB,
                PrivacyBudgetConsumed = useDifferentialPrivacy ? privacyEventsThisRound * dpEpsilon : 0.0
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

    private static IPrivacyAccountant CreateDefaultPrivacyAccountant(string name, double clipNorm)
    {
        if (string.Equals(name, "Basic", StringComparison.OrdinalIgnoreCase))
        {
            return new BasicCompositionPrivacyAccountant();
        }

        if (string.Equals(name, "RDP", StringComparison.OrdinalIgnoreCase))
        {
            return new RdpPrivacyAccountant(clipNorm);
        }

        throw new InvalidOperationException($"Unknown privacy accountant '{name}'. Supported values: Basic, RDP.");
    }

    private List<int> SelectClients(
        Dictionary<int, FederatedClientDataset<TInput, TOutput>> clientData,
        int roundNumber,
        double fraction)
    {
        var allClientIds = clientData.Keys.OrderBy(id => id).ToList();
        if (allClientIds.Count == 0)
        {
            return new List<int>();
        }

        var weights = new Dictionary<int, double>(allClientIds.Count);
        foreach (var clientId in allClientIds)
        {
            if (clientData.TryGetValue(clientId, out var dataset))
            {
                weights[clientId] = Math.Max(1.0, dataset.SampleCount);
            }
            else
            {
                weights[clientId] = 1.0;
            }
        }

        var flOptions = _federatedLearningOptions;
        var selectionOptions = flOptions?.ClientSelection;
        var strategyName = selectionOptions?.Strategy?.Trim() ?? "UniformRandom";

        var random = FederatedRandom.CreateRoundRandom(_randomSeed, roundNumber, salt: 1337);

        var request = new ClientSelectionRequest
        {
            RoundNumber = roundNumber,
            FractionToSelect = fraction,
            CandidateClientIds = allClientIds,
            ClientWeights = weights,
            ClientGroupKeys = selectionOptions?.ClientGroupKeys,
            ClientAvailabilityProbabilities = selectionOptions?.ClientAvailabilityProbabilities,
            ClientPerformanceScores = _clientPerformanceScores.Count > 0 ? _clientPerformanceScores : null,
            ClientEmbeddings = _clientEmbeddings.Count > 0 ? _clientEmbeddings : null,
            Random = random
        };

        var strategy = _clientSelectionStrategyOverride ?? CreateBuiltInSelectionStrategy(strategyName, selectionOptions);
        return strategy.SelectClients(request);
    }

    private static IClientSelectionStrategy CreateBuiltInSelectionStrategy(string name, ClientSelectionOptions? options)
    {
        if (string.Equals(name, "UniformRandom", StringComparison.OrdinalIgnoreCase))
        {
            return new UniformRandomClientSelectionStrategy();
        }

        if (string.Equals(name, "WeightedRandom", StringComparison.OrdinalIgnoreCase))
        {
            return new WeightedRandomClientSelectionStrategy();
        }

        if (string.Equals(name, "Stratified", StringComparison.OrdinalIgnoreCase))
        {
            return new StratifiedClientSelectionStrategy();
        }

        if (string.Equals(name, "AvailabilityAware", StringComparison.OrdinalIgnoreCase))
        {
            var threshold = options?.AvailabilityThreshold ?? 0.0;
            return new AvailabilityAwareClientSelectionStrategy(threshold);
        }

        if (string.Equals(name, "PerformanceAware", StringComparison.OrdinalIgnoreCase))
        {
            var rate = options?.ExplorationRate ?? 0.1;
            return new PerformanceAwareClientSelectionStrategy(rate);
        }

        if (string.Equals(name, "Clustered", StringComparison.OrdinalIgnoreCase))
        {
            int clusters = options?.ClusterCount ?? 3;
            int iterations = options?.KMeansIterations ?? 5;
            return new ClusteredClientSelectionStrategy(clusters, iterations);
        }

        throw new InvalidOperationException($"Unknown client selection strategy '{name}'. Supported values: UniformRandom, WeightedRandom, Stratified, AvailabilityAware, PerformanceAware, Clustered.");
    }

    private void UpdateClientPerformanceScores(List<int> selectedClientIds, Dictionary<int, double> clientWeights)
    {
        foreach (var clientId in selectedClientIds)
        {
            // Initial, simple proxy score: prefer clients that contributed more data.
            // More advanced scoring (e.g., validation improvement per update) can be layered on later.
            if (clientWeights.TryGetValue(clientId, out var w))
            {
                _clientPerformanceScores[clientId] = w;
            }
        }
    }

    private void UpdateClientEmbedding(int clientId, Vector<T> globalParams, Vector<T> localParams)
    {
        if (globalParams.Length != localParams.Length)
        {
            return;
        }

        // Lightweight embedding: take the first few parameter deltas (as doubles) as a coarse signature.
        int dim = Math.Min(8, globalParams.Length);
        var embedding = new double[dim];
        for (int i = 0; i < dim; i++)
        {
            var delta = NumOps.Subtract(localParams[i], globalParams[i]);
            embedding[i] = NumOps.ToDouble(delta);
        }

        _clientEmbeddings[clientId] = embedding;
    }

    private static double EstimateRoundCommunicationMB(int selectedClientCount, int parameterCount)
    {
        if (selectedClientCount <= 0 || parameterCount <= 0)
        {
            return 0.0;
        }

        int bytesPerParam = EstimateBytesPerNumericType(typeof(T));
        long bytesPerVector = (long)parameterCount * bytesPerParam;

        // Approximate: each selected client downloads global params + uploads one update vector.
        long totalBytes = (long)selectedClientCount * (bytesPerVector + bytesPerVector);
        return totalBytes / 1_000_000.0;
    }

    private static int EstimateBytesPerNumericType(Type numericType)
    {
        try
        {
            return System.Runtime.InteropServices.Marshal.SizeOf(numericType);
        }
        catch
        {
            // Conservative default for unknown numeric types (treat as double-like).
            return 8;
        }
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
