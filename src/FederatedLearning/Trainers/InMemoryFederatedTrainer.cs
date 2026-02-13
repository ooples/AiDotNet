using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Heterogeneity;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.FederatedLearning.Privacy;
using AiDotNet.FederatedLearning.Privacy.Accounting;
using AiDotNet.FederatedLearning.Selection;
using AiDotNet.FederatedLearning.ServerOptimizers;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Validation;

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
    private readonly IFederatedServerOptimizer<T>? _serverOptimizerOverride;
    private readonly IFederatedHeterogeneityCorrection<T>? _heterogeneityCorrectionOverride;
    private readonly IHomomorphicEncryptionProvider<T>? _homomorphicEncryptionProviderOverride;
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
        IClientSelectionStrategy? clientSelectionStrategy = null,
        IFederatedServerOptimizer<T>? serverOptimizer = null,
        IFederatedHeterogeneityCorrection<T>? heterogeneityCorrection = null,
        IHomomorphicEncryptionProvider<T>? homomorphicEncryptionProvider = null)
    {
        Guard.NotNull(optimizerPrototype);
        _optimizerPrototype = optimizerPrototype;
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
        _serverOptimizerOverride = serverOptimizer;
        _heterogeneityCorrectionOverride = heterogeneityCorrection;
        _homomorphicEncryptionProviderOverride = homomorphicEncryptionProvider;
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

        var secureAggregationOptions = flOptions?.SecureAggregation;
        bool useSecureAggregation = flOptions?.UseSecureAggregation == true || secureAggregationOptions?.Enabled == true;
        SecureAggregationMode secureAggregationMode = secureAggregationOptions?.Mode ?? SecureAggregationMode.FullParticipation;

        if (useSecureAggregation && !Enum.IsDefined(typeof(SecureAggregationMode), secureAggregationMode))
        {
            throw new InvalidOperationException($"Unknown secure aggregation mode: '{secureAggregationMode}'.");
        }

        if (useSecureAggregation && !string.Equals(aggregationName, "FedAvg", StringComparison.OrdinalIgnoreCase) &&
            !string.Equals(aggregationName, "FedProx", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"Secure aggregation is currently only supported with FedAvg/FedProx. Requested strategy: '{aggregationName}'.");
        }

        var dpMode = flOptions?.DifferentialPrivacyMode ?? DifferentialPrivacyMode.None;
        double dpEpsilon = flOptions?.PrivacyEpsilon ?? 0.0;
        double dpDelta = flOptions?.PrivacyDelta ?? 0.0;
        double dpClipNorm = flOptions?.DifferentialPrivacyClipNorm ?? 1.0;
        FederatedPrivacyAccountant accountant = flOptions?.PrivacyAccountant ?? FederatedPrivacyAccountant.Rdp;

        IPrivacyMechanism<Vector<T>>? dpMechanism = null;
        IPrivacyAccountant? privacyAccountant = null;

        if (useDifferentialPrivacy)
        {
            dpMechanism = _differentialPrivacyMechanismOverride ?? new GaussianDifferentialPrivacyVector<T>(dpClipNorm, _randomSeed);
            privacyAccountant = _privacyAccountantOverride ?? CreateDefaultPrivacyAccountant(accountant, dpClipNorm);
        }

        metadata.DifferentialPrivacyEnabled = useDifferentialPrivacy;
        metadata.SecureAggregationEnabled = useSecureAggregation;
        metadata.SecureAggregationModeUsed = useSecureAggregation ? secureAggregationMode.ToString() : "None";
        metadata.SecureAggregationMinimumUploaderCountUsed = 0;
        metadata.SecureAggregationReconstructionThresholdUsed = 0;
        metadata.AggregationStrategyUsed = aggregationName;
        if (privacyAccountant != null)
        {
            metadata.PrivacyAccountantUsed = privacyAccountant.GetAccountantName();
        }

        var serverOptimizer = _serverOptimizerOverride ?? CreateDefaultServerOptimizer(flOptions?.ServerOptimizer);
        metadata.ServerOptimizerUsed = serverOptimizer?.GetOptimizerName() ?? "None";

        var heterogeneityCorrection = _heterogeneityCorrectionOverride ?? CreateDefaultHeterogeneityCorrection(flOptions?.HeterogeneityCorrection);
        metadata.HeterogeneityCorrectionUsed = heterogeneityCorrection?.GetCorrectionName() ?? "None";

        var compressionOptions = ResolveCompressionOptions(flOptions);
        bool useCompression = compressionOptions != null &&
                              !string.Equals(compressionOptions.Strategy?.Trim() ?? "None", "None", StringComparison.OrdinalIgnoreCase);
        metadata.CompressionEnabled = useCompression;
        metadata.CompressionStrategyUsed = useCompression ? (compressionOptions!.Strategy?.Trim() ?? "None") : "None";
        Dictionary<int, Vector<T>>? compressionResiduals = useCompression && compressionOptions!.UseErrorFeedback
            ? new Dictionary<int, Vector<T>>()
            : null;

        var heOptions = flOptions?.HomomorphicEncryption;
        bool useHomomorphicEncryption = heOptions?.Enabled == true;
        HomomorphicEncryptionScheme heScheme = useHomomorphicEncryption ? heOptions!.Scheme : HomomorphicEncryptionScheme.Ckks;
        HomomorphicEncryptionMode heMode = useHomomorphicEncryption ? heOptions!.Mode : HomomorphicEncryptionMode.HeOnly;
        var heProvider = useHomomorphicEncryption ? (_homomorphicEncryptionProviderOverride ?? new SealHomomorphicEncryptionProvider<T>()) : null;
        var encryptedIndices = useHomomorphicEncryption
            ? ResolveEncryptedIndices(heOptions!, GetGlobalModel().ParameterCount, heMode)
            : Array.Empty<int>();

        metadata.HomomorphicEncryptionEnabled = useHomomorphicEncryption;
        metadata.HomomorphicEncryptionSchemeUsed = useHomomorphicEncryption ? GetHomomorphicEncryptionSchemeName(heScheme) : "None";
        metadata.HomomorphicEncryptionModeUsed = useHomomorphicEncryption ? GetHomomorphicEncryptionModeName(heMode) : "None";
        metadata.HomomorphicEncryptionProviderUsed = heProvider?.GetProviderName() ?? "None";

        var personalizationOptions = ResolvePersonalizationOptions(flOptions);
        bool usePersonalization = personalizationOptions != null &&
                                  personalizationOptions.Enabled &&
                                  !string.Equals(personalizationOptions.Strategy?.Trim() ?? "None", "None", StringComparison.OrdinalIgnoreCase);

        metadata.PersonalizationEnabled = usePersonalization;
        metadata.PersonalizationStrategyUsed = usePersonalization ? (personalizationOptions!.Strategy?.Trim() ?? "None") : "None";
        metadata.PersonalizedParameterFraction = usePersonalization ? personalizationOptions!.PersonalizedParameterFraction : 0.0;
        metadata.PersonalizationLocalAdaptationEpochs = usePersonalization ? Math.Max(0, personalizationOptions!.LocalAdaptationEpochs) : 0;

        var metaLearningOptions = flOptions?.MetaLearning;
        bool useMetaLearning = metaLearningOptions != null &&
                               metaLearningOptions.Enabled &&
                               !string.Equals(metaLearningOptions.Strategy?.Trim() ?? "None", "None", StringComparison.OrdinalIgnoreCase);

        metadata.MetaLearningEnabled = useMetaLearning;
        metadata.MetaLearningStrategyUsed = useMetaLearning ? (metaLearningOptions!.Strategy?.Trim() ?? "None") : "None";
        metadata.MetaLearningRateUsed = useMetaLearning ? metaLearningOptions!.MetaLearningRate : 0.0;
        metadata.MetaLearningInnerEpochsUsed = useMetaLearning ? (metaLearningOptions!.InnerEpochs > 0 ? metaLearningOptions.InnerEpochs : localEpochs) : 0;

        if (usePersonalization && useMetaLearning)
        {
            throw new InvalidOperationException("Personalization and federated meta-learning are not supported together in the v1 in-memory trainer. Choose one.");
        }

        var asyncOptions = flOptions?.AsyncFederatedLearning;
        FederatedAsyncMode asyncMode = asyncOptions?.Mode ?? FederatedAsyncMode.None;
        metadata.AsyncModeUsed = asyncMode.ToString();

        if (asyncMode != FederatedAsyncMode.None)
        {
            if (usePersonalization || useMetaLearning)
            {
                throw new InvalidOperationException("Personalization and federated meta-learning are currently only supported in synchronous mode in the v1 in-memory trainer.");
            }

            if (useSecureAggregation)
            {
                throw new InvalidOperationException("Secure aggregation is currently not supported with asynchronous federated learning modes.");
            }

            if (useHomomorphicEncryption && heMode == HomomorphicEncryptionMode.HeOnly)
            {
                throw new InvalidOperationException("HE-only aggregation is currently not supported with asynchronous federated learning modes in the in-memory simulator.");
            }

            TrainAsyncInMemory(
                clientData,
                rounds,
                clientSelectionFraction,
                localEpochs,
                metadata,
                uniqueParticipants,
                previousGlobalParams,
                aggregator,
                serverOptimizer,
                useDifferentialPrivacy,
                dpMechanism,
                privacyAccountant,
                dpMode,
                dpEpsilon,
                dpDelta,
                asyncOptions,
                compressionOptions,
                compressionResiduals,
                heterogeneityCorrection,
                heProvider,
                heOptions,
                encryptedIndices);

            var asyncElapsed = DateTime.UtcNow - start;
            metadata.TotalTrainingTimeSeconds = asyncElapsed.TotalSeconds;
            metadata.AverageRoundTimeSeconds = metadata.RoundsCompleted > 0 ? asyncElapsed.TotalSeconds / metadata.RoundsCompleted : 0.0;
            metadata.AverageClientsPerRound = metadata.RoundsCompleted > 0 ? (double)metadata.RoundMetrics.Sum(r => r.SelectedClientIds.Count) / metadata.RoundsCompleted : 0.0;
            return metadata;
        }

        var personalizationStrategy = metadata.PersonalizationStrategyUsed;
        bool isHeadSplitPersonalization = usePersonalization && IsHeadSplitPersonalization(personalizationStrategy);
        bool isClusteredPersonalization = usePersonalization && IsClusteredPersonalization(personalizationStrategy);
        var personalizedIndices = (isHeadSplitPersonalization || isClusteredPersonalization)
            ? ResolvePersonalizedIndices(metadata.PersonalizedParameterFraction, GetGlobalModel().ParameterCount)
            : Array.Empty<int>();

        Dictionary<int, Vector<T>>? perClientPersonalState = usePersonalization ? new Dictionary<int, Vector<T>>() : null;
        Dictionary<int, Vector<T>>? perClusterPersonalState = isClusteredPersonalization ? new Dictionary<int, Vector<T>>() : null;

        for (int round = 0; round < rounds; round++)
        {
            var roundStart = DateTime.UtcNow;
            var globalBefore = GetGlobalModel();
            var globalBeforeParams = globalBefore.GetParameters();
            var selectedClientIds = SelectClients(clientData, round, clientSelectionFraction);

            foreach (var id in selectedClientIds)
            {
                uniqueParticipants.Add(id);
            }

            var clientModels = new Dictionary<int, IFullModel<T, TInput, TOutput>>();
            var clientWeights = new Dictionary<int, double>();
            var maskedParameters = new Dictionary<int, Vector<T>>();
            var heClientParameters = useHomomorphicEncryption ? new Dictionary<int, Vector<T>>() : null;
            double uploadRatioSum = 0.0;
            int uploadRatioCount = 0;

            SecureAggregationVector<T>? secureAggregation = null;
            ThresholdSecureAggregationVector<T>? thresholdSecureAggregation = null;
            if (useSecureAggregation)
            {
                if (secureAggregationMode == SecureAggregationMode.ThresholdDropoutResilient)
                {
                    thresholdSecureAggregation = new ThresholdSecureAggregationVector<T>(globalBefore.ParameterCount, _randomSeed);
                    thresholdSecureAggregation.InitializeRound(
                        selectedClientIds,
                        minimumUploaderCount: secureAggregationOptions?.MinimumUploaderCount ?? 0,
                        reconstructionThreshold: secureAggregationOptions?.ReconstructionThreshold ?? 0,
                        maxDropoutFraction: secureAggregationOptions?.MaxDropoutFraction ?? 0.2);

                    metadata.SecureAggregationMinimumUploaderCountUsed = thresholdSecureAggregation.MinimumUploaderCount;
                    metadata.SecureAggregationReconstructionThresholdUsed = thresholdSecureAggregation.ReconstructionThreshold;
                }
                else
                {
                    secureAggregation = new SecureAggregationVector<T>(globalBefore.ParameterCount, _randomSeed);
                    secureAggregation.GeneratePairwiseSecrets(selectedClientIds);

                    // Full participation mode: "threshold" is effectively the entire selected set.
                    metadata.SecureAggregationMinimumUploaderCountUsed = selectedClientIds.Count;
                    metadata.SecureAggregationReconstructionThresholdUsed = selectedClientIds.Count;
                }
            }

            int privacyEventsThisRound = 0;

            foreach (var clientId in selectedClientIds)
            {
                if (!clientData.TryGetValue(clientId, out var dataset))
                {
                    continue;
                }

                var clientStartModel = usePersonalization
                    ? CreatePersonalizedStartModel(
                        personalizationStrategy,
                        personalizationOptions!,
                        clientId,
                        globalBefore,
                        globalBeforeParams,
                        personalizedIndices,
                        perClientPersonalState!,
                        perClusterPersonalState)
                    : globalBefore;

                var localModel = CloneModelByParameters(clientStartModel);
                var localOptimizer = CreateOptimizerForModel(localModel);
                int effectiveLocalEpochs = useMetaLearning && metaLearningOptions!.InnerEpochs > 0 ? metaLearningOptions.InnerEpochs : localEpochs;
                ConfigureLocalOptimizer(localOptimizer, effectiveLocalEpochs);

                var inputData = CreateLocalOptimizationInputData(dataset, localModel);
                OptimizationResult<T, TInput, TOutput> localResult = localOptimizer.Optimize(inputData);

                var trainedModel = localResult.BestSolution ?? localModel;

                double weight = Math.Max(1.0, dataset.SampleCount);
                clientWeights[clientId] = weight;

                var trainedParameters = trainedModel.GetParameters();
                var parameters = usePersonalization
                    ? ApplyPersonalizationAfterLocalTraining(
                        personalizationStrategy,
                        personalizationOptions!,
                        clientId,
                        globalBeforeParams,
                        trainedParameters,
                        personalizedIndices,
                        perClientPersonalState!)
                    : trainedParameters;

                UpdateClientEmbedding(clientId, globalBeforeParams, parameters);

                if (heterogeneityCorrection != null)
                {
                    parameters = heterogeneityCorrection.Correct(
                        clientId,
                        round,
                        globalBeforeParams,
                        parameters,
                        effectiveLocalEpochs);
                }

                if (useCompression)
                {
                    var clientRandom = FederatedRandom.CreateClientRandom(_randomSeed, round, clientId, salt: 4242);
                    parameters = ApplyCompressionToParameters(
                        clientId,
                        globalBeforeParams,
                        parameters,
                        compressionOptions!,
                        compressionResiduals,
                        clientRandom,
                        out var uploadRatio);

                    uploadRatioSum += uploadRatio;
                    uploadRatioCount++;
                }

                if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Local || dpMode == DifferentialPrivacyMode.LocalAndCentral))
                {
                    parameters = dpMechanism!.ApplyPrivacy(parameters, dpEpsilon, dpDelta);
                }

                var parametersForAggregation = parameters;
                if (useHomomorphicEncryption)
                {
                    heClientParameters![clientId] = parameters;

                    parametersForAggregation = heMode == HomomorphicEncryptionMode.HeOnly
                        ? globalBeforeParams
                        : MaskEncryptedIndices(parameters, globalBeforeParams, encryptedIndices);
                }

                if (useHomomorphicEncryption && heMode == HomomorphicEncryptionMode.HeOnly)
                {
                    // Do not add to plaintext aggregation dictionaries in HE-only mode.
                    continue;
                }

                if (useSecureAggregation)
                {
                    maskedParameters[clientId] = thresholdSecureAggregation != null
                        ? thresholdSecureAggregation.MaskUpdate(clientId, parametersForAggregation, weight)
                        : secureAggregation!.MaskUpdate(clientId, parametersForAggregation, weight);
                }
                else
                {
                    clientModels[clientId] = trainedModel.WithParameters(parametersForAggregation);
                }
            }

            if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Local || dpMode == DifferentialPrivacyMode.LocalAndCentral))
            {
                privacyAccountant!.AddRound(dpEpsilon, dpDelta, samplingRate: (double)selectedClientIds.Count / GetNumberOfClientsOrThrow());
                privacyEventsThisRound++;
            }

            IFullModel<T, TInput, TOutput> newGlobalModel;
            if (useHomomorphicEncryption && heMode == HomomorphicEncryptionMode.HeOnly)
            {
                if (useSecureAggregation)
                {
                    throw new InvalidOperationException("Secure aggregation is not applicable when HE-only aggregation is enabled.");
                }

                var heAggregated = heProvider!.AggregateEncryptedWeightedAverage(
                    heClientParameters!,
                    clientWeights,
                    globalBefore.GetParameters(),
                    encryptedIndices,
                    heOptions!);

                newGlobalModel = globalBefore.WithParameters(heAggregated);
            }
            else if (useSecureAggregation)
            {
                var averagedParameters = thresholdSecureAggregation != null
                    ? thresholdSecureAggregation.AggregateSecurely(maskedParameters, clientWeights)
                    : secureAggregation!.AggregateSecurely(maskedParameters, clientWeights);

                if (thresholdSecureAggregation != null)
                {
                    thresholdSecureAggregation.ClearSecrets();
                }
                else
                {
                    secureAggregation!.ClearSecrets();
                }

                newGlobalModel = globalBefore.WithParameters(averagedParameters);
            }
            else
            {
                newGlobalModel = aggregator.Aggregate(clientModels, clientWeights);
            }

            if (useHomomorphicEncryption && heMode != HomomorphicEncryptionMode.HeOnly)
            {
                var heAggregated = heProvider!.AggregateEncryptedWeightedAverage(
                    heClientParameters!,
                    clientWeights,
                    globalBeforeParams,
                    encryptedIndices,
                    heOptions!);

                var merged = newGlobalModel.GetParameters();
                foreach (var idx in encryptedIndices)
                {
                    merged[idx] = heAggregated[idx];
                }

                newGlobalModel = newGlobalModel.WithParameters(merged);
            }

            if (useMetaLearning)
            {
                var metaRate = metaLearningOptions!.MetaLearningRate;
                var averaged = newGlobalModel.GetParameters();
                var metaUpdated = ApplyMetaLearningUpdate(globalBeforeParams, averaged, metaRate);
                newGlobalModel = globalBefore.WithParameters(metaUpdated);
            }

            if (serverOptimizer != null)
            {
                var updatedParams = serverOptimizer.Step(globalBeforeParams, newGlobalModel.GetParameters());
                newGlobalModel = newGlobalModel.WithParameters(updatedParams);
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

            if (usePersonalization)
            {
                ApplyPostAggregationPersonalization(
                    personalizationStrategy,
                    personalizationOptions!,
                    clientData,
                    selectedClientIds,
                    newGlobalModel.GetParameters(),
                    personalizedIndices,
                    perClientPersonalState!,
                    perClusterPersonalState);
            }

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

            double averageUploadRatio = uploadRatioCount > 0 ? uploadRatioSum / uploadRatioCount : 1.0;
            double roundCommunicationMB = EstimateRoundCommunicationMB(selectedClientIds.Count, globalBefore.ParameterCount, averageUploadRatio);
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
                UploadCompressionRatio = averageUploadRatio,
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

    private void TrainAsyncInMemory(
        Dictionary<int, FederatedClientDataset<TInput, TOutput>> clientData,
        int serverSteps,
        double clientSelectionFraction,
        int localEpochs,
        FederatedLearningMetadata metadata,
        HashSet<int> uniqueParticipants,
        Vector<T> previousGlobalParams,
        IAggregationStrategy<IFullModel<T, TInput, TOutput>> aggregator,
        IFederatedServerOptimizer<T>? serverOptimizer,
        bool useDifferentialPrivacy,
        IPrivacyMechanism<Vector<T>>? dpMechanism,
        IPrivacyAccountant? privacyAccountant,
        DifferentialPrivacyMode dpMode,
        double dpEpsilon,
        double dpDelta,
        AsyncFederatedLearningOptions? asyncOptions,
        FederatedCompressionOptions? compressionOptions,
        Dictionary<int, Vector<T>>? compressionResiduals,
        IFederatedHeterogeneityCorrection<T>? heterogeneityCorrection,
        IHomomorphicEncryptionProvider<T>? heProvider,
        HomomorphicEncryptionOptions? heOptions,
        int[] encryptedIndices)
    {
        FederatedAsyncMode mode = asyncOptions?.Mode ?? FederatedAsyncMode.None;
        int maxDelay = Math.Max(0, asyncOptions?.SimulatedMaxClientDelaySteps ?? 0);
        int rejectStale = Math.Max(0, asyncOptions?.RejectUpdatesWithStalenessGreaterThan ?? 0);
        int bufferSize = Math.Max(1, asyncOptions?.FedBuffBufferSize ?? 5);
        double mixingRate = asyncOptions?.FedAsyncMixingRate ?? 0.5;

        var pending = new List<(int ClientId, Vector<T> Parameters, double Weight, int StartStep, int ArrivalStep)>();
        var bufferModels = new Dictionary<int, IFullModel<T, TInput, TOutput>>();
        var bufferWeights = new Dictionary<int, double>();
        var bufferHeParameters = heProvider != null && heOptions?.Enabled == true && encryptedIndices.Length > 0
            ? new Dictionary<int, Vector<T>>()
            : null;
        int bufferedUpdateKey = 0;

        bool useCompression = compressionOptions != null &&
                              !string.Equals(compressionOptions.Strategy?.Trim() ?? "None", "None", StringComparison.OrdinalIgnoreCase);
        bool useHomomorphicEncryption = heProvider != null &&
                                        heOptions != null &&
                                        heOptions.Enabled &&
                                        encryptedIndices.Length > 0 &&
                                        heOptions.Mode == HomomorphicEncryptionMode.Hybrid;

        for (int step = 0; step < serverSteps; step++)
        {
            var stepStart = DateTime.UtcNow;
            var globalAtStepStart = GetGlobalModel();
            var selectedClientIds = SelectClients(clientData, step, clientSelectionFraction);

            foreach (var id in selectedClientIds)
            {
                uniqueParticipants.Add(id);
            }

            var startedClientWeights = new Dictionary<int, double>();
            double uploadRatioSum = 0.0;
            int uploadRatioCount = 0;
            int privacyEventsThisStep = 0;

            foreach (var clientId in selectedClientIds)
            {
                if (!clientData.TryGetValue(clientId, out var dataset))
                {
                    continue;
                }

                var localModel = CloneModelByParameters(globalAtStepStart);
                var localOptimizer = CreateOptimizerForModel(localModel);
                ConfigureLocalOptimizer(localOptimizer, localEpochs);

                var inputData = CreateLocalOptimizationInputData(dataset, globalAtStepStart);
                OptimizationResult<T, TInput, TOutput> localResult = localOptimizer.Optimize(inputData);
                var trainedModel = localResult.BestSolution ?? localModel;

                double weight = Math.Max(1.0, dataset.SampleCount);
                startedClientWeights[clientId] = weight;

                var parameters = trainedModel.GetParameters();
                UpdateClientEmbedding(clientId, globalAtStepStart.GetParameters(), parameters);

                if (heterogeneityCorrection != null)
                {
                    parameters = heterogeneityCorrection.Correct(
                        clientId,
                        step,
                        globalAtStepStart.GetParameters(),
                        parameters,
                        localEpochs);
                }

                if (useCompression)
                {
                    var clientRandom = FederatedRandom.CreateClientRandom(_randomSeed, step, clientId, salt: 4242);
                    parameters = ApplyCompressionToParameters(
                        clientId,
                        globalAtStepStart.GetParameters(),
                        parameters,
                        compressionOptions!,
                        compressionResiduals,
                        clientRandom,
                        out var uploadRatio);

                    uploadRatioSum += uploadRatio;
                    uploadRatioCount++;
                }

                if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Local || dpMode == DifferentialPrivacyMode.LocalAndCentral))
                {
                    parameters = dpMechanism!.ApplyPrivacy(parameters, dpEpsilon, dpDelta);
                }

                int delay = 0;
                if (maxDelay > 0)
                {
                    var delayRandom = FederatedRandom.CreateClientRandom(_randomSeed, step, clientId, salt: 9001);
                    delay = delayRandom.Next(0, maxDelay + 1);
                }

                pending.Add((clientId, parameters, weight, step, step + delay));
            }

            if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Local || dpMode == DifferentialPrivacyMode.LocalAndCentral))
            {
                privacyAccountant!.AddRound(dpEpsilon, dpDelta, samplingRate: (double)selectedClientIds.Count / GetNumberOfClientsOrThrow());
                privacyEventsThisStep++;
            }

            var due = pending
                .Where(u => u.ArrivalStep <= step)
                .OrderBy(u => u.ArrivalStep)
                .ThenBy(u => u.ClientId)
                .ToList();

            pending.RemoveAll(u => u.ArrivalStep <= step);

            if (rejectStale > 0)
            {
                due = due.Where(u => (step - u.StartStep) <= rejectStale).ToList();
            }

            if (mode == FederatedAsyncMode.FedAsync)
            {
                foreach (var update in due)
                {
                    int staleness = step - update.StartStep;
                    double stalenessWeight = ComputeStalenessWeight(staleness, asyncOptions);
                    double alpha = Clamp01(mixingRate * stalenessWeight);

                    var currentModel = GetGlobalModel();

                    var targetParameters = update.Parameters;
                    if (useHomomorphicEncryption)
                    {
                        var singleParams = new Dictionary<int, Vector<T>> { [update.ClientId] = update.Parameters };
                        var singleWeights = new Dictionary<int, double> { [update.ClientId] = update.Weight };
                        var heAggregated = heProvider!.AggregateEncryptedWeightedAverage(
                            singleParams,
                            singleWeights,
                            currentModel.GetParameters(),
                            encryptedIndices,
                            heOptions!);

                        var maskedPlain = MaskEncryptedIndices(update.Parameters, currentModel.GetParameters(), encryptedIndices);
                        foreach (var idx in encryptedIndices)
                        {
                            maskedPlain[idx] = heAggregated[idx];
                        }

                        targetParameters = maskedPlain;
                    }

                    var mixed = MixParameters(currentModel.GetParameters(), targetParameters, alpha);
                    var targetModel = currentModel.WithParameters(mixed);

                    if (serverOptimizer != null)
                    {
                        var updatedParams = serverOptimizer.Step(currentModel.GetParameters(), targetModel.GetParameters());
                        targetModel = targetModel.WithParameters(updatedParams);
                    }

                    if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Central || dpMode == DifferentialPrivacyMode.LocalAndCentral))
                    {
                        var globalParams = targetModel.GetParameters();
                        var privateGlobalParams = dpMechanism!.ApplyPrivacy(globalParams, dpEpsilon, dpDelta);
                        privacyAccountant!.AddRound(dpEpsilon, dpDelta, samplingRate: (double)selectedClientIds.Count / GetNumberOfClientsOrThrow());
                        privacyEventsThisStep++;
                        targetModel = targetModel.WithParameters(privateGlobalParams);
                    }

                    SetGlobalModel(targetModel);
                }
            }
            else if (mode == FederatedAsyncMode.FedBuff)
            {
                foreach (var update in due)
                {
                    var parametersForPlain = useHomomorphicEncryption
                        ? MaskEncryptedIndices(update.Parameters, globalAtStepStart.GetParameters(), encryptedIndices)
                        : update.Parameters;

                    bufferModels[bufferedUpdateKey] = globalAtStepStart.WithParameters(parametersForPlain);
                    bufferWeights[bufferedUpdateKey] = update.Weight;
                    if (useHomomorphicEncryption)
                    {
                        bufferHeParameters![bufferedUpdateKey] = update.Parameters;
                    }
                    bufferedUpdateKey++;
                }

                if (bufferModels.Count >= bufferSize)
                {
                    var aggregated = aggregator.Aggregate(bufferModels, bufferWeights);
                    var currentModel = GetGlobalModel();
                    var newGlobalModel = aggregated;

                    if (useHomomorphicEncryption)
                    {
                        var heAggregated = heProvider!.AggregateEncryptedWeightedAverage(
                            bufferHeParameters!,
                            bufferWeights,
                            globalAtStepStart.GetParameters(),
                            encryptedIndices,
                            heOptions!);

                        var merged = newGlobalModel.GetParameters();
                        foreach (var idx in encryptedIndices)
                        {
                            merged[idx] = heAggregated[idx];
                        }

                        newGlobalModel = newGlobalModel.WithParameters(merged);
                    }

                    if (serverOptimizer != null)
                    {
                        var updatedParams = serverOptimizer.Step(currentModel.GetParameters(), newGlobalModel.GetParameters());
                        newGlobalModel = newGlobalModel.WithParameters(updatedParams);
                    }

                    if (useDifferentialPrivacy && (dpMode == DifferentialPrivacyMode.Central || dpMode == DifferentialPrivacyMode.LocalAndCentral))
                    {
                        var globalParams = newGlobalModel.GetParameters();
                        var privateGlobalParams = dpMechanism!.ApplyPrivacy(globalParams, dpEpsilon, dpDelta);
                        privacyAccountant!.AddRound(dpEpsilon, dpDelta, samplingRate: (double)selectedClientIds.Count / GetNumberOfClientsOrThrow());
                        privacyEventsThisStep++;
                        newGlobalModel = newGlobalModel.WithParameters(privateGlobalParams);
                    }

                    SetGlobalModel(newGlobalModel);
                    bufferModels.Clear();
                    bufferWeights.Clear();
                    bufferHeParameters?.Clear();
                }
            }
            else
            {
                throw new InvalidOperationException($"Unknown async federated learning mode '{mode}'. Supported values: None, FedAsync, FedBuff.");
            }

            metadata.RoundsCompleted = step + 1;
            metadata.TotalClientsParticipated = uniqueParticipants.Count;
            if (privacyAccountant != null)
            {
                metadata.TotalPrivacyBudgetConsumed = privacyAccountant.GetTotalEpsilonConsumed();
                metadata.TotalPrivacyDeltaConsumed = privacyAccountant.GetTotalDeltaConsumed();
                metadata.ReportedDelta = dpDelta;
                metadata.ReportedEpsilonAtDelta = privacyAccountant.GetEpsilonAtDelta(dpDelta);
            }

            var newParams = GetGlobalModel().GetParameters();
            var deltaNorm = ComputeL2Distance(previousGlobalParams, newParams);

            UpdateClientPerformanceScores(selectedClientIds, startedClientWeights);

            double averageUploadRatio = uploadRatioCount > 0 ? uploadRatioSum / uploadRatioCount : 1.0;
            double stepCommunicationMB = EstimateRoundCommunicationMB(selectedClientIds.Count, globalAtStepStart.ParameterCount, averageUploadRatio);
            metadata.TotalCommunicationMB += stepCommunicationMB;
            metadata.RoundMetrics.Add(new RoundMetadata
            {
                RoundNumber = step,
                SelectedClientIds = selectedClientIds,
                RoundTimeSeconds = (DateTime.UtcNow - stepStart).TotalSeconds,
                GlobalLoss = double.NaN,
                GlobalAccuracy = double.NaN,
                AverageLocalLoss = double.NaN,
                CommunicationMB = stepCommunicationMB,
                UploadCompressionRatio = averageUploadRatio,
                PrivacyBudgetConsumed = useDifferentialPrivacy ? privacyEventsThisStep * dpEpsilon : 0.0
            });

            previousGlobalParams = newParams;

            if (step + 1 >= _minRoundsBeforeConvergence && deltaNorm <= _convergenceThreshold)
            {
                metadata.Converged = true;
                metadata.ConvergenceRound = step + 1;
                metadata.Notes = $"Converged by parameter delta L2 <= {_convergenceThreshold:0.########}.";
                break;
            }
        }
    }

    private static double Clamp01(double value)
    {
        if (value < 0.0)
        {
            return 0.0;
        }

        if (value > 1.0)
        {
            return 1.0;
        }

        return value;
    }

    private static string GetHomomorphicEncryptionSchemeName(HomomorphicEncryptionScheme scheme)
    {
        switch (scheme)
        {
            case HomomorphicEncryptionScheme.Ckks:
                return "CKKS";

            case HomomorphicEncryptionScheme.Bfv:
                return "BFV";

            default:
                return scheme.ToString();
        }
    }

    private static string GetHomomorphicEncryptionModeName(HomomorphicEncryptionMode mode)
    {
        switch (mode)
        {
            case HomomorphicEncryptionMode.HeOnly:
                return "HEOnly";

            case HomomorphicEncryptionMode.Hybrid:
                return "Hybrid";

            default:
                return mode.ToString();
        }
    }

    private static double ComputeStalenessWeight(int staleness, AsyncFederatedLearningOptions? options)
    {
        if (staleness <= 0)
        {
            return 1.0;
        }

        FederatedStalenessWeighting mode = options?.StalenessWeighting ?? FederatedStalenessWeighting.Inverse;
        double rate = options?.StalenessDecayRate ?? 1.0;

        if (mode == FederatedStalenessWeighting.Constant)
        {
            return 1.0;
        }

        if (mode == FederatedStalenessWeighting.Inverse)
        {
            return 1.0 / (1.0 + staleness);
        }

        if (mode == FederatedStalenessWeighting.Exponential)
        {
            return Math.Exp(-rate * staleness);
        }

        if (mode == FederatedStalenessWeighting.Polynomial)
        {
            return 1.0 / Math.Pow(1.0 + staleness, rate);
        }

        throw new InvalidOperationException($"Unknown staleness weighting '{mode}'. Supported values: Constant, Inverse, Exponential, Polynomial.");
    }

    private Vector<T> MixParameters(Vector<T> current, Vector<T> target, double alpha)
    {
        if (current.Length != target.Length)
        {
            throw new ArgumentException("Parameter vectors must have the same length for async mixing.");
        }

        if (alpha <= 0.0)
        {
            return current;
        }

        if (alpha >= 1.0)
        {
            return target;
        }

        var result = new Vector<T>(current.Length);
        var alphaT = NumOps.FromDouble(alpha);
        for (int i = 0; i < current.Length; i++)
        {
            var diff = NumOps.Subtract(target[i], current[i]);
            result[i] = NumOps.Add(current[i], NumOps.Multiply(alphaT, diff));
        }

        return result;
    }

    private static IFederatedServerOptimizer<T>? CreateDefaultServerOptimizer(FederatedServerOptimizerOptions? options)
    {
        if (options == null)
        {
            return null;
        }

        switch (options.Optimizer)
        {
            case FederatedServerOptimizer.None:
                return null;

            case FederatedServerOptimizer.FedAvgM:
                return new FedAvgMServerOptimizer<T>(options.LearningRate, options.Momentum);

            case FederatedServerOptimizer.FedAdagrad:
                return new FedAdagradServerOptimizer<T>(options.LearningRate, options.Epsilon);

            case FederatedServerOptimizer.FedAdam:
                return new FedAdamServerOptimizer<T>(options.LearningRate, options.Beta1, options.Beta2, options.Epsilon);

            case FederatedServerOptimizer.FedYogi:
                return new FedYogiServerOptimizer<T>(options.LearningRate, options.Beta1, options.Beta2, options.Epsilon);

            default:
                throw new InvalidOperationException($"Unknown server optimizer '{options.Optimizer}'. Supported values: None, FedAvgM, FedAdagrad, FedAdam, FedYogi.");
        }
    }

    private static IFederatedHeterogeneityCorrection<T>? CreateDefaultHeterogeneityCorrection(FederatedHeterogeneityCorrectionOptions? options)
    {
        if (options == null)
        {
            return null;
        }

        switch (options.Algorithm)
        {
            case FederatedHeterogeneityCorrection.None:
                return null;

            case FederatedHeterogeneityCorrection.Scaffold:
                return new ScaffoldHeterogeneityCorrection<T>(options.ClientLearningRate);

            case FederatedHeterogeneityCorrection.FedNova:
                return new FedNovaHeterogeneityCorrection<T>();

            case FederatedHeterogeneityCorrection.FedDyn:
                return new FedDynHeterogeneityCorrection<T>(options.FedDynAlpha);

            default:
                throw new InvalidOperationException($"Unknown heterogeneity correction '{options.Algorithm}'. Supported values: None, Scaffold, FedNova, FedDyn.");
        }
    }

    private static IPrivacyAccountant CreateDefaultPrivacyAccountant(FederatedPrivacyAccountant accountant, double clipNorm)
    {
        switch (accountant)
        {
            case FederatedPrivacyAccountant.Basic:
                return new BasicCompositionPrivacyAccountant();

            case FederatedPrivacyAccountant.Rdp:
                return new RdpPrivacyAccountant(clipNorm);

            default:
                throw new InvalidOperationException($"Unknown privacy accountant '{accountant}'. Supported values: Basic, Rdp.");
        }
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
            weights[clientId] = clientData.TryGetValue(clientId, out var dataset)
                ? Math.Max(1.0, dataset.SampleCount)
                : 1.0;
        }

        var flOptions = _federatedLearningOptions;
        var selectionOptions = flOptions?.ClientSelection;
        FederatedClientSelectionStrategy strategyName = selectionOptions?.Strategy ?? FederatedClientSelectionStrategy.UniformRandom;

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

    private static IClientSelectionStrategy CreateBuiltInSelectionStrategy(FederatedClientSelectionStrategy name, ClientSelectionOptions? options)
    {
        if (name == FederatedClientSelectionStrategy.UniformRandom)
        {
            return new UniformRandomClientSelectionStrategy();
        }

        if (name == FederatedClientSelectionStrategy.WeightedRandom)
        {
            return new WeightedRandomClientSelectionStrategy();
        }

        if (name == FederatedClientSelectionStrategy.Stratified)
        {
            return new StratifiedClientSelectionStrategy();
        }

        if (name == FederatedClientSelectionStrategy.AvailabilityAware)
        {
            var threshold = options?.AvailabilityThreshold ?? 0.0;
            return new AvailabilityAwareClientSelectionStrategy(threshold);
        }

        if (name == FederatedClientSelectionStrategy.PerformanceAware)
        {
            var rate = options?.ExplorationRate ?? 0.1;
            return new PerformanceAwareClientSelectionStrategy(rate);
        }

        if (name == FederatedClientSelectionStrategy.Clustered)
        {
            int clusters = options?.ClusterCount ?? 3;
            int iterations = options?.KMeansIterations ?? 5;
            return new ClusteredClientSelectionStrategy(clusters, iterations);
        }

        throw new InvalidOperationException($"Unknown client selection strategy '{name}'. Supported values: UniformRandom, WeightedRandom, Stratified, AvailabilityAware, PerformanceAware, Clustered.");
    }

    private void UpdateClientPerformanceScores(List<int> selectedClientIds, Dictionary<int, double> clientWeights)
    {
        foreach (var (id, _, weight) in selectedClientIds
            .Select(id => (Id: id, HasWeight: clientWeights.TryGetValue(id, out var w), Weight: w))
            .Where(x => x.HasWeight))
        {
            // Initial, simple proxy score: prefer clients that contributed more data.
            // More advanced scoring (e.g., validation improvement per update) can be layered on later.
            _clientPerformanceScores[id] = weight;
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

    private static double EstimateRoundCommunicationMB(int selectedClientCount, int parameterCount, double uploadRatio = 1.0)
    {
        if (selectedClientCount <= 0 || parameterCount <= 0)
        {
            return 0.0;
        }

        uploadRatio = Math.Max(0.0, Math.Min(1.0, uploadRatio));
        int bytesPerParam = EstimateBytesPerNumericType(typeof(T));
        long bytesPerVector = (long)parameterCount * bytesPerParam;

        // Approximate: each selected client downloads global params (uncompressed) + uploads one update vector (optionally compressed).
        long uploadBytes = (long)Math.Round(bytesPerVector * uploadRatio);
        long totalBytes = (long)selectedClientCount * (bytesPerVector + uploadBytes);
        return totalBytes / 1_000_000.0;
    }

    private static FederatedCompressionOptions? ResolveCompressionOptions(FederatedLearningOptions? options)
    {
        if (options == null)
        {
            return null;
        }

        if (options.Compression != null)
        {
            return options.Compression;
        }

        if (options.UseCompression)
        {
            return new FederatedCompressionOptions
            {
                Strategy = "TopK",
                Ratio = options.CompressionRatio,
                UseErrorFeedback = true
            };
        }

        return null;
    }

    private static FederatedPersonalizationOptions? ResolvePersonalizationOptions(FederatedLearningOptions? options)
    {
        if (options == null)
        {
            return null;
        }

        if (options.Personalization != null)
        {
            return options.Personalization;
        }

        if (options.EnablePersonalization)
        {
            return new FederatedPersonalizationOptions
            {
                Enabled = true,
                Strategy = "FedPer",
                PersonalizedParameterFraction = options.PersonalizationLayerFraction,
                LocalAdaptationEpochs = 0
            };
        }

        return null;
    }

    private static bool IsHeadSplitPersonalization(string strategy) =>
        string.Equals(strategy, "FedPer", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(strategy, "FedRep", StringComparison.OrdinalIgnoreCase);

    private static bool IsClusteredPersonalization(string strategy) =>
        string.Equals(strategy, "Clustered", StringComparison.OrdinalIgnoreCase);

    private static bool IsDittoPersonalization(string strategy) =>
        string.Equals(strategy, "Ditto", StringComparison.OrdinalIgnoreCase);

    private static bool IsPFedMePersonalization(string strategy) =>
        string.Equals(strategy, "pFedMe", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(strategy, "PFedMe", StringComparison.OrdinalIgnoreCase);

    private static int[] ResolvePersonalizedIndices(double fraction, int parameterCount)
    {
        if (parameterCount <= 0)
        {
            return Array.Empty<int>();
        }

        if (double.IsNaN(fraction) || double.IsInfinity(fraction))
        {
            fraction = 0.0;
        }

        fraction = Math.Max(0.0, Math.Min(1.0, fraction));
        int personalizedCount = (int)Math.Ceiling(parameterCount * fraction);
        personalizedCount = Math.Max(0, Math.Min(parameterCount, personalizedCount));

        if (personalizedCount == 0)
        {
            return Array.Empty<int>();
        }

        var indices = new int[personalizedCount];
        int start = parameterCount - personalizedCount;
        for (int i = 0; i < personalizedCount; i++)
        {
            indices[i] = start + i;
        }

        return indices;
    }

    private IFullModel<T, TInput, TOutput> CreatePersonalizedStartModel(
        string personalizationStrategy,
        FederatedPersonalizationOptions options,
        int clientId,
        IFullModel<T, TInput, TOutput> globalModel,
        Vector<T> globalParameters,
        int[] personalizedIndices,
        Dictionary<int, Vector<T>> perClientState,
        Dictionary<int, Vector<T>>? perClusterState)
    {
        if (IsClusteredPersonalization(personalizationStrategy))
        {
            int clusters = Math.Max(1, options.ClusterCount);
            int clusterId = GetClusterId(clientId, clusters);
            Vector<T>? clusterVector = null;
            if (perClusterState != null)
            {
                perClusterState.TryGetValue(clusterId, out clusterVector);
            }

            var start = globalParameters.Clone();
            if (clusterVector != null && personalizedIndices.Length > 0)
            {
                foreach (var idx in personalizedIndices)
                {
                    start[idx] = clusterVector[idx];
                }
            }

            return globalModel.WithParameters(start);
        }

        if (IsPFedMePersonalization(personalizationStrategy))
        {
            if (!perClientState.TryGetValue(clientId, out var theta))
            {
                theta = globalParameters.Clone();
                perClientState[clientId] = theta;
            }

            return globalModel.WithParameters(theta);
        }

        if (IsHeadSplitPersonalization(personalizationStrategy))
        {
            if (!perClientState.TryGetValue(clientId, out var clientVector))
            {
                clientVector = globalParameters.Clone();
                perClientState[clientId] = clientVector;
            }

            var start = globalParameters.Clone();
            foreach (var idx in personalizedIndices)
            {
                start[idx] = clientVector[idx];
            }

            return globalModel.WithParameters(start);
        }

        // Ditto (and unknown strategies) start from the current global model for the global update path.
        return globalModel;
    }

    private Vector<T> ApplyPersonalizationAfterLocalTraining(
        string personalizationStrategy,
        FederatedPersonalizationOptions options,
        int clientId,
        Vector<T> globalBaseline,
        Vector<T> trainedParameters,
        int[] personalizedIndices,
        Dictionary<int, Vector<T>> perClientState)
    {
        if (IsHeadSplitPersonalization(personalizationStrategy))
        {
            perClientState[clientId] = trainedParameters;
            return MaskIndices(trainedParameters, globalBaseline, personalizedIndices);
        }

        if (IsClusteredPersonalization(personalizationStrategy))
        {
            // Store client head values so the post-round clustering step can compute cluster-specific heads.
            perClientState[clientId] = trainedParameters;
            return MaskIndices(trainedParameters, globalBaseline, personalizedIndices);
        }

        if (IsPFedMePersonalization(personalizationStrategy))
        {
            if (!perClientState.TryGetValue(clientId, out var theta))
            {
                theta = globalBaseline.Clone();
            }

            double mu = Math.Max(0.0, options.PFedMeMu);
            double alpha = 1.0 / (1.0 + mu);
            int steps = Math.Max(1, options.PFedMeInnerSteps);
            for (int k = 0; k < steps; k++)
            {
                theta = ApplyMetaLearningUpdate(theta, trainedParameters, alpha);
            }

            perClientState[clientId] = theta;
            return theta;
        }

        if (IsDittoPersonalization(personalizationStrategy))
        {
            // Ditto keeps a personalized model per client; the global update remains the standard client-trained parameters.
            // Store the client-trained parameters so the post-round step can apply proximal personalization toward the new global model.
            perClientState[clientId] = trainedParameters;
            return trainedParameters;
        }

        // Unknown strategy: treat as no personalization.
        return trainedParameters;
    }

    private void ApplyPostAggregationPersonalization(
        string personalizationStrategy,
        FederatedPersonalizationOptions options,
        Dictionary<int, FederatedClientDataset<TInput, TOutput>> clientData,
        List<int> selectedClientIds,
        Vector<T> globalAfterAggregation,
        int[] personalizedIndices,
        Dictionary<int, Vector<T>> perClientState,
        Dictionary<int, Vector<T>>? perClusterState)
    {
        if (selectedClientIds.Count == 0)
        {
            return;
        }

        if (IsClusteredPersonalization(personalizationStrategy))
        {
            if (perClusterState == null)
            {
                return;
            }

            int clusters = Math.Max(1, options.ClusterCount);
            var clusterSums = new Dictionary<int, Vector<T>>();
            var clusterWeights = new Dictionary<int, double>();

            foreach (var clientId in selectedClientIds)
            {
                if (!clientData.TryGetValue(clientId, out var dataset))
                {
                    continue;
                }

                if (!perClientState.TryGetValue(clientId, out var clientParams) || clientParams.Length != globalAfterAggregation.Length)
                {
                    continue;
                }

                int clusterId = GetClusterId(clientId, clusters);
                double w = Math.Max(1.0, dataset.SampleCount);

                if (!clusterSums.TryGetValue(clusterId, out var sum))
                {
                    sum = new Vector<T>(new T[globalAfterAggregation.Length]);
                    clusterSums[clusterId] = sum;
                    clusterWeights[clusterId] = 0.0;
                }

                foreach (var idx in personalizedIndices)
                {
                    sum[idx] = NumOps.Add(sum[idx], NumOps.Multiply(clientParams[idx], NumOps.FromDouble(w)));
                }

                clusterWeights[clusterId] += w;
            }

            foreach (var kvp in clusterSums)
            {
                int clusterId = kvp.Key;
                var sum = kvp.Value;
                double totalW = clusterWeights.TryGetValue(clusterId, out var tw) ? tw : 0.0;
                if (totalW <= 0.0)
                {
                    continue;
                }

                var head = globalAfterAggregation.Clone();
                foreach (var idx in personalizedIndices)
                {
                    head[idx] = NumOps.Divide(sum[idx], NumOps.FromDouble(totalW));
                }

                perClusterState[clusterId] = head;
            }
        }

        if (IsDittoPersonalization(personalizationStrategy))
        {
            double lambda = Math.Max(0.0, options.DittoLambda);
            double denom = 1.0 + lambda;
            if (denom <= 0.0)
            {
                return;
            }

            var lambdaT = NumOps.FromDouble(lambda);
            var denomT = NumOps.FromDouble(denom);

            foreach (var clientId in selectedClientIds)
            {
                if (!perClientState.TryGetValue(clientId, out var clientTrained) || clientTrained.Length != globalAfterAggregation.Length)
                {
                    continue;
                }

                // Proximal-style averaging: argmin_theta ||theta - clientTrained||^2 + lambda||theta - global||^2
                var personalized = clientTrained.Clone();
                for (int i = 0; i < personalized.Length; i++)
                {
                    var num = NumOps.Add(clientTrained[i], NumOps.Multiply(lambdaT, globalAfterAggregation[i]));
                    personalized[i] = NumOps.Divide(num, denomT);
                }

                perClientState[clientId] = personalized;
            }
        }

        int adaptationEpochs = Math.Max(0, options.LocalAdaptationEpochs);
        if (adaptationEpochs <= 0)
        {
            return;
        }

        if (!IsHeadSplitPersonalization(personalizationStrategy) && !IsClusteredPersonalization(personalizationStrategy))
        {
            return;
        }

        var globalModel = GetGlobalModel();
        var globalAfterParams = globalAfterAggregation;

        if (IsHeadSplitPersonalization(personalizationStrategy))
        {
            foreach (var clientId in selectedClientIds)
            {
                if (!clientData.TryGetValue(clientId, out var dataset))
                {
                    continue;
                }

                if (!perClientState.TryGetValue(clientId, out var clientParams) || clientParams.Length != globalAfterParams.Length)
                {
                    clientParams = globalAfterParams.Clone();
                }

                var start = globalAfterParams.Clone();
                foreach (var idx in personalizedIndices)
                {
                    start[idx] = clientParams[idx];
                }

                var localModel = CloneModelByParameters(globalModel.WithParameters(start));
                var localOptimizer = CreateOptimizerForModel(localModel);
                ConfigureLocalOptimizer(localOptimizer, adaptationEpochs);
                var input = CreateLocalOptimizationInputData(dataset, localModel);
                var result = localOptimizer.Optimize(input);
                var adapted = (result.BestSolution ?? localModel).GetParameters();

                perClientState[clientId] = adapted;
            }

            return;
        }

        if (IsClusteredPersonalization(personalizationStrategy) && perClusterState != null)
        {
            int clusters = Math.Max(1, options.ClusterCount);
            var clusterSums = new Dictionary<int, Vector<T>>();
            var clusterWeights = new Dictionary<int, double>();

            foreach (var clientId in selectedClientIds)
            {
                if (!clientData.TryGetValue(clientId, out var dataset))
                {
                    continue;
                }

                int clusterId = GetClusterId(clientId, clusters);
                if (!perClusterState.TryGetValue(clusterId, out var clusterHead) || clusterHead.Length != globalAfterParams.Length)
                {
                    clusterHead = globalAfterParams.Clone();
                }

                var start = globalAfterParams.Clone();
                foreach (var idx in personalizedIndices)
                {
                    start[idx] = clusterHead[idx];
                }

                var localModel = CloneModelByParameters(globalModel.WithParameters(start));
                var localOptimizer = CreateOptimizerForModel(localModel);
                ConfigureLocalOptimizer(localOptimizer, adaptationEpochs);
                var input = CreateLocalOptimizationInputData(dataset, localModel);
                var result = localOptimizer.Optimize(input);
                var adapted = (result.BestSolution ?? localModel).GetParameters();

                double w = Math.Max(1.0, dataset.SampleCount);
                if (!clusterSums.TryGetValue(clusterId, out var sum))
                {
                    sum = new Vector<T>(new T[globalAfterParams.Length]);
                    clusterSums[clusterId] = sum;
                    clusterWeights[clusterId] = 0.0;
                }

                foreach (var idx in personalizedIndices)
                {
                    sum[idx] = NumOps.Add(sum[idx], NumOps.Multiply(adapted[idx], NumOps.FromDouble(w)));
                }

                clusterWeights[clusterId] += w;
            }

            foreach (var kvp in clusterSums)
            {
                int clusterId = kvp.Key;
                var sum = kvp.Value;
                double totalW = clusterWeights.TryGetValue(clusterId, out var tw) ? tw : 0.0;
                if (totalW <= 0.0)
                {
                    continue;
                }

                var head = globalAfterParams.Clone();
                foreach (var idx in personalizedIndices)
                {
                    head[idx] = NumOps.Divide(sum[idx], NumOps.FromDouble(totalW));
                }

                perClusterState[clusterId] = head;
            }
        }
    }

    private static int GetClusterId(int clientId, int clusterCount)
    {
        clusterCount = Math.Max(1, clusterCount);
        int mod = clientId % clusterCount;
        return mod < 0 ? mod + clusterCount : mod;
    }

    private Vector<T> ApplyMetaLearningUpdate(Vector<T> baseline, Vector<T> averaged, double metaRate)
    {
        if (baseline.Length != averaged.Length)
        {
            throw new ArgumentException("Vector length mismatch when applying meta-learning update.");
        }

        if (double.IsNaN(metaRate) || double.IsInfinity(metaRate))
        {
            metaRate = 0.0;
        }

        var rateT = NumOps.FromDouble(metaRate);
        var updated = baseline.Clone();
        for (int i = 0; i < updated.Length; i++)
        {
            var diff = NumOps.Subtract(averaged[i], baseline[i]);
            updated[i] = NumOps.Add(baseline[i], NumOps.Multiply(diff, rateT));
        }

        return updated;
    }

    private static Vector<T> MaskIndices(Vector<T> parameters, Vector<T> baseline, int[] indices)
    {
        if (parameters.Length != baseline.Length)
        {
            throw new ArgumentException("Baseline and parameter vectors must have the same length for masking.");
        }

        if (indices == null || indices.Length == 0)
        {
            return parameters;
        }

        var masked = parameters.Clone();
        foreach (var idx in indices.Where(idx => idx >= 0 && idx < masked.Length))
        {
            masked[idx] = baseline[idx];
        }

        return masked;
    }

    private static int[] ResolveEncryptedIndices(HomomorphicEncryptionOptions options, int parameterCount, HomomorphicEncryptionMode mode)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (parameterCount <= 0)
        {
            return Array.Empty<int>();
        }

        if (mode == HomomorphicEncryptionMode.HeOnly)
        {
            var all = new int[parameterCount];
            for (int i = 0; i < parameterCount; i++)
            {
                all[i] = i;
            }
            return all;
        }

        if (mode != HomomorphicEncryptionMode.Hybrid)
        {
            throw new InvalidOperationException($"Unknown homomorphic encryption mode '{mode}'. Supported values: HeOnly, Hybrid.");
        }

        var indices = new List<int>();
        foreach (var range in options.EncryptedRanges ?? new List<ParameterIndexRange>())
        {
            if (range == null || range.Length <= 0)
            {
                continue;
            }

            int start = Math.Max(0, range.Start);
            int endExclusive = start + range.Length;
            if (endExclusive <= 0)
            {
                continue;
            }

            endExclusive = Math.Min(parameterCount, endExclusive);
            for (int i = start; i < endExclusive; i++)
            {
                indices.Add(i);
            }
        }

        return indices.Distinct().OrderBy(i => i).ToArray();
    }

    private Vector<T> MaskEncryptedIndices(Vector<T> parameters, Vector<T> baseline, int[] encryptedIndices)
    {
        if (parameters.Length != baseline.Length)
        {
            throw new ArgumentException("Baseline and parameter vectors must have the same length for masking.");
        }

        if (encryptedIndices == null || encryptedIndices.Length == 0)
        {
            return parameters;
        }

        var masked = parameters.Clone();
        foreach (var idx in encryptedIndices.Where(idx => idx >= 0 && idx < masked.Length))
        {
            masked[idx] = baseline[idx];
        }

        return masked;
    }

    private Vector<T> ApplyCompressionToParameters(
        int clientId,
        Vector<T> globalParameters,
        Vector<T> localParameters,
        FederatedCompressionOptions options,
        Dictionary<int, Vector<T>>? residuals,
        Random random,
        out double uploadRatio)
    {
        if (globalParameters.Length != localParameters.Length)
        {
            throw new ArgumentException("Global and local parameter vectors must have the same length for compression.");
        }

        int n = globalParameters.Length;
        var delta = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            delta[i] = NumOps.Subtract(localParameters[i], globalParameters[i]);
        }

        if (residuals != null && residuals.TryGetValue(clientId, out var residual) && residual.Length == n)
        {
            for (int i = 0; i < n; i++)
            {
                delta[i] = NumOps.Add(delta[i], residual[i]);
            }
        }

        var compressedDelta = CompressDelta(delta, options, random, out uploadRatio);

        if (residuals != null && options.UseErrorFeedback)
        {
            var newResidual = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                newResidual[i] = NumOps.Subtract(delta[i], compressedDelta[i]);
            }

            residuals[clientId] = newResidual;
        }

        var parametersToSend = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            parametersToSend[i] = NumOps.Add(globalParameters[i], compressedDelta[i]);
        }

        return parametersToSend;
    }

    private Vector<T> CompressDelta(Vector<T> delta, FederatedCompressionOptions options, Random random, out double uploadRatio)
    {
        string strategy = options.Strategy?.Trim() ?? "None";
        int n = delta.Length;

        if (n == 0 || string.Equals(strategy, "None", StringComparison.OrdinalIgnoreCase))
        {
            uploadRatio = 1.0;
            return delta;
        }

        if (string.Equals(strategy, "TopK", StringComparison.OrdinalIgnoreCase))
        {
            double ratio = Math.Max(0.0, Math.Min(1.0, options.Ratio));
            int k = Math.Max(1, (int)Math.Round(ratio * n));
            k = Math.Min(k, n);

            var magnitudes = new double[n];
            var indices = new int[n];
            for (int i = 0; i < n; i++)
            {
                indices[i] = i;
                magnitudes[i] = Math.Abs(NumOps.ToDouble(delta[i]));
            }

            Array.Sort(magnitudes, indices);

            var result = new Vector<T>(n);
            for (int t = n - k; t < n; t++)
            {
                int idx = indices[t];
                result[idx] = delta[idx];
            }

            uploadRatio = (double)k / n;
            return result;
        }

        if (string.Equals(strategy, "RandomK", StringComparison.OrdinalIgnoreCase))
        {
            double ratio = Math.Max(0.0, Math.Min(1.0, options.Ratio));
            int k = Math.Max(1, (int)Math.Round(ratio * n));
            k = Math.Min(k, n);

            var indices = new int[n];
            for (int i = 0; i < n; i++)
            {
                indices[i] = i;
            }

            for (int i = 0; i < k; i++)
            {
                int j = i + random.Next(n - i);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            var result = new Vector<T>(n);
            for (int i = 0; i < k; i++)
            {
                int idx = indices[i];
                result[idx] = delta[idx];
            }

            uploadRatio = (double)k / n;
            return result;
        }

        if (string.Equals(strategy, "Threshold", StringComparison.OrdinalIgnoreCase))
        {
            double threshold = Math.Abs(options.Threshold);
            int kept = 0;
            var result = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                double v = NumOps.ToDouble(delta[i]);
                if (Math.Abs(v) >= threshold)
                {
                    result[i] = delta[i];
                    kept++;
                }
            }

            uploadRatio = n > 0 ? (double)kept / n : 1.0;
            return result;
        }

        if (string.Equals(strategy, "UniformQuantization", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(strategy, "StochasticQuantization", StringComparison.OrdinalIgnoreCase))
        {
            int bits = Math.Max(1, Math.Min(16, options.QuantizationBits));
            int levels = (1 << bits) - 1;

            double maxAbs = 0.0;
            for (int i = 0; i < n; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(NumOps.ToDouble(delta[i])));
            }

            if (maxAbs <= 0.0)
            {
                uploadRatio = ComputeQuantizationRatio(bits);
                return delta;
            }

            var result = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                double v = NumOps.ToDouble(delta[i]);
                double normalized = v / maxAbs;
                normalized = Math.Max(-1.0, Math.Min(1.0, normalized));
                double qReal = (normalized + 1.0) * 0.5 * levels;

                double q;
                if (string.Equals(strategy, "StochasticQuantization", StringComparison.OrdinalIgnoreCase))
                {
                    double floor = Math.Floor(qReal);
                    double prob = qReal - floor;
                    q = (random.NextDouble() < prob) ? floor + 1.0 : floor;
                }
                else
                {
                    q = Math.Round(qReal);
                }

                q = Math.Max(0.0, Math.Min(levels, q));
                double deNormalized = (q / levels) * 2.0 - 1.0;
                double dequantized = deNormalized * maxAbs;
                result[i] = NumOps.FromDouble(dequantized);
            }

            uploadRatio = ComputeQuantizationRatio(bits);
            return result;
        }

        throw new InvalidOperationException($"Unknown compression strategy '{strategy}'. Supported values: None, TopK, RandomK, Threshold, UniformQuantization, StochasticQuantization.");
    }

    private static double ComputeQuantizationRatio(int bits)
    {
        int bytesPerParam = EstimateBytesPerNumericType(typeof(T));
        int fullBits = Math.Max(1, bytesPerParam * 8);
        return Math.Max(0.0, Math.Min(1.0, (double)bits / fullBits));
    }

    private static int EstimateBytesPerNumericType(Type numericType)
    {
        try
        {
            return System.Runtime.InteropServices.Marshal.SizeOf(numericType);
        }
        catch (ArgumentException)
        {
            // Conservative default for unknown numeric types (treat as double-like).
            return 8;
        }
        catch (NotSupportedException)
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
