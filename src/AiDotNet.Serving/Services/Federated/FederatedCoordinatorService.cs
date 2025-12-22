using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Interfaces;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models.Federated;
using AiDotNet.Serving.Services;
using AiDotNet.Serving.Security;
using AiDotNet.Serving.Security.Attestation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Serving.Services.Federated;

/// <summary>
/// In-process federated coordinator that aggregates client-submitted parameter vectors.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This coordinator is the "server" in federated learning:
/// it collects client updates and combines them into a new global model for the next round.
/// </remarks>
public sealed class FederatedCoordinatorService : IFederatedCoordinatorService
{
    private readonly IFederatedRunStore _runStore;
    private readonly IModelRepository _modelRepository;
    private readonly ILogger<FederatedCoordinatorService> _logger;
    private readonly ITierResolver _tierResolver;
    private readonly ITierPolicyProvider _tierPolicyProvider;
    private readonly IAttestationVerifier _attestationVerifier;
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly IModelArtifactProtector _artifactProtector;
    private readonly IModelArtifactStore _artifactStore;

    public FederatedCoordinatorService(
        IFederatedRunStore runStore,
        IModelRepository modelRepository,
        ILogger<FederatedCoordinatorService> logger,
        ITierResolver tierResolver,
        ITierPolicyProvider tierPolicyProvider,
        IAttestationVerifier attestationVerifier,
        IHttpContextAccessor httpContextAccessor,
        IModelArtifactProtector artifactProtector,
        IModelArtifactStore artifactStore)
    {
        _runStore = runStore ?? throw new ArgumentNullException(nameof(runStore));
        _modelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _tierResolver = tierResolver ?? throw new ArgumentNullException(nameof(tierResolver));
        _tierPolicyProvider = tierPolicyProvider ?? throw new ArgumentNullException(nameof(tierPolicyProvider));
        _attestationVerifier = attestationVerifier ?? throw new ArgumentNullException(nameof(attestationVerifier));
        _httpContextAccessor = httpContextAccessor ?? throw new ArgumentNullException(nameof(httpContextAccessor));
        _artifactProtector = artifactProtector ?? throw new ArgumentNullException(nameof(artifactProtector));
        _artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    }

    public CreateFederatedRunResponse CreateRun(CreateFederatedRunRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        if (string.IsNullOrWhiteSpace(request.ModelName))
        {
            throw new ArgumentException("ModelName is required.", nameof(request));
        }

        var modelInfo = _modelRepository.GetModelInfo(request.ModelName);
        if (modelInfo == null)
        {
            throw new FileNotFoundException($"Model '{request.ModelName}' not found.");
        }

        if (string.IsNullOrWhiteSpace(modelInfo.SourcePath))
        {
            throw new InvalidOperationException($"Model '{request.ModelName}' does not have a source artifact path.");
        }

        var runId = Guid.NewGuid().ToString("N");

        var sourcePath = Path.GetFullPath(modelInfo.SourcePath);
        var (parameterCount, globalParameters) = LoadInitialParameters(modelInfo.NumericType, sourcePath);

        var runArtifactPath = CreateRunArtifactPath(sourcePath, request.ModelName, runId);
        File.Copy(sourcePath, runArtifactPath, overwrite: true);

        var state = new FederatedRunState(
            runId: runId,
            modelName: request.ModelName,
            modelArtifactPath: sourcePath,
            runArtifactPath: runArtifactPath,
            numericType: modelInfo.NumericType,
            options: request.Options ?? new FederatedLearningOptions(),
            minClientUpdatesPerRound: request.MinClientUpdatesPerRound,
            parameterCount: parameterCount,
            globalParameters: globalParameters);

        _runStore.Create(state);

        return new CreateFederatedRunResponse
        {
            RunId = runId,
            CurrentRound = state.CurrentRound,
            ParameterCount = state.ParameterCount
        };
    }

    public async Task<JoinFederatedRunResponse> JoinRunAsync(string runId, JoinFederatedRunRequest request, CancellationToken cancellationToken = default)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var state = GetStateOrThrow(runId);

        var httpContext = _httpContextAccessor.HttpContext;
        if (httpContext is null)
        {
            throw new InvalidOperationException("JoinRunAsync requires an HTTP context.");
        }

        var tier = _tierResolver.ResolveTier(httpContext);
        var policy = _tierPolicyProvider.GetPolicy(tier);

        if (policy.RequireAttestationForJoin)
        {
            if (request.Attestation == null)
            {
                throw new UnauthorizedAccessException("Attestation evidence is required.");
            }

            var result = await _attestationVerifier.VerifyAsync(
                request.Attestation,
                cancellationToken == default ? httpContext.RequestAborted : cancellationToken);

            if (!result.IsSuccess)
            {
                throw new UnauthorizedAccessException(result.FailureReason ?? "Attestation failed.");
            }
        }

        int assignedId;
        lock (state.SyncRoot)
        {
            if (request.ClientId.HasValue && request.ClientId.Value < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(request.ClientId), "ClientId must be non-negative when provided.");
            }

            assignedId = request.ClientId ?? AllocateClientId(state);
            if (!state.JoinedClients.Add(assignedId))
            {
                throw new InvalidOperationException($"Client {assignedId} has already joined run '{state.RunId}'.");
            }

            if (assignedId >= state.NextClientId)
            {
                state.NextClientId = assignedId + 1;
            }

            _runStore.Update(state);
        }

        return new JoinFederatedRunResponse
        {
            ClientId = assignedId,
            CurrentRound = state.CurrentRound
        };
    }

    public FederatedRunParametersResponse GetParameters(string runId, int clientId)
    {
        var state = GetStateOrThrow(runId);

        lock (state.SyncRoot)
        {
            if (!state.JoinedClients.Contains(clientId))
            {
                throw new UnauthorizedAccessException("Client has not joined this run.");
            }

            return new FederatedRunParametersResponse
            {
                RunId = state.RunId,
                RoundNumber = state.CurrentRound,
                ParameterCount = state.ParameterCount,
                Parameters = state.GlobalParameters.ToArray()
            };
        }
    }

    public SubmitFederatedUpdateResponse SubmitUpdate(string runId, SubmitFederatedUpdateRequest request)
    {
        var state = GetStateOrThrow(runId);

        lock (state.SyncRoot)
        {
            if (!state.JoinedClients.Contains(request.ClientId))
            {
                throw new UnauthorizedAccessException("Client has not joined this run.");
            }

            if (request.RoundNumber != state.CurrentRound)
            {
                throw new InvalidOperationException($"Update round mismatch. Expected {state.CurrentRound}, got {request.RoundNumber}.");
            }

            if (request.Parameters == null || request.Parameters.Length == 0)
            {
                throw new ArgumentException("Parameters are required.", nameof(request));
            }

            if (request.Parameters.Length != state.ParameterCount)
            {
                throw new ArgumentException($"Parameter length mismatch. Expected {state.ParameterCount}, got {request.Parameters.Length}.", nameof(request));
            }

            if (request.ClientWeight <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(request.ClientWeight), "ClientWeight must be positive.");
            }

            if (state.PendingClientParameters.ContainsKey(request.ClientId))
            {
                _logger.LogWarning(
                    "Duplicate federated update submission overwriting previous update. RunId={RunId} ClientId={ClientId} Round={RoundNumber}",
                    state.RunId,
                    request.ClientId,
                    state.CurrentRound);
            }

            state.PendingClientParameters[request.ClientId] = request.Parameters;
            state.PendingClientWeights[request.ClientId] = request.ClientWeight;
            _runStore.Update(state);

            return new SubmitFederatedUpdateResponse
            {
                Accepted = true,
                ReceivedUpdatesForRound = state.PendingClientParameters.Count,
                MinUpdatesRequired = state.MinClientUpdatesPerRound
            };
        }
    }

    public AggregateFederatedRoundResponse AggregateRound(string runId)
    {
        var state = GetStateOrThrow(runId);

        lock (state.SyncRoot)
        {
            if (state.PendingClientParameters.Count < state.MinClientUpdatesPerRound)
            {
                throw new InvalidOperationException(
                    $"Not enough updates to aggregate. Received {state.PendingClientParameters.Count}, need {state.MinClientUpdatesPerRound}.");
            }

            int aggregatedClientCount = state.PendingClientParameters.Count;
            var aggregated = state.NumericType switch
            {
                NumericType.Float => AggregateRoundInternal<float>(state),
                NumericType.Decimal => AggregateRoundInternal<decimal>(state),
                _ => AggregateRoundInternal<double>(state)
            };

            PersistRunArtifact(state, aggregated);

            state.GlobalParameters = aggregated;
            state.CurrentRound++;
            state.PendingClientParameters.Clear();
            state.PendingClientWeights.Clear();

            _runStore.Update(state);

            return new AggregateFederatedRoundResponse
            {
                NewCurrentRound = state.CurrentRound,
                AggregatedClientCount = aggregatedClientCount
            };
        }
    }

    public FederatedRunStatusResponse GetStatus(string runId)
    {
        var state = GetStateOrThrow(runId);
        lock (state.SyncRoot)
        {
            return new FederatedRunStatusResponse
            {
                RunId = state.RunId,
                CurrentRound = state.CurrentRound,
                ParameterCount = state.ParameterCount,
                JoinedClients = state.JoinedClients.Count,
                UpdatesReceivedForCurrentRound = state.PendingClientParameters.Count,
                MinUpdatesRequired = state.MinClientUpdatesPerRound
            };
        }
    }

    private FederatedRunState GetStateOrThrow(string runId)
    {
        if (string.IsNullOrWhiteSpace(runId))
        {
            throw new ArgumentException("RunId is required.", nameof(runId));
        }

        if (!_runStore.TryGet(runId, out var state) || state == null)
        {
            throw new FileNotFoundException($"Federated run '{runId}' not found.");
        }

        return state;
    }

    private static int AllocateClientId(FederatedRunState state)
    {
        int candidate = state.NextClientId;
        while (state.JoinedClients.Contains(candidate))
        {
            candidate++;
        }

        state.NextClientId = candidate + 1;
        return candidate;
    }

    private static (int parameterCount, double[] globalParameters) LoadInitialParameters(NumericType numericType, string sourcePath)
    {
        if (numericType == NumericType.Float)
        {
            var model = new PredictionModelResult<float, Matrix<float>, Vector<float>>();
            model.LoadFromFile(sourcePath);
            var p = model.GetParameters();
            return (p.Length, p.Select(v => (double)v).ToArray());
        }

        if (numericType == NumericType.Decimal)
        {
            var model = new PredictionModelResult<decimal, Matrix<decimal>, Vector<decimal>>();
            model.LoadFromFile(sourcePath);
            var p = model.GetParameters();
            return (p.Length, p.Select(v => (double)v).ToArray());
        }

        var modelDouble = new PredictionModelResult<double, Matrix<double>, Vector<double>>();
        modelDouble.LoadFromFile(sourcePath);
        var pd = modelDouble.GetParameters();
        return (pd.Length, pd.ToArray());
    }

    private static double[] AggregateRoundInternal<T>(FederatedRunState state)
    {
        var globalBaseline = ToVector<T>(state.GlobalParameters);
        var model = new PredictionModelResult<T, Matrix<T>, Vector<T>>();
        model.LoadFromFile(state.ModelArtifactPath);
        var globalModel = model.WithParameters(globalBaseline);

        var clientModels = new Dictionary<int, IFullModel<T, Matrix<T>, Vector<T>>>();
        var clientWeights = new Dictionary<int, double>();

        foreach (var kvp in state.PendingClientParameters)
        {
            int clientId = kvp.Key;
            var clientParams = ToVector<T>(kvp.Value);
            clientModels[clientId] = (IFullModel<T, Matrix<T>, Vector<T>>)globalModel.WithParameters(clientParams);
            clientWeights[clientId] = state.PendingClientWeights[clientId];
        }

        var aggregator = CreateAggregationStrategy<T>(state.Options);
        var aggregated = aggregator.Aggregate(clientModels, clientWeights);
        var p = aggregated.GetParameters();
        return p.Select(v => Convert.ToDouble(v)).ToArray();
    }

    private static IAggregationStrategy<IFullModel<T, Matrix<T>, Vector<T>>> CreateAggregationStrategy<T>(FederatedLearningOptions options)
    {
        switch (options.AggregationStrategy)
        {
            case FederatedAggregationStrategy.FedAvg:
                return new AiDotNet.FederatedLearning.Aggregators.FedAvgFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>();

            case FederatedAggregationStrategy.FedProx:
                return new AiDotNet.FederatedLearning.Aggregators.FedProxFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>(options.ProximalMu);

            case FederatedAggregationStrategy.FedBN:
                return new AiDotNet.FederatedLearning.Aggregators.FedBNFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>();

            case FederatedAggregationStrategy.Median:
                return new AiDotNet.FederatedLearning.Aggregators.MedianFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>();

            case FederatedAggregationStrategy.TrimmedMean:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.TrimmedMeanFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>(robust.TrimFraction);
            }

            case FederatedAggregationStrategy.WinsorizedMean:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.WinsorizedMeanFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>(robust.TrimFraction);
            }

            case FederatedAggregationStrategy.Rfa:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return AiDotNet.FederatedLearning.Aggregators.RfaFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>.FromOptions(robust);
            }

            case FederatedAggregationStrategy.Krum:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.KrumFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>(robust.ByzantineClientCount);
            }

            case FederatedAggregationStrategy.MultiKrum:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.MultiKrumFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>(
                    robust.ByzantineClientCount,
                    robust.MultiKrumSelectionCount,
                    robust.UseClientWeightsWhenAveragingSelectedUpdates);
            }

            case FederatedAggregationStrategy.Bulyan:
            {
                var robust = options.RobustAggregation ?? new RobustAggregationOptions();
                return new AiDotNet.FederatedLearning.Aggregators.BulyanFullModelAggregationStrategy<T, Matrix<T>, Vector<T>>(
                    robust.ByzantineClientCount,
                    robust.UseClientWeightsWhenAveragingSelectedUpdates);
            }

            default:
                throw new InvalidOperationException($"Unsupported federated aggregation strategy '{options.AggregationStrategy}'.");
        }
    }

    private static Vector<T> ToVector<T>(double[] values)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var v = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            v[i] = ops.FromDouble(values[i]);
        }

        return v;
    }

    public string GetRunArtifactPath(string runId)
    {
        var state = GetStateOrThrow(runId);
        if (string.IsNullOrWhiteSpace(state.RunArtifactPath) || !File.Exists(state.RunArtifactPath))
        {
            throw new FileNotFoundException("Federated run artifact not found.");
        }

        return state.RunArtifactPath;
    }

    public ProtectedModelArtifact GetOrCreateEncryptedRunArtifact(string runId)
    {
        var state = GetStateOrThrow(runId);
        lock (state.SyncRoot)
        {
            if (string.IsNullOrWhiteSpace(state.RunArtifactPath) || !File.Exists(state.RunArtifactPath))
            {
                throw new FileNotFoundException("Federated run artifact not found.");
            }

            var runArtifactPath = state.RunArtifactPath;
            var directory = Path.GetDirectoryName(runArtifactPath) ?? throw new InvalidOperationException("Invalid run artifact path.");
            var protectedDir = Path.Combine(directory, ".protected");
            var storeKey = $"run:{state.RunId}";

            return _artifactStore.GetOrCreate(storeKey, () => _artifactProtector.ProtectToFile(state.RunId, runArtifactPath, protectedDir));
        }
    }

    private static string CreateRunArtifactPath(string sourcePath, string modelName, string runId)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            throw new ArgumentException("Source path is required.", nameof(sourcePath));
        }

        var root = Path.GetDirectoryName(Path.GetFullPath(sourcePath)) ?? throw new InvalidOperationException("Invalid model artifact path.");
        var runDir = Path.Combine(root, ".runs");
        Directory.CreateDirectory(runDir);

        var ext = Path.GetExtension(sourcePath);
        if (string.IsNullOrWhiteSpace(ext))
        {
            ext = ".model";
        }

        var safeName = SanitizeFileName(modelName);
        var fileName = $"{safeName}-{runId}{ext}";
        var candidatePath = Path.GetFullPath(Path.Combine(runDir, fileName));

        var runDirFull = Path.GetFullPath(runDir);
        if (!runDirFull.EndsWith(Path.DirectorySeparatorChar.ToString()) &&
            !runDirFull.EndsWith(Path.AltDirectorySeparatorChar.ToString()))
        {
            runDirFull += Path.DirectorySeparatorChar;
        }

        if (!candidatePath.StartsWith(runDirFull, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Run artifact path resolves outside of the run directory.");
        }

        return candidatePath;
    }

    private static string SanitizeFileName(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = (name ?? string.Empty).Select(c => invalid.Contains(c) ? '_' : c).ToArray();
        var sanitized = new string(chars).Trim();
        return string.IsNullOrWhiteSpace(sanitized) ? "model" : sanitized;
    }

    private static void PersistRunArtifact(FederatedRunState state, double[] globalParameters)
    {
        if (state == null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        if (globalParameters == null)
        {
            throw new ArgumentNullException(nameof(globalParameters));
        }

        var artifactPath = state.RunArtifactPath;
        if (string.IsNullOrWhiteSpace(artifactPath))
        {
            throw new InvalidOperationException("Run artifact path is not set.");
        }

        if (state.NumericType == NumericType.Float)
        {
            PersistTypedArtifact<float>(artifactPath, globalParameters);
            return;
        }

        if (state.NumericType == NumericType.Decimal)
        {
            PersistTypedArtifact<decimal>(artifactPath, globalParameters);
            return;
        }

        PersistTypedArtifact<double>(artifactPath, globalParameters);
    }

    private static void PersistTypedArtifact<T>(string artifactPath, double[] globalParameters)
    {
        var model = new PredictionModelResult<T, Matrix<T>, Vector<T>>();
        model.LoadFromFile(artifactPath);
        var updated = model.WithParameters(ToVector<T>(globalParameters));
        updated.SaveModel(artifactPath);
    }
}
