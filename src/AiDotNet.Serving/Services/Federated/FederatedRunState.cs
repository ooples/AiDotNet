using AiDotNet.Models.Options;

using AiDotNet.Serving.Configuration;

namespace AiDotNet.Serving.Services.Federated;

/// <summary>
/// Represents mutable state for an active federated training run.
/// </summary>
public sealed class FederatedRunState
{
    public FederatedRunState(
        string runId,
        string modelName,
        string modelArtifactPath,
        string runArtifactPath,
        NumericType numericType,
        FederatedLearningOptions options,
        int minClientUpdatesPerRound,
        int parameterCount,
        double[] globalParameters)
    {
        RunId = runId;
        ModelName = modelName;
        ModelArtifactPath = modelArtifactPath;
        RunArtifactPath = runArtifactPath;
        NumericType = numericType;
        Options = options;
        MinClientUpdatesPerRound = Math.Max(1, minClientUpdatesPerRound);
        ParameterCount = parameterCount;
        GlobalParameters = globalParameters;
    }

    internal object SyncRoot { get; } = new();

    public string RunId { get; }
    public string ModelName { get; }
    public string ModelArtifactPath { get; }
    public string RunArtifactPath { get; }
    public NumericType NumericType { get; }
    public FederatedLearningOptions Options { get; }
    public int MinClientUpdatesPerRound { get; }
    public int ParameterCount { get; }

    public int CurrentRound { get; set; } = 0;

    public int NextClientId { get; set; } = 0;

    public HashSet<int> JoinedClients { get; } = new();

    public Dictionary<int, double[]> PendingClientParameters { get; } = new();

    public Dictionary<int, double> PendingClientWeights { get; } = new();

    public double[] GlobalParameters { get; set; }
}
