using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

/// <summary>
/// Exploration strategy stub that always returns a fixed one-hot action and records every call.
/// Lets a test prove the RL training loop actually routes action selection through the configured
/// exploration strategy (and drives its Reset/Update schedule), rather than dropping it in a field.
/// </summary>
internal sealed class RecordingExplorationStrategy<T> : IExplorationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _forcedActionIndex;
    private readonly int _actionSpaceSize;

    public RecordingExplorationStrategy(int forcedActionIndex, int actionSpaceSize)
    {
        _forcedActionIndex = forcedActionIndex;
        _actionSpaceSize = actionSpaceSize;
    }

    public int GetExplorationActionCalls { get; private set; }
    public int UpdateCalls { get; private set; }
    public int ResetCalls { get; private set; }
    public System.Collections.Generic.List<int> ObservedActionSpaceSizes { get; } = new();

    public Vector<T> GetExplorationAction(Vector<T> state, Vector<T> policyAction, int actionSpaceSize, Random random)
    {
        GetExplorationActionCalls++;
        ObservedActionSpaceSizes.Add(actionSpaceSize);
        var action = new Vector<T>(_actionSpaceSize);
        action[_forcedActionIndex] = NumOps.One;
        return action;
    }

    public void Update() => UpdateCalls++;

    public void Reset() => ResetCalls++;
}

internal sealed class DeterministicBanditEnvironment<T> : IEnvironment<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _maxSteps;
    private int _steps;

    public DeterministicBanditEnvironment(
        int actionSpaceSize = 2,
        int observationSpaceDimension = 1,
        int maxSteps = 1)
    {
        if (actionSpaceSize <= 0)
            throw new ArgumentException("Action space size must be positive.", nameof(actionSpaceSize));
        if (observationSpaceDimension <= 0)
            throw new ArgumentException("Observation space dimension must be positive.", nameof(observationSpaceDimension));
        if (maxSteps <= 0)
            throw new ArgumentException("Max steps must be positive.", nameof(maxSteps));

        _numOps = MathHelper.GetNumericOperations<T>();
        ActionSpaceSize = actionSpaceSize;
        ObservationSpaceDimension = observationSpaceDimension;
        _maxSteps = maxSteps;
    }

    public int ObservationSpaceDimension { get; }
    public int ActionSpaceSize { get; }
    public bool IsContinuousActionSpace => false;

    /// <summary>
    /// Total Reset() calls over this instance's lifetime (not reset per episode).
    /// Lets a test prove THIS environment instance was driven by the training loop.
    /// </summary>
    public int TotalResets { get; private set; }

    /// <summary>
    /// Total Step() calls over this instance's lifetime (not reset per episode).
    /// Lets a test prove THIS environment instance was driven by the training loop.
    /// </summary>
    public int TotalSteps { get; private set; }

    public Vector<T> Reset()
    {
        _steps = 0;
        TotalResets++;
        return new Vector<T>(ObservationSpaceDimension);
    }

    public (Vector<T> NextState, T Reward, bool Done, Dictionary<string, object> Info) Step(Vector<T> action)
    {
        if (action is null)
            throw new ArgumentNullException(nameof(action));

        int actionIndex = ResolveDiscreteActionIndex(action);
        if (actionIndex < 0 || actionIndex >= ActionSpaceSize)
            throw new ArgumentException("Action index out of range.", nameof(action));

        _steps++;
        TotalSteps++;

        var reward = actionIndex == 0 ? _numOps.One : _numOps.Zero;
        var done = _steps >= _maxSteps;
        var nextState = new Vector<T>(ObservationSpaceDimension);
        var info = new Dictionary<string, object>
        {
            ["ActionIndex"] = actionIndex
        };

        return (nextState, reward, done, info);
    }

    public void Seed(int seed)
    {
        // Deterministic environment; seed accepted for interface compatibility.
    }

    public void Close()
    {
    }

    private int ResolveDiscreteActionIndex(Vector<T> action)
    {
        if (action.Length == 1)
            return (int)Math.Round(_numOps.ToDouble(action[0]));

        if (action.Length != ActionSpaceSize)
        {
            throw new ArgumentException(
                $"Action vector length {action.Length} does not match action space size {ActionSpaceSize}.",
                nameof(action));
        }

        int bestIndex = 0;
        double bestValue = _numOps.ToDouble(action[0]);

        for (int i = 1; i < action.Length; i++)
        {
            double value = _numOps.ToDouble(action[i]);
            if (value > bestValue)
            {
                bestValue = value;
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
