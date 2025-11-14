using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.SARSA;

/// <summary>
/// SARSA (State-Action-Reward-State-Action) agent using tabular methods.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SARSA is an on-policy TD control algorithm that learns Q-values based on
/// the action actually taken by the current policy, not the optimal action.
/// </para>
/// <para><b>For Beginners:</b>
/// SARSA is like Q-Learning's more cautious cousin. While Q-Learning learns
/// the optimal policy assuming perfect future actions, SARSA learns based on
/// what you actually do (including exploratory mistakes).
///
/// Key differences from Q-Learning:
/// - **On-Policy**: Learns from actions it actually takes
/// - **More Conservative**: Safer in risky environments (cliff walking)
/// - **Exploration Aware**: Updates reflect exploration strategy
/// - **Convergence**: Converges to optimal policy only if exploration decreases
///
/// Update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
/// (Uses actual next action a', not max)
///
/// Perfect for: Environments where safety matters, risky state transitions
/// Famous for: Rummery & Niranjan 1994, on-policy TD control
/// </para>
/// </remarks>
public class SARSAAgent<T> : ReinforcementLearningAgentBase<T>
{
    private SARSAOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private double _epsilon;

    // Track last state-action for SARSA update
    private Vector<T>? _lastState;
    private Vector<T>? _lastAction;

    public SARSAAgent(SARSAOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _epsilon = _options.EpsilonStart;
        _lastState = null;
        _lastAction = null;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = VectorToStateKey(state);

        // Epsilon-greedy exploration
        int actionIndex;
        if (training && Random.NextDouble() < _epsilon)
        {
            // Random action
            actionIndex = Random.Next(_options.ActionSize);
        }
        else
        {
            // Greedy action
            actionIndex = GetBestAction(stateKey);
        }

        var action = new Vector<T>(_options.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // SARSA: (s, a, r, s', a') - need to select next action first
        string stateKey = VectorToStateKey(state);
        string nextStateKey = VectorToStateKey(nextState);
        int actionIndex = GetActionIndex(action);

        EnsureStateExists(stateKey);
        EnsureStateExists(nextStateKey);

        // Get next action using current policy (on-policy)
        Vector<T> nextAction = SelectAction(nextState, training: true);
        int nextActionIndex = GetActionIndex(nextAction);

        // SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        T currentQ = _qTable[stateKey][actionIndex];
        T nextQ = done ? NumOps.Zero : _qTable[nextStateKey][nextActionIndex];

        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextQ));
        T tdError = NumOps.Subtract(target, currentQ);
        T update = NumOps.Multiply(LearningRate, tdError);

        _qTable[stateKey][actionIndex] = NumOps.Add(currentQ, update);

        // Decay epsilon
        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);

        // Store for next update
        _lastState = state;
        _lastAction = action;
    }

    public override T Train()
    {
        // SARSA updates immediately in StoreExperience
        // No separate training step needed
        return NumOps.Zero;
    }

    public override void ResetEpisode()
    {
        _lastState = null;
        _lastAction = null;
        base.ResetEpisode();
    }

    private string VectorToStateKey(Vector<T> state)
    {
        var parts = new string[state.Length];
        for (int i = 0; i < state.Length; i++)
        {
            parts[i] = NumOps.ToDouble(state[i]).ToString("F4");
        }
        return string.Join(",", parts);
    }

    private int GetActionIndex(Vector<T> action)
    {
        for (int i = 0; i < action.Length; i++)
        {
            if (NumOps.GreaterThan(action[i], NumOps.Zero))
            {
                return i;
            }
        }
        return 0;
    }

    private void EnsureStateExists(string stateKey)
    {
        if (!_qTable.ContainsKey(stateKey))
        {
            _qTable[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private int GetBestAction(string stateKey)
    {
        EnsureStateExists(stateKey);

        int bestAction = 0;
        T bestValue = _qTable[stateKey][0];

        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(_qTable[stateKey][a], bestValue))
            {
                bestValue = _qTable[stateKey][a];
                bestAction = a;
            }
        }

        return bestAction;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = "SARSA",
        };
    }

    public override int ParameterCount => _qTable.Count * _options.ActionSize;

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("SARSA serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("SARSA deserialization not yet implemented");
    }

    public override Vector<T> GetParameters()
    {
        // Flatten Q-table into vector
        int stateCount = _qTable.Count;
        var parameters = new Vector<T>(stateCount * _options.ActionSize);

        int idx = 0;
        foreach (var stateQValues in _qTable.Values)
        {
            for (int action = 0; action < _options.ActionSize; action++)
            {
                parameters[idx++] = stateQValues[action];
            }
        }

        return parameters;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // Reconstruct Q-table from vector
        var stateKeys = _qTable.Keys.ToList();
        _qTable.Clear();

        int maxStates = parameters.Length / _options.ActionSize;

        for (int i = 0; i < Math.Min(maxStates, stateKeys.Count); i++)
        {
            var qValues = new Dictionary<int, T>();
            for (int action = 0; action < _options.ActionSize; action++)
            {
                int idx = i * _options.ActionSize + action;
                qValues[action] = parameters[idx];
            }
            _qTable[stateKeys[i]] = qValues;
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new SARSAAgent<T>(_options);
        clone._qTable = new Dictionary<string, Dictionary<int, T>>(_qTable);
        clone._epsilon = _epsilon;
        return clone;
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        return GetParameters();
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Tabular methods don't use gradients
    }

    public override void SaveModel(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
