using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private double _epsilon;
    private Random _random;

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
        _random = RandomHelper.CreateSecureRandom();
        _lastState = null;
        _lastAction = null;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = VectorToStateKey(state);

        // Epsilon-greedy exploration
        int actionIndex;
        if (training && _random.NextDouble() < _epsilon)
        {
            // Random action
            actionIndex = _random.Next(_options.ActionSize);
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
            ModelType = ModelType.ReinforcementLearning,
        };
    }

    public override int ParameterCount => _qTable.Count * _options.ActionSize;

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        var state = new
        {
            QTable = _qTable,
            Epsilon = _epsilon,
            Options = _options
        };
        string json = JsonConvert.SerializeObject(state);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }

    public override void Deserialize(byte[] data)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentException("Serialized data cannot be null or empty", nameof(data));
        }

        string json = System.Text.Encoding.UTF8.GetString(data);
        var state = JsonConvert.DeserializeObject<dynamic>(json);
        if (state is null)
        {
            throw new InvalidOperationException("Deserialization returned null");
        }

        _qTable = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.QTable.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
        _epsilon = state.Epsilon;
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
        // Tabular RL methods cannot restore Q-values from parameters alone
        // because the parameter vector contains only Q-values, not state keys.
        //
        // For a fresh agent (empty Q-table), state keys are unknown, so restoration fails.
        // For proper save/load, use Serialize()/Deserialize() which preserves state mappings.
        //
        // This is a fundamental limitation of tabular methods - unlike neural networks,
        // the "parameters" (Q-values) are meaningless without their state associations.

        throw new NotSupportedException(
            "Tabular SARSA agents do not support parameter restoration without state information. " +
            "Use Serialize()/Deserialize() methods instead, which preserve state-to-Q-value mappings.");
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new SARSAAgent<T>(_options);

        // Deep copy Q-table to avoid shared state
        foreach (var kvp in _qTable)
        {
            clone._qTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

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
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        }

        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void LoadModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        }

        if (!System.IO.File.Exists(filepath))
        {
            throw new System.IO.FileNotFoundException($"Model file not found: {filepath}", filepath);
        }

        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
