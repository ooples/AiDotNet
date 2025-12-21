using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.TabularQLearning;

/// <summary>
/// Tabular Q-Learning agent using lookup table for Q-values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Tabular Q-Learning is the foundational RL algorithm that maintains a table
/// of Q-values for each state-action pair. No neural networks required.
/// </para>
/// <para><b>For Beginners:</b>
/// Q-Learning is like creating a cheat sheet: for every situation (state) and
/// action you could take, you write down how good that choice is (Q-value).
/// Over time, you update this sheet based on actual rewards you receive.
///
/// Key features:
/// - **Off-Policy**: Learns optimal policy while following exploratory policy
/// - **Tabular**: Uses lookup table, no function approximation
/// - **Model-Free**: Doesn't need to know environment dynamics
/// - **Value-Based**: Learns action values, derives policy from them
///
/// Perfect for: Small discrete state/action spaces (grid worlds, simple games)
/// Famous for: Watkins 1989, the foundation of modern RL
/// </para>
/// </remarks>
public class TabularQLearningAgent<T> : ReinforcementLearningAgentBase<T>
{
    private TabularQLearningOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Random _random;
    private double _epsilon;

    public TabularQLearningAgent(TabularQLearningOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _random = RandomHelper.CreateSecureRandom();
        _epsilon = _options.EpsilonStart;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = VectorToStateKey(state);

        // Epsilon-greedy exploration
        if (training && _random.NextDouble() < _epsilon)
        {
            // Random action
            int randomAction = _random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        // Greedy action selection
        int bestAction = GetBestAction(stateKey);
        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = VectorToStateKey(state);
        string nextStateKey = VectorToStateKey(nextState);
        int actionIndex = GetActionIndex(action);

        // Ensure state exists in Q-table
        EnsureStateExists(stateKey);
        EnsureStateExists(nextStateKey);

        // Q-Learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        T currentQ = _qTable[stateKey][actionIndex];
        T maxNextQ = done ? NumOps.Zero : GetMaxQValue(nextStateKey);

        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, maxNextQ));
        T tdError = NumOps.Subtract(target, currentQ);
        T update = NumOps.Multiply(LearningRate, tdError);

        _qTable[stateKey][actionIndex] = NumOps.Add(currentQ, update);

        // Decay epsilon
        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
    }

    public override T Train()
    {
        // Tabular Q-learning updates immediately in StoreExperience
        // No separate training step needed
        return NumOps.Zero;
    }

    private string VectorToStateKey(Vector<T> state)
    {
        // Convert state vector to string key for dictionary
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

    private T GetMaxQValue(string stateKey)
    {
        EnsureStateExists(stateKey);

        T maxValue = _qTable[stateKey][0];
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(_qTable[stateKey][a], maxValue))
            {
                maxValue = _qTable[stateKey][a];
            }
        }

        return maxValue;
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
        // Get state keys BEFORE clearing to preserve them
        var stateKeys = _qTable.Keys.ToList();
        int maxStates = parameters.Length / _options.ActionSize;

        // Update Q-values for existing states
        for (int i = 0; i < Math.Min(maxStates, stateKeys.Count); i++)
        {
            if (_qTable.ContainsKey(stateKeys[i]))
            {
                for (int action = 0; action < _options.ActionSize; action++)
                {
                    int idx = i * _options.ActionSize + action;
                    _qTable[stateKeys[i]][action] = parameters[idx];
                }
            }
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new TabularQLearningAgent<T>(_options);

        // Deep copy the Q-table
        clone._qTable = new Dictionary<string, Dictionary<int, T>>();
        foreach (var stateEntry in _qTable)
        {
            var actionDict = new Dictionary<int, T>();
            foreach (var actionEntry in stateEntry.Value)
            {
                actionDict[actionEntry.Key] = actionEntry.Value;
            }
            clone._qTable[stateEntry.Key] = actionDict;
        }

        clone._epsilon = _epsilon;
        return clone;
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        // Tabular methods don't use gradients
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
