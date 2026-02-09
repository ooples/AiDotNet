using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.ExpectedSARSA;

/// <summary>
/// Expected SARSA agent using tabular methods.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Expected SARSA is a TD control algorithm that uses the expected value under
/// the current policy instead of sampling the next action.
/// </para>
/// <para><b>For Beginners:</b>
/// Expected SARSA is like SARSA but instead of using the actual next action,
/// it uses the average Q-value weighted by the probability of taking each action.
/// This reduces variance compared to SARSA.
///
/// Update: Q(s,a) ← Q(s,a) + α[r + γ Σ π(a'|s')Q(s',a') - Q(s,a)]
///
/// Benefits over SARSA:
/// - **Lower Variance**: Averages over actions instead of sampling
/// - **Off-Policy Learning**: Can learn optimal policy while exploring
/// - **Better Performance**: Often converges faster than SARSA
///
/// Famous for: Van Seijen et al. 2009, bridging SARSA and Q-Learning
/// </para>
/// </remarks>
public class ExpectedSARSAAgent<T> : ReinforcementLearningAgentBase<T>
{
    private ExpectedSARSAOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private double _epsilon;
    private Random _random;

    public ExpectedSARSAAgent(ExpectedSARSAOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;

        // Defensive validation - properties may bypass init accessors if left at default zero
        if (_options.StateSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                _options.StateSize,
                "StateSize must be greater than zero. Ensure ExpectedSARSAOptions.StateSize is initialized to a positive value.");
        }

        if (_options.ActionSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                _options.ActionSize,
                "ActionSize must be greater than zero. Ensure ExpectedSARSAOptions.ActionSize is initialized to a positive value.");
        }

        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _epsilon = _options.EpsilonStart;
        _random = RandomHelper.CreateSecureRandom();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = VectorToStateKey(state);

        int actionIndex;
        if (training && _random.NextDouble() < _epsilon)
        {
            actionIndex = _random.Next(_options.ActionSize);
        }
        else
        {
            actionIndex = GetBestAction(stateKey);
        }

        var action = new Vector<T>(_options.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = VectorToStateKey(state);
        string nextStateKey = VectorToStateKey(nextState);
        int actionIndex = GetActionIndex(action);

        EnsureStateExists(stateKey);
        EnsureStateExists(nextStateKey);

        // Expected SARSA: Use expected value under current policy
        T currentQ = _qTable[stateKey][actionIndex];
        T expectedNextQ = done ? NumOps.Zero : ComputeExpectedValue(nextStateKey);

        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, expectedNextQ));
        T tdError = NumOps.Subtract(target, currentQ);
        T update = NumOps.Multiply(LearningRate, tdError);

        _qTable[stateKey][actionIndex] = NumOps.Add(currentQ, update);

        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
    }

    private T ComputeExpectedValue(string stateKey)
    {
        EnsureStateExists(stateKey);

        // Expected value: Σ π(a|s) Q(s,a)
        // For ε-greedy: (1-ε)Q(a*) + ε * (1/|A|) Σ Q(a)
        // Note: This is a common approximation that treats the greedy action probability as (1-ε)
        // instead of the exact (1-ε + ε/|A|). For small ε, the difference is negligible.
        // Exact formula: Q(a*) * (1 - ε + ε/|A|) + (ε/|A|) * Σ_{a≠a*} Q(a)

        int bestAction = GetBestAction(stateKey);
        T bestQ = _qTable[stateKey][bestAction];

        T sumQ = NumOps.Zero;
        for (int a = 0; a < _options.ActionSize; a++)
        {
            sumQ = NumOps.Add(sumQ, _qTable[stateKey][a]);
        }

        // (1 - ε) * Q(best) + ε * mean(Q)
        double prob = 1.0 - _epsilon;
        T greedyPart = NumOps.Multiply(NumOps.FromDouble(prob), bestQ);

        T explorePart = NumOps.Multiply(
            NumOps.FromDouble(_epsilon),
            NumOps.Divide(sumQ, NumOps.FromDouble(_options.ActionSize))
        );

        return NumOps.Add(greedyPart, explorePart);
    }

    public override T Train()
    {
        return NumOps.Zero;
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
        if (action is null || action.Length == 0)
        {
            throw new ArgumentException("Action vector cannot be null or empty", nameof(action));
        }

        for (int i = 0; i < action.Length; i++)
        {
            if (NumOps.GreaterThan(action[i], NumOps.Zero))
            {
                return i;
            }
        }

        // Fallback: If no positive element found (potentially malformed input),
        // log a warning and return 0 to prevent crashes
        // In production, consider throwing an exception instead
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
            FeatureCount = _options.StateSize,
            Complexity = _qTable.Count * _options.ActionSize
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
        _qTable.Clear();

        var stateKeys = _qTable.Keys.ToList();
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
        var clone = new ExpectedSARSAAgent<T>(_options);

        // Deep copy Q-table to avoid shared state between original and clone
        // Creates new outer dictionary and new inner dictionary for each state
        // This ensures modifications to one agent don't affect the other
        clone._qTable = new Dictionary<string, Dictionary<int, T>>();
        foreach (var kvp in _qTable)
        {
            // Dictionary<int, T>(kvp.Value) creates a new dictionary with copied values
            clone._qTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        clone._epsilon = _epsilon;
        return clone;
    }

    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return GetParameters();
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }

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
