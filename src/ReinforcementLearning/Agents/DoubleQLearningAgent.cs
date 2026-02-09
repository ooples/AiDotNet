using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.DoubleQLearning;

/// <summary>
/// Double Q-Learning agent using two Q-tables to reduce overestimation bias.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Double Q-Learning maintains two Q-tables and uses one to select actions
/// and the other to evaluate them, reducing maximization bias.
/// </para>
/// <para><b>For Beginners:</b>
/// Q-Learning tends to overestimate Q-values because it uses max(Q) for both
/// selecting and evaluating actions. Double Q-Learning fixes this by using
/// two separate Q-tables and randomly switching which one is updated.
///
/// Key innovation:
/// - **Two Q-tables**: Q1 and Q2
/// - **Decorrelation**: Use Q1 to select action, Q2 to evaluate (or vice versa)
/// - **Reduced Bias**: Prevents overestimation from max operator
///
/// Famous for: Hado van Hasselt 2010, foundation for Double DQN
/// </para>
/// </remarks>
public class DoubleQLearningAgent<T> : ReinforcementLearningAgentBase<T>
{
    private DoubleQLearningOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable1;
    private Dictionary<string, Dictionary<int, T>> _qTable2;
    private double _epsilon;
    private Random _random;

    public DoubleQLearningAgent(DoubleQLearningOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable1 = new Dictionary<string, Dictionary<int, T>>();
        _qTable2 = new Dictionary<string, Dictionary<int, T>>();
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
            // Use sum of both Q-tables for action selection
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

        // Randomly choose which Q-table to update
        bool updateQ1 = _random.NextDouble() < 0.5;

        if (updateQ1)
        {
            // Update Q1 using Q2 for evaluation
            T currentQ = _qTable1[stateKey][actionIndex];

            if (done)
            {
                T target = reward;
                T tdError = NumOps.Subtract(target, currentQ);
                T update = NumOps.Multiply(LearningRate, tdError);
                _qTable1[stateKey][actionIndex] = NumOps.Add(currentQ, update);
            }
            else
            {
                // Use Q1 to select action, Q2 to evaluate
                int bestAction = GetBestActionFromTable(_qTable1, nextStateKey);
                T nextQ = _qTable2[nextStateKey][bestAction];
                T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextQ));
                T tdError = NumOps.Subtract(target, currentQ);
                T update = NumOps.Multiply(LearningRate, tdError);
                _qTable1[stateKey][actionIndex] = NumOps.Add(currentQ, update);
            }
        }
        else
        {
            // Update Q2 using Q1 for evaluation
            T currentQ = _qTable2[stateKey][actionIndex];

            if (done)
            {
                T target = reward;
                T tdError = NumOps.Subtract(target, currentQ);
                T update = NumOps.Multiply(LearningRate, tdError);
                _qTable2[stateKey][actionIndex] = NumOps.Add(currentQ, update);
            }
            else
            {
                // Use Q2 to select action, Q1 to evaluate
                int bestAction = GetBestActionFromTable(_qTable2, nextStateKey);
                T nextQ = _qTable1[nextStateKey][bestAction];
                T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextQ));
                T tdError = NumOps.Subtract(target, currentQ);
                T update = NumOps.Multiply(LearningRate, tdError);
                _qTable2[stateKey][actionIndex] = NumOps.Add(currentQ, update);
            }
        }

        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
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
        if (!_qTable1.ContainsKey(stateKey))
        {
            _qTable1[stateKey] = new Dictionary<int, T>();
            _qTable2[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable1[stateKey][a] = NumOps.Zero;
                _qTable2[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private int GetBestAction(string stateKey)
    {
        EnsureStateExists(stateKey);
        int bestAction = 0;
        T bestValue = NumOps.Add(_qTable1[stateKey][0], _qTable2[stateKey][0]);

        for (int a = 1; a < _options.ActionSize; a++)
        {
            T sumValue = NumOps.Add(_qTable1[stateKey][a], _qTable2[stateKey][a]);
            if (NumOps.GreaterThan(sumValue, bestValue))
            {
                bestValue = sumValue;
                bestAction = a;
            }
        }
        return bestAction;
    }

    private int GetBestActionFromTable(Dictionary<string, Dictionary<int, T>> qTable, string stateKey)
    {
        int bestAction = 0;
        T bestValue = qTable[stateKey][0];

        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(qTable[stateKey][a], bestValue))
            {
                bestValue = qTable[stateKey][a];
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

    public override int ParameterCount => _qTable1.Count * _options.ActionSize * 2;
    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        var state = new
        {
            QTable1 = _qTable1,
            QTable2 = _qTable2,
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

        _qTable1 = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.QTable1.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
        _qTable2 = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.QTable2.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
        _epsilon = state.Epsilon;
    }

    public override Vector<T> GetParameters()
    {
        // Flatten both Q-tables into vector using linear indexing
        // Vector size: stateCount * 2 * actionSize
        int stateCount = Math.Max(_qTable1.Count, 1);
        int vectorSize = stateCount * 2 * _options.ActionSize;
        var parameters = new Vector<T>(vectorSize);

        // Fill _qTable1 values (indices 0 to stateCount*actionSize-1)
        int idx = 0;
        foreach (var stateQValues in _qTable1.Values)
        {
            for (int action = 0; action < _options.ActionSize; action++)
            {
                parameters[idx++] = stateQValues[action];
            }
        }

        // Fill _qTable2 values (indices stateCount*actionSize to stateCount*2*actionSize-1)
        foreach (var stateQValues in _qTable2.Values)
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
        // For a fresh agent (empty Q-tables), state keys are unknown, so restoration fails.
        // For proper save/load, use Serialize()/Deserialize() which preserves state mappings.
        //
        // This is a fundamental limitation of tabular methods - unlike neural networks,
        // the "parameters" (Q-values) are meaningless without their state associations.

        throw new NotSupportedException(
            "Tabular Double Q-Learning agents do not support parameter restoration without state information. " +
            "Use Serialize()/Deserialize() methods instead, which preserve state-to-Q-value mappings for both Q-tables.");
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new DoubleQLearningAgent<T>(_options);

        // Deep copy Q-table 1 to avoid shared state
        foreach (var kvp in _qTable1)
        {
            clone._qTable1[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        // Deep copy Q-table 2 to avoid shared state
        foreach (var kvp in _qTable2)
        {
            clone._qTable2[kvp.Key] = new Dictionary<int, T>(kvp.Value);
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
