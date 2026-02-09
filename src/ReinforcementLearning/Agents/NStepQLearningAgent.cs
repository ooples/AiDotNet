using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.NStepQLearning;

/// <summary>
/// N-step Q-Learning agent using multi-step off-policy returns.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NStepQLearningAgent<T> : ReinforcementLearningAgentBase<T>
{
    private NStepQLearningOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private List<(string state, int action, T reward)> _nStepBuffer;
    private double _epsilon;
    private Random _random;

    public NStepQLearningAgent(NStepQLearningOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _nStepBuffer = new List<(string, int, T)>();
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
        int actionIndex = GetActionIndex(action);
        _nStepBuffer.Add((stateKey, actionIndex, reward));

        if (_nStepBuffer.Count >= _options.NSteps || done)
        {
            UpdateNStep(nextState, done);
            if (done)
            {
                _nStepBuffer.Clear();
            }
            else if (_nStepBuffer.Count >= _options.NSteps)
            {
                _nStepBuffer.RemoveAt(0);
            }
        }
        // Epsilon decay moved to ResetEpisode to decay per episode, not per step
    }

    private void UpdateNStep(Vector<T> finalState, bool done)
    {
        if (_nStepBuffer.Count == 0) return;
        var (firstState, firstAction, firstReward) = _nStepBuffer[0];
        EnsureStateExists(firstState);

        T G = NumOps.Zero;
        T discount = NumOps.One;
        for (int i = 0; i < _nStepBuffer.Count; i++)
        {
            G = NumOps.Add(G, NumOps.Multiply(discount, _nStepBuffer[i].reward));
            discount = NumOps.Multiply(discount, DiscountFactor);
        }

        if (!done)
        {
            string finalStateKey = VectorToStateKey(finalState);
            EnsureStateExists(finalStateKey);
            T maxQ = GetMaxQValue(finalStateKey);
            G = NumOps.Add(G, NumOps.Multiply(discount, maxQ));
        }

        T currentQ = _qTable[firstState][firstAction];
        T tdError = NumOps.Subtract(G, currentQ);
        T update = NumOps.Multiply(LearningRate, tdError);
        _qTable[firstState][firstAction] = NumOps.Add(currentQ, update);
    }

    private T GetMaxQValue(string stateKey)
    {
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

    public override T Train() { return NumOps.Zero; }
    public override void ResetEpisode()
    {
        _nStepBuffer.Clear();
        // Decay epsilon per episode, not per step
        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
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
            if (NumOps.GreaterThan(action[i], NumOps.Zero)) return i;
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
        return new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
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
        // Flatten Q-table into vector using linear indexing
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
            "Tabular N-Step Q-Learning agents do not support parameter restoration without state information. " +
            "Use Serialize()/Deserialize() methods instead, which preserve state-to-Q-value mappings.");
    }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new NStepQLearningAgent<T>(_options);

        // Deep copy Q-table to avoid shared state
        foreach (var kvp in _qTable)
        {
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
