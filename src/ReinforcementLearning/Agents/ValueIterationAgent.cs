using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.DynamicProgramming;

/// <summary>
/// Value Iteration agent for reinforcement learning using dynamic programming.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Value Iteration combines policy evaluation and improvement in a single update step,
/// converging to the optimal value function.
/// </remarks>
public class ValueIterationAgent<T> : ReinforcementLearningAgentBase<T>
{
    private ValueIterationOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, T> _valueTable;
    private Dictionary<string, Dictionary<int, List<(string nextState, T reward, T probability)>>> _model;
    private Random _random;

    public ValueIterationAgent(ValueIterationOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _valueTable = new Dictionary<string, T>();
        _model = new Dictionary<string, Dictionary<int, List<(string, T, T)>>>();
        _random = RandomHelper.CreateSecureRandom();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = GetStateKey(state);

        // Initialize value for new states
        if (!_valueTable.ContainsKey(stateKey))
        {
            _valueTable[stateKey] = NumOps.Zero;
        }

        // Select action greedily with respect to value function
        int bestAction = 0;
        T bestValue = NumOps.FromDouble(double.NegativeInfinity);

        for (int a = 0; a < _options.ActionSize; a++)
        {
            T actionValue = ComputeActionValue(stateKey, a);

            if (NumOps.GreaterThan(actionValue, bestValue))
            {
                bestValue = actionValue;
                bestAction = a;
            }
        }

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        // Build model from experience
        string stateKey = GetStateKey(state);
        string nextStateKey = GetStateKey(nextState);
        int actionIndex = ArgMax(action);

        if (!_model.ContainsKey(stateKey))
        {
            _model[stateKey] = new Dictionary<int, List<(string, T, T)>>();
        }

        if (!_model[stateKey].ContainsKey(actionIndex))
        {
            _model[stateKey][actionIndex] = new List<(string, T, T)>();
        }

        // For deterministic transitions, replace existing transition instead of accumulating
        // This prevents duplicate entries for the same (state, action) pair
        var transitions = _model[stateKey][actionIndex];

        // Check if this exact transition already exists
        bool exists = false;
        for (int i = 0; i < transitions.Count; i++)
        {
            if (transitions[i].Item1 == nextStateKey)
            {
                // Update existing transition with latest reward
                transitions[i] = (nextStateKey, reward, NumOps.One);
                exists = true;
                break;
            }
        }

        // Add new transition if it doesn't exist
        if (!exists)
        {
            transitions.Add((nextStateKey, reward, NumOps.One));
        }
    }

    public override T Train()
    {
        if (_model.Count == 0)
        {
            return NumOps.Zero;
        }

        T delta;
        int iterations = 0;

        do
        {
            delta = NumOps.Zero;

            foreach (var stateKey in _valueTable.Keys.ToList())
            {
                T oldValue = _valueTable[stateKey];

                // Find max action value (Bellman optimality equation)
                T maxActionValue = NumOps.FromDouble(double.NegativeInfinity);

                for (int a = 0; a < _options.ActionSize; a++)
                {
                    T actionValue = ComputeActionValue(stateKey, a);

                    if (NumOps.GreaterThan(actionValue, maxActionValue))
                    {
                        maxActionValue = actionValue;
                    }
                }

                _valueTable[stateKey] = maxActionValue;

                // Track maximum change
                T diff = NumOps.Subtract(maxActionValue, oldValue);
                T absDiff = NumOps.GreaterThanOrEquals(diff, NumOps.Zero) ? diff : NumOps.Negate(diff);
                if (NumOps.GreaterThan(absDiff, delta))
                {
                    delta = absDiff;
                }
            }

            iterations++;
        }
        while (NumOps.GreaterThanOrEquals(delta, NumOps.FromDouble(_options.Theta)) && iterations < _options.MaxIterations);

        return NumOps.FromDouble(iterations);
    }

    private T ComputeActionValue(string stateKey, int action)
    {
        if (!_model.ContainsKey(stateKey) || !_model[stateKey].ContainsKey(action))
        {
            return NumOps.Zero;
        }

        T expectedValue = NumOps.Zero;

        foreach (var (nextStateKey, reward, probability) in _model[stateKey][action])
        {
            T nextValue = NumOps.Zero;
            if (_valueTable.ContainsKey(nextStateKey))
            {
                nextValue = _valueTable[nextStateKey];
            }

            T transitionValue = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextValue));
            expectedValue = NumOps.Add(expectedValue, NumOps.Multiply(probability, transitionValue));
        }

        return expectedValue;
    }

    private string GetStateKey(Vector<T> state)
    {
        return string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    }

    private int ArgMax(Vector<T> values)
    {
        int maxIndex = 0;
        T maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (NumOps.GreaterThan(values[i], maxValue))
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["states_visited"] = NumOps.FromDouble(_valueTable.Count),
            ["model_transitions"] = NumOps.FromDouble(_model.Count)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-specific state
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ReinforcementLearning,
        };
    }

    public override int ParameterCount => _valueTable.Count;

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        var state = new
        {
            ValueTable = _valueTable,
            Model = _model,
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

        _valueTable = JsonConvert.DeserializeObject<Dictionary<string, T>>(state.ValueTable.ToString()) ?? new Dictionary<string, T>();
        _model = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, List<(string, T, T)>>>>(state.Model.ToString()) ?? new Dictionary<string, Dictionary<int, List<(string, T, T)>>>();
    }

    public override Vector<T> GetParameters()
    {
        // Flatten value table into vector
        var paramsList = new List<T>();
        foreach (var value in _valueTable.Values)
        {
            paramsList.Add(value);
        }

        if (paramsList.Count == 0)
        {
            paramsList.Add(NumOps.Zero);
        }

        var paramsVector = new Vector<T>(paramsList.Count);
        for (int i = 0; i < paramsList.Count; i++)
        {
            paramsVector[i] = paramsList[i];
        }

        return paramsVector;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // Reconstruct value table from vector
        int index = 0;
        foreach (var stateKey in _valueTable.Keys.ToList())
        {
            if (index < parameters.Length)
            {
                _valueTable[stateKey] = parameters[index];
                index++;
            }
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new ValueIterationAgent<T>(_options);

        // Deep copy value table
        foreach (var kvp in _valueTable)
        {
            clone._valueTable[kvp.Key] = kvp.Value;
        }

        // Deep copy model
        foreach (var stateKvp in _model)
        {
            clone._model[stateKvp.Key] = new Dictionary<int, List<(string, T, T)>>();
            foreach (var actionKvp in stateKvp.Value)
            {
                clone._model[stateKvp.Key][actionKvp.Key] = new List<(string, T, T)>(actionKvp.Value);
            }
        }

        return clone;
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(prediction, target);
        var gradient = usedLossFunction.CalculateDerivative(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // DP methods don't use gradients
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
