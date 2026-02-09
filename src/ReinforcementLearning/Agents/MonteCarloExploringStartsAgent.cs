using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.MonteCarlo;

/// <summary>
/// Monte Carlo Exploring Starts agent for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Monte Carlo ES ensures exploration by starting each episode from a randomly
/// chosen state-action pair, then following the greedy policy thereafter.
/// </remarks>
public class MonteCarloExploringStartsAgent<T> : ReinforcementLearningAgentBase<T>
{
    private MonteCarloExploringStartsOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, List<T>>> _returns;
    private List<(Vector<T> state, int action, T reward)> _episode;
    private bool _isFirstAction;
    private Random _random;

    public MonteCarloExploringStartsAgent(MonteCarloExploringStartsOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _returns = new Dictionary<string, Dictionary<int, List<T>>>();
        _episode = new List<(Vector<T>, int, T)>();
        _isFirstAction = true;
        _random = Random;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        if (_isFirstAction && training)
        {
            // Exploring start: random action for first step
            _isFirstAction = false;
            int randomAction = _random.Next(_options.ActionSize);
            var action = new Vector<T>(_options.ActionSize);
            action[randomAction] = NumOps.One;
            return action;
        }

        // Greedy action selection based on Q-table
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

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

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int actionIndex = ArgMax(action);
        _episode.Add((state, actionIndex, reward));

        if (done)
        {
            UpdateFromEpisode();
            _episode.Clear();
            _isFirstAction = true;
        }
    }

    public override T Train()
    {
        // Training happens during episode completion in StoreExperience
        return NumOps.Zero;
    }

    private void UpdateFromEpisode()
    {
        T G = NumOps.Zero;
        var visited = new HashSet<string>();

        // Process episode backward
        for (int t = _episode.Count - 1; t >= 0; t--)
        {
            var (state, action, reward) = _episode[t];
            G = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, G));

            string stateKey = GetStateKey(state);
            string stateActionKey = $"{stateKey}_{action}";

            // First-visit MC: only update first occurrence
            if (!visited.Contains(stateActionKey))
            {
                visited.Add(stateActionKey);

                EnsureStateExists(state);
                if (!_returns.ContainsKey(stateKey))
                {
                    _returns[stateKey] = new Dictionary<int, List<T>>();
                }
                if (!_returns[stateKey].ContainsKey(action))
                {
                    _returns[stateKey][action] = new List<T>();
                }

                _returns[stateKey][action].Add(G);
                _qTable[stateKey][action] = ComputeAverage(_returns[stateKey][action]);
            }
        }
    }

    private void EnsureStateExists(Vector<T> state)
    {
        string stateKey = GetStateKey(state);

        if (!_qTable.ContainsKey(stateKey))
        {
            _qTable[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private string GetStateKey(Vector<T> state)
    {
        return string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    }

    private T ComputeAverage(List<T> values)
    {
        if (values.Count == 0)
        {
            return NumOps.Zero;
        }

        T sum = NumOps.Zero;
        foreach (var value in values)
        {
            sum = NumOps.Add(sum, value);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(values.Count));
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
            ["states_visited"] = NumOps.FromDouble(_qTable.Count),
            ["episode_length"] = NumOps.FromDouble(_episode.Count)
        };
    }

    public override void ResetEpisode()
    {
        _episode.Clear();
        _isFirstAction = true;
    }

    public override Vector<T> Predict(Vector<T> input)
    {
        _isFirstAction = false;
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

    public override int ParameterCount => _qTable.Count * _options.ActionSize;

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        var state = new
        {
            QTable = _qTable,
            Returns = _returns,
            Options = _options,
            IsFirstAction = _isFirstAction
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
        _returns = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, List<T>>>>(state.Returns.ToString()) ?? new Dictionary<string, Dictionary<int, List<T>>>();

        // Safely parse IsFirstAction with backward compatibility
        // Default to true if field is missing to preserve exploring-starts behavior
        _isFirstAction = true;
        if (state.IsFirstAction is not null)
        {
            if (state.IsFirstAction is bool boolValue)
            {
                _isFirstAction = boolValue;
            }
            else if (bool.TryParse(state.IsFirstAction.ToString(), out bool parsedValue))
            {
                _isFirstAction = parsedValue;
            }
        }
    }

    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();
        foreach (var stateEntry in _qTable)
        {
            foreach (var actionValue in stateEntry.Value)
            {
                paramsList.Add(actionValue.Value);
            }
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
        int index = 0;
        foreach (var stateEntry in _qTable.ToList())
        {
            for (int a = 0; a < _options.ActionSize; a++)
            {
                if (index < parameters.Length)
                {
                    _qTable[stateEntry.Key][a] = parameters[index];
                    index++;
                }
            }
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new MonteCarloExploringStartsAgent<T>(_options);

        // Deep copy Q-table and returns to avoid shared state
        foreach (var kvp in _qTable)
        {
            clone._qTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        foreach (var kvp in _returns)
        {
            clone._returns[kvp.Key] = new Dictionary<int, List<T>>();
            foreach (var returnKvp in kvp.Value)
            {
                clone._returns[kvp.Key][returnKvp.Key] = new List<T>(returnKvp.Value);
            }
        }

        // Preserve mid-episode state
        clone._isFirstAction = this._isFirstAction;

        return clone;
    }

    public override Vector<T> ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        // Loss computation not used in Monte Carlo methods

        var gradient = usedLossFunction.CalculateDerivative(prediction, target);
        return gradient;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Monte Carlo methods don't use gradients in the traditional sense
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
