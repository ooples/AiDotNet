using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.MonteCarlo;

/// <summary>
/// On-Policy Monte Carlo Control agent with epsilon-greedy exploration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// On-Policy MC Control uses epsilon-greedy policy for both behavior and target,
/// ensuring exploration while learning the optimal policy.
/// </remarks>
public class OnPolicyMonteCarloAgent<T> : ReinforcementLearningAgentBase<T>
{
    private OnPolicyMonteCarloOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, List<T>>> _returns;
    private List<(Vector<T> state, int action, T reward)> _episode;
    private double _epsilon;
    private Random _random;

    public OnPolicyMonteCarloAgent(OnPolicyMonteCarloOptions<T> options)
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
        _epsilon = options.EpsilonStart;
        _random = RandomHelper.CreateSecureRandom();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

        int selectedAction;

        if (training && _random.NextDouble() < _epsilon)
        {
            // Explore: random action
            selectedAction = _random.Next(_options.ActionSize);
        }
        else
        {
            // Exploit: greedy action
            selectedAction = 0;
            T bestValue = _qTable[stateKey][0];

            for (int a = 1; a < _options.ActionSize; a++)
            {
                if (NumOps.GreaterThan(_qTable[stateKey][a], bestValue))
                {
                    bestValue = _qTable[stateKey][a];
                    selectedAction = a;
                }
            }
        }

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
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

            // Decay epsilon
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
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

        // Process episode backward (first-visit MC)
        for (int t = _episode.Count - 1; t >= 0; t--)
        {
            var (state, action, reward) = _episode[t];
            G = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, G));

            string stateKey = GetStateKey(state);
            string stateActionKey = $"{stateKey}_{action}";

            // First-visit: only update first occurrence
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
            ["episode_length"] = NumOps.FromDouble(_episode.Count),
            ["epsilon"] = NumOps.FromDouble(_epsilon)
        };
    }

    public override void ResetEpisode()
    {
        _episode.Clear();
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

    public override int ParameterCount => _qTable.Count * _options.ActionSize;

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        var state = new
        {
            QTable = _qTable,
            Returns = _returns,
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
        _returns = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, List<T>>>>(state.Returns.ToString()) ?? new Dictionary<string, Dictionary<int, List<T>>>();
        _epsilon = state.Epsilon;
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
        var clone = new OnPolicyMonteCarloAgent<T>(_options);

        // Deep copy Q-table
        foreach (var kvp in _qTable)
        {
            clone._qTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        // Deep copy returns
        foreach (var kvp in _returns)
        {
            clone._returns[kvp.Key] = new Dictionary<int, List<T>>();
            foreach (var returnKvp in kvp.Value)
            {
                clone._returns[kvp.Key][returnKvp.Key] = new List<T>(returnKvp.Value);
            }
        }

        clone._epsilon = _epsilon;
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
        throw new NotSupportedException(
            "OnPolicyMonteCarlo persistence is not yet fully supported. " +
            "Use Serialize() for manual serialization or GetParameters() to extract Q-values.");
    }

    public override void LoadModel(string filepath)
    {
        throw new NotSupportedException(
            "OnPolicyMonteCarlo persistence is not yet fully supported. " +
            "Use Deserialize() for manual deserialization or SetParameters() for parameter restoration.");
    }
}
