using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.MonteCarlo;

/// <summary>
/// Off-Policy Monte Carlo Control agent with weighted importance sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Off-Policy MC uses importance sampling to learn an optimal policy (target)
/// while following a different exploratory policy (behavior).
/// </remarks>
public class OffPolicyMonteCarloAgent<T> : ReinforcementLearningAgentBase<T>
{
    private OffPolicyMonteCarloOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, T>> _cTable;  // Cumulative weights
    private List<(Vector<T> state, int action, T reward)> _episode;
    private Random _random;

    public OffPolicyMonteCarloAgent(OffPolicyMonteCarloOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _cTable = new Dictionary<string, Dictionary<int, T>>();
        _episode = new List<(Vector<T>, int, T)>();
        _random = RandomHelper.CreateSecureRandom();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

        int selectedAction;

        if (training && _random.NextDouble() < _options.BehaviorEpsilon)
        {
            // Behavior policy: epsilon-greedy exploration
            selectedAction = _random.Next(_options.ActionSize);
        }
        else
        {
            // Target policy: greedy
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
        T W = NumOps.One;

        // Process episode backward for weighted importance sampling
        for (int t = _episode.Count - 1; t >= 0; t--)
        {
            var (state, action, reward) = _episode[t];
            G = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, G));

            string stateKey = GetStateKey(state);
            EnsureStateExists(state);

            // Update cumulative weight
            _cTable[stateKey][action] = NumOps.Add(_cTable[stateKey][action], W);

            // Weighted importance sampling update: Q(S,A) ← Q(S,A) + (W/C(S,A)) * (G - Q(S,A))
            var currentQ = _qTable[stateKey][action];
            var error = NumOps.Subtract(G, currentQ);
            var weightRatio = NumOps.Divide(W, _cTable[stateKey][action]);
            var increment = NumOps.Multiply(weightRatio, error);
            _qTable[stateKey][action] = NumOps.Add(currentQ, increment);

            // Get greedy action according to target policy
            int greedyAction = GetGreedyAction(state);

            // If behavior action != target action, break (importance sampling ratio becomes 0)
            if (action != greedyAction)
            {
                break;
            }

            // Update importance sampling ratio
            // π(a|s) / b(a|s) where π is greedy (prob=1) and b is epsilon-greedy
            double behaviorProb = (1.0 - _options.BehaviorEpsilon) + (_options.BehaviorEpsilon / _options.ActionSize);
            W = NumOps.Divide(W, NumOps.FromDouble(behaviorProb));
        }
    }

    private int GetGreedyAction(Vector<T> state)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

        int greedyAction = 0;
        T bestValue = _qTable[stateKey][0];

        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(_qTable[stateKey][a], bestValue))
            {
                bestValue = _qTable[stateKey][a];
                greedyAction = a;
            }
        }

        return greedyAction;
    }

    private void EnsureStateExists(Vector<T> state)
    {
        string stateKey = GetStateKey(state);

        if (!_qTable.ContainsKey(stateKey))
        {
            _qTable[stateKey] = new Dictionary<int, T>();
            _cTable[stateKey] = new Dictionary<int, T>();

            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[stateKey][a] = NumOps.Zero;
                _cTable[stateKey][a] = NumOps.Zero;
            }
        }
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
            ["states_visited"] = NumOps.FromDouble(_qTable.Count),
            ["episode_length"] = NumOps.FromDouble(_episode.Count)
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

    public override int FeatureCount => _options.ActionSize;

    public override byte[] Serialize()
    {
        var state = new
        {
            QTable = _qTable,
            CTable = _cTable,
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
        _cTable = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.CTable.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
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
        var clone = new OffPolicyMonteCarloAgent<T>(_options);

        // Deep copy Q-table and C-table to avoid shared state
        foreach (var kvp in _qTable)
        {
            clone._qTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        foreach (var kvp in _cTable)
        {
            clone._cTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
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
