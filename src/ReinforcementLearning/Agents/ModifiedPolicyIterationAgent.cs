using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.DynamicProgramming;

/// <summary>
/// Helper class for serializing model transition data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TransitionData<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string NextState { get; set; } = string.Empty;
    public T Reward { get; set; }
    public T Probability { get; set; }

    public TransitionData()
    {
        Reward = NumOps.Zero;
        Probability = NumOps.Zero;
    }
}

/// <summary>
/// Modified Policy Iteration agent - hybrid of Policy Iteration and Value Iteration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Modified PI performs limited policy evaluation sweeps before improvement,
/// trading off between the efficiency of VI and the stability of PI.
/// </remarks>
public class ModifiedPolicyIterationAgent<T> : ReinforcementLearningAgentBase<T>
{
    private ModifiedPolicyIterationOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, T> _valueTable;
    private Dictionary<string, int> _policy;
    private Dictionary<string, Dictionary<int, List<(string nextState, T reward, T probability)>>> _model;
    private Random _random;

    public ModifiedPolicyIterationAgent(ModifiedPolicyIterationOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _valueTable = new Dictionary<string, T>();
        _policy = new Dictionary<string, int>();
        _model = new Dictionary<string, Dictionary<int, List<(string, T, T)>>>();
        _random = RandomHelper.CreateSecureRandom();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = GetStateKey(state);

        if (!_policy.ContainsKey(stateKey))
        {
            _policy[stateKey] = _random.Next(_options.ActionSize);
            _valueTable[stateKey] = NumOps.Zero;
        }

        int selectedAction = _policy[stateKey];

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
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

        // Store transition with count of 1 initially
        // Probabilities will be normalized when computing expected values
        _model[stateKey][actionIndex].Add((nextStateKey, reward, NumOps.One));
    }

    public override T Train()
    {
        if (_model.Count == 0)
        {
            return NumOps.Zero;
        }

        bool policyStable = false;
        int iterations = 0;

        while (!policyStable && iterations < 100)
        {
            // Modified Policy Evaluation (limited sweeps)
            ModifiedPolicyEvaluation();

            // Policy Improvement
            policyStable = PolicyImprovement();

            iterations++;
        }

        return NumOps.FromDouble(iterations);
    }

    private void ModifiedPolicyEvaluation()
    {
        // Only do k sweeps instead of iterating to convergence
        for (int sweep = 0; sweep < _options.MaxEvaluationSweeps; sweep++)
        {
            foreach (var stateKey in _valueTable.Keys.ToList())
            {
                if (!_policy.ContainsKey(stateKey))
                {
                    continue;
                }

                int action = _policy[stateKey];
                T newValue = ComputeActionValue(stateKey, action);
                _valueTable[stateKey] = newValue;
            }
        }
    }

    private bool PolicyImprovement()
    {
        bool policyStable = true;

        foreach (var stateKey in _policy.Keys.ToList())
        {
            int oldAction = _policy[stateKey];

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

            _policy[stateKey] = bestAction;

            if (oldAction != bestAction)
            {
                policyStable = false;
            }
        }

        return policyStable;
    }

    private T ComputeActionValue(string stateKey, int action)
    {
        if (!_model.ContainsKey(stateKey) || !_model[stateKey].ContainsKey(action))
        {
            return NumOps.Zero;
        }

        T expectedValue = NumOps.Zero;

        // Normalize probabilities by total count to prevent blow-up
        var transitions = _model[stateKey][action];
        T totalCount = NumOps.FromDouble(transitions.Count);

        foreach (var (nextStateKey, reward, probability) in transitions)
        {
            T nextValue = NumOps.Zero;
            if (_valueTable.ContainsKey(nextStateKey))
            {
                nextValue = _valueTable[nextStateKey];
            }

            // Normalize probability: each transition gets weight 1/N
            T normalizedProb = NumOps.Divide(probability, totalCount);
            T transitionValue = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextValue));
            expectedValue = NumOps.Add(expectedValue, NumOps.Multiply(normalizedProb, transitionValue));
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
        // Convert model tuples to serializable format
        var serializableModel = new Dictionary<string, Dictionary<int, List<TransitionData<T>>>>();
        foreach (var stateEntry in _model)
        {
            var actionDict = new Dictionary<int, List<TransitionData<T>>>();
            foreach (var actionEntry in stateEntry.Value)
            {
                var transitionList = new List<TransitionData<T>>();
                foreach (var transition in actionEntry.Value)
                {
                    transitionList.Add(new TransitionData<T>
                    {
                        NextState = transition.nextState,
                        Reward = transition.reward,
                        Probability = transition.probability
                    });
                }
                actionDict[actionEntry.Key] = transitionList;
            }
            serializableModel[stateEntry.Key] = actionDict;
        }

        var state = new
        {
            ValueTable = _valueTable,
            Policy = _policy,
            Model = serializableModel
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
        _policy = JsonConvert.DeserializeObject<Dictionary<string, int>>(state.Policy.ToString()) ?? new Dictionary<string, int>();

        // Deserialize model from serializable format
        var serializableModel = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, List<TransitionData<T>>>>>(state.Model.ToString()) ?? new Dictionary<string, Dictionary<int, List<TransitionData<T>>>>();
        _model = new Dictionary<string, Dictionary<int, List<(string, T, T)>>>();

        foreach (var stateEntry in serializableModel)
        {
            var actionDict = new Dictionary<int, List<(string, T, T)>>();
            foreach (var actionEntry in stateEntry.Value)
            {
                var transitionList = new List<(string, T, T)>();
                foreach (var transition in actionEntry.Value)
                {
                    transitionList.Add((transition.NextState, transition.Reward, transition.Probability));
                }
                actionDict[actionEntry.Key] = transitionList;
            }
            _model[stateEntry.Key] = actionDict;
        }
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
        var clone = new ModifiedPolicyIterationAgent<T>(_options);

        // Deep copy value table
        foreach (var kvp in _valueTable)
        {
            clone._valueTable[kvp.Key] = kvp.Value;
        }

        // Deep copy policy
        foreach (var kvp in _policy)
        {
            clone._policy[kvp.Key] = kvp.Value;
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
