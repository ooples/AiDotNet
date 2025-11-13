using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

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
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, T>> _cTable;  // Cumulative weights
    private List<(Vector<T> state, int action, T reward)> _episode;

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
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

        int selectedAction;

        if (training && Random.NextDouble() < _options.BehaviorEpsilon)
        {
            // Behavior policy: epsilon-greedy exploration
            selectedAction = Random.Next(_options.ActionSize);
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

            // Weighted importance sampling update
            var weightedReturn = NumOps.Multiply(W, G);
            var increment = NumOps.Divide(weightedReturn, _cTable[stateKey][action]);
            _qTable[stateKey][action] = NumOps.Add(_qTable[stateKey][action], increment);

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
            ModelType = "OffPolicyMonteCarlo",
            InputSize = _options.StateSize,
            OutputSize = _options.ActionSize,
            ParameterCount = ParameterCount
        };
    }

    public override int ParameterCount => _qTable.Count * _options.ActionSize;

    public override int FeatureCount => _options.StateSize;

    public override byte[] Serialize()
    {
        throw new NotImplementedException("OffPolicyMonteCarlo serialization not yet implemented");
    }

    public override void Deserialize(byte[] data)
    {
        throw new NotImplementedException("OffPolicyMonteCarlo deserialization not yet implemented");
    }

    public override Matrix<T> GetParameters()
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

        return new Matrix<T>(new[] { paramsVector });
    }

    public override void SetParameters(Matrix<T> parameters)
    {
        int index = 0;
        foreach (var stateEntry in _qTable.ToList())
        {
            for (int a = 0; a < _options.ActionSize; a++)
            {
                if (index < parameters.Columns)
                {
                    _qTable[stateEntry.Key][a] = parameters[0, index];
                    index++;
                }
            }
        }
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        return new OffPolicyMonteCarloAgent<T>(_options);
    }

    public override (Matrix<T> Gradients, T Loss) ComputeGradients(
        Vector<T> input,
        Vector<T> target,
        ILossFunction<T>? lossFunction = null)
    {
        var prediction = Predict(input);
        var usedLossFunction = lossFunction ?? LossFunction;
        var loss = usedLossFunction.CalculateLoss(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));

        var gradient = usedLossFunction.CalculateDerivative(new Matrix<T>(new[] { prediction }), new Matrix<T>(new[] { target }));
        return (gradient, loss);
    }

    public override void ApplyGradients(Matrix<T> gradients, T learningRate)
    {
        // Monte Carlo methods don't use gradients in the traditional sense
    }

    public override void Save(string filepath)
    {
        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    public override void Load(string filepath)
    {
        var data = System.IO.File.ReadAllBytes(filepath);
        Deserialize(data);
    }
}
