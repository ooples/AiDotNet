using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.Planning;

/// <summary>
/// Dyna-Q+ agent with exploration bonus for handling changing environments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DynaQPlusAgent<T> : ReinforcementLearningAgentBase<T>
{
    private DynaQPlusOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, (string nextState, T reward)>> _model;
    private Dictionary<string, Dictionary<int, int>> _timeSteps;  // Track last visit time
    private List<(string state, int action)> _visitedStateActions;
    private double _epsilon;
    private int _totalSteps;
    private Random _random;

    public DynaQPlusAgent(DynaQPlusOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _model = new Dictionary<string, Dictionary<int, (string, T)>>();
        _timeSteps = new Dictionary<string, Dictionary<int, int>>();
        _visitedStateActions = new List<(string, int)>();
        _epsilon = options.EpsilonStart;
        _totalSteps = 0;
        _random = RandomHelper.CreateSecureRandom();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);
        int selectedAction = (training && _random.NextDouble() < _epsilon) ? _random.Next(_options.ActionSize) : GetGreedyAction(stateKey);
        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = GetStateKey(state);
        string nextStateKey = GetStateKey(nextState);
        int actionIndex = ArgMax(action);

        EnsureStateExists(state);
        EnsureStateExists(nextState);

        _totalSteps++;

        // Direct RL update
        T currentQ = _qTable[stateKey][actionIndex];
        T maxNextQ = GetMaxQValue(nextStateKey);
        T target = done ? reward : NumOps.Add(reward, NumOps.Multiply(DiscountFactor, maxNextQ));
        T delta = NumOps.Subtract(target, currentQ);
        _qTable[stateKey][actionIndex] = NumOps.Add(currentQ, NumOps.Multiply(LearningRate, delta));

        // Model learning
        if (!_model.ContainsKey(stateKey))
        {
            _model[stateKey] = new Dictionary<int, (string, T)>();
            _timeSteps[stateKey] = new Dictionary<int, int>();
        }
        _model[stateKey][actionIndex] = (nextStateKey, reward);
        _timeSteps[stateKey][actionIndex] = _totalSteps;

        var stateAction = (stateKey, actionIndex);
        if (!_visitedStateActions.Contains(stateAction))
        {
            _visitedStateActions.Add(stateAction);
        }

        // Planning with exploration bonus
        for (int i = 0; i < _options.PlanningSteps; i++)
        {
            if (_visitedStateActions.Count == 0) break;

            var (planState, planAction) = _visitedStateActions[_random.Next(_visitedStateActions.Count)];

            if (_model.ContainsKey(planState) && _model[planState].ContainsKey(planAction))
            {
                var (planNextState, planReward) = _model[planState][planAction];

                // Add exploration bonus: r + κ√τ where τ is time since last visit
                int timeSinceVisit = _totalSteps - _timeSteps[planState][planAction];
                double explorationBonus = _options.Kappa * Math.Sqrt(timeSinceVisit);
                T bonusReward = NumOps.Add(planReward, NumOps.FromDouble(explorationBonus));

                T planCurrentQ = _qTable[planState][planAction];
                T planMaxNextQ = GetMaxQValue(planNextState);
                T planTarget = NumOps.Add(bonusReward, NumOps.Multiply(DiscountFactor, planMaxNextQ));
                T planDelta = NumOps.Subtract(planTarget, planCurrentQ);
                _qTable[planState][planAction] = NumOps.Add(planCurrentQ, NumOps.Multiply(LearningRate, planDelta));
            }
        }

        if (done)
        {
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        }
    }

    public override T Train() => NumOps.Zero;

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

    private string GetStateKey(Vector<T> state) => string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    private int GetGreedyAction(string stateKey) { int best = 0; T bestVal = _qTable[stateKey][0]; for (int a = 1; a < _options.ActionSize; a++) if (NumOps.GreaterThan(_qTable[stateKey][a], bestVal)) { bestVal = _qTable[stateKey][a]; best = a; } return best; }
    private T GetMaxQValue(string stateKey) { if (!_qTable.ContainsKey(stateKey)) return NumOps.Zero; T max = _qTable[stateKey][0]; for (int a = 1; a < _options.ActionSize; a++) if (NumOps.GreaterThan(_qTable[stateKey][a], max)) max = _qTable[stateKey][a]; return max; }
    private int ArgMax(Vector<T> values) { int maxIndex = 0; T maxValue = values[0]; for (int i = 1; i < values.Length; i++) if (NumOps.GreaterThan(values[i], maxValue)) { maxValue = values[i]; maxIndex = i; } return maxIndex; }

    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T> { ["states_visited"] = NumOps.FromDouble(_qTable.Count), ["model_size"] = NumOps.FromDouble(_model.Count), ["epsilon"] = NumOps.FromDouble(_epsilon), ["total_steps"] = NumOps.FromDouble(_totalSteps) };
    public override void ResetEpisode() { }
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _qTable.Count * _options.ActionSize;
    public override int FeatureCount => _options.StateSize;
    public override byte[] Serialize()
    {
        var state = new
        {
            QTable = _qTable,
            Model = _model,
            TimeSteps = _timeSteps,
            VisitedStateActions = _visitedStateActions,
            Epsilon = _epsilon,
            TotalSteps = _totalSteps,
            Options = _options
        };
        string json = JsonConvert.SerializeObject(state);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }

    public override void Deserialize(byte[] data)
    {
        string json = System.Text.Encoding.UTF8.GetString(data);
        var state = JsonConvert.DeserializeObject<dynamic>(json);
        if (state is null)
        {
            throw new InvalidOperationException("Deserialization returned null");
        }

        _qTable = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.QTable.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
        _model = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, (string, T)>>>(state.Model.ToString()) ?? new Dictionary<string, Dictionary<int, (string, T)>>();
        _timeSteps = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, int>>>(state.TimeSteps.ToString()) ?? new Dictionary<string, Dictionary<int, int>>();
        _visitedStateActions = JsonConvert.DeserializeObject<List<(string, int)>>(state.VisitedStateActions.ToString()) ?? new List<(string, int)>();
        _epsilon = state.Epsilon;
        _totalSteps = state.TotalSteps;
    }
    public override Vector<T> GetParameters()
    {
        int paramCount = _qTable.Count > 0 ? _qTable.Count * _options.ActionSize : 1;
        var v = new Vector<T>(paramCount);
        int idx = 0;

        // Sort state keys for deterministic ordering
        var sortedStates = _qTable.Keys.OrderBy(k => k).ToList();
        foreach (var stateKey in sortedStates)
        {
            var actionDict = _qTable[stateKey];
            for (int a = 0; a < _options.ActionSize; a++)
            {
                if (actionDict.ContainsKey(a))
                {
                    v[idx++] = actionDict[a];
                }
                else
                {
                    v[idx++] = NumOps.Zero;
                }
            }
        }

        if (idx == 0)
            v[0] = NumOps.Zero;

        return v;
    }
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null || parameters.Length == 0)
        {
            return;
        }

        int idx = 0;
        var sortedStates = _qTable.Keys.OrderBy(k => k).ToList();

        foreach (var stateKey in sortedStates)
        {
            for (int a = 0; a < _options.ActionSize; a++)
            {
                if (idx < parameters.Length)
                {
                    if (!_qTable[stateKey].ContainsKey(a))
                    {
                        _qTable[stateKey][a] = NumOps.Zero;
                    }
                    _qTable[stateKey][a] = parameters[idx++];
                }
            }
        }
    }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new DynaQPlusAgent<T>(_options);

        // Deep copy Q-table
        foreach (var stateEntry in _qTable)
        {
            clone._qTable[stateEntry.Key] = new Dictionary<int, T>();
            foreach (var actionEntry in stateEntry.Value)
            {
                clone._qTable[stateEntry.Key][actionEntry.Key] = actionEntry.Value;
            }
        }

        // Deep copy model
        foreach (var stateEntry in _model)
        {
            clone._model[stateEntry.Key] = new Dictionary<int, (string, T)>();
            foreach (var actionEntry in stateEntry.Value)
            {
                clone._model[stateEntry.Key][actionEntry.Key] = actionEntry.Value;
            }
        }

        // Deep copy time steps
        foreach (var stateEntry in _timeSteps)
        {
            clone._timeSteps[stateEntry.Key] = new Dictionary<int, int>();
            foreach (var actionEntry in stateEntry.Value)
            {
                clone._timeSteps[stateEntry.Key][actionEntry.Key] = actionEntry.Value;
            }
        }

        // Deep copy visited state-actions
        foreach (var stateAction in _visitedStateActions)
        {
            clone._visitedStateActions.Add(stateAction);
        }

        // Copy scalar values
        clone._epsilon = _epsilon;
        clone._totalSteps = _totalSteps;

        return clone;
    }
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.CalculateLoss(pred, target); var grad = lf.CalculateDerivative(pred, target); return grad; }
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Dyna-Q+ uses model-based planning with Q-learning updates, not gradient-based optimization
        // This method is not applicable for tabular Q-learning methods
        throw new NotSupportedException("Dyna-Q+ uses model-based planning with Q-learning updates, not gradient-based optimization. Use StoreExperience for updates.");
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
