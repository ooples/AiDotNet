using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.Planning;

/// <summary>
/// Dyna-Q+ agent with exploration bonus for handling changing environments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DynaQPlusAgent<T> : ReinforcementLearningAgentBase<T>
{
    private DynaQPlusOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, (string nextState, T reward)>> _model;
    private Dictionary<string, Dictionary<int, int>> _timeSteps;  // Track last visit time
    private List<(string state, int action)> _visitedStateActions;
    private double _epsilon;
    private int _totalSteps;

    public DynaQPlusAgent(DynaQPlusOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _model = new Dictionary<string, Dictionary<int, (string, T)>>();
        _timeSteps = new Dictionary<string, Dictionary<int, int>>();
        _visitedStateActions = new List<(string, int)>();
        _epsilon = options.EpsilonStart;
        _totalSteps = 0;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);
        int selectedAction = (training && Random.NextDouble() < _epsilon) ? Random.Next(_options.ActionSize) : GetGreedyAction(stateKey);
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

            var (planState, planAction) = _visitedStateActions[Random.Next(_visitedStateActions.Count)];

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
    public override byte[] Serialize() => throw new NotImplementedException();
    public override void Deserialize(byte[] data) => throw new NotImplementedException();
    public override Matrix<T> GetParameters()
    {
        int paramCount = _qTable.Count > 0 ? _qTable.Count * _options.ActionSize : 1;
        var v = new Vector<T>(paramCount);
        int idx = 0;

        foreach (var s in _qTable)
            foreach (var a in s.Value)
                v[idx++] = a.Value;

        if (idx == 0)
            v[0] = NumOps.Zero;

        return new Matrix<T>(new[] { v });
    }
    public override void SetParameters(Matrix<T> parameters) { int idx = 0; foreach (var s in _qTable.ToList()) for (int a = 0; a < _options.ActionSize; a++) if (idx < parameters.Columns) _qTable[s.Key][a] = parameters[0, idx++]; }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new DynaQPlusAgent<T>(_options);
    public override (Matrix<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.CalculateLoss(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); var grad = lf.CalculateDerivative(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); return (grad, loss); }
    public override void ApplyGradients(Matrix<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
