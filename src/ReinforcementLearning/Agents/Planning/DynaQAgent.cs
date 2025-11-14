using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.Planning;

/// <summary>
/// Dyna-Q agent combining learning and planning using a learned model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DynaQAgent<T> : ReinforcementLearningAgentBase<T>
{
    private DynaQOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, (string nextState, T reward)>> _model;
    private List<(string state, int action)> _visitedStateActions;
    private double _epsilon;

    public DynaQAgent(DynaQOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _model = new Dictionary<string, Dictionary<int, (string, T)>>();
        _visitedStateActions = new List<(string, int)>();
        _epsilon = options.EpsilonStart;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

        int selectedAction;
        if (training && Random.NextDouble() < _epsilon)
        {
            selectedAction = Random.Next(_options.ActionSize);
        }
        else
        {
            selectedAction = GetGreedyAction(stateKey);
        }

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

        // Direct RL update (Q-learning)
        T currentQ = _qTable[stateKey][actionIndex];
        T maxNextQ = GetMaxQValue(nextStateKey);
        T target = done ? reward : NumOps.Add(reward, NumOps.Multiply(DiscountFactor, maxNextQ));
        T delta = NumOps.Subtract(target, currentQ);
        _qTable[stateKey][actionIndex] = NumOps.Add(currentQ, NumOps.Multiply(LearningRate, delta));

        // Model learning
        if (!_model.ContainsKey(stateKey))
        {
            _model[stateKey] = new Dictionary<int, (string, T)>();
        }
        _model[stateKey][actionIndex] = (nextStateKey, reward);

        // Track visited state-actions
        var stateAction = (stateKey, actionIndex);
        if (!_visitedStateActions.Contains(stateAction))
        {
            _visitedStateActions.Add(stateAction);
        }

        // Planning: perform n simulated experiences
        for (int i = 0; i < _options.PlanningSteps; i++)
        {
            if (_visitedStateActions.Count == 0) break;

            // Random previously observed state-action
            var (planState, planAction) = _visitedStateActions[Random.Next(_visitedStateActions.Count)];

            if (_model.ContainsKey(planState) && _model[planState].ContainsKey(planAction))
            {
                var (planNextState, planReward) = _model[planState][planAction];

                // Simulated Q-learning update
                T planCurrentQ = _qTable[planState][planAction];
                T planMaxNextQ = GetMaxQValue(planNextState);
                T planTarget = NumOps.Add(planReward, NumOps.Multiply(DiscountFactor, planMaxNextQ));
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

    private int GetGreedyAction(string stateKey)
    {
        int best = 0;
        T bestVal = _qTable[stateKey][0];
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(_qTable[stateKey][a], bestVal))
            {
                bestVal = _qTable[stateKey][a];
                best = a;
            }
        }
        return best;
    }

    private T GetMaxQValue(string stateKey)
    {
        if (!_qTable.ContainsKey(stateKey)) return NumOps.Zero;
        T max = _qTable[stateKey][0];
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(_qTable[stateKey][a], max))
            {
                max = _qTable[stateKey][a];
            }
        }
        return max;
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

    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T>
    {
        ["states_visited"] = NumOps.FromDouble(_qTable.Count),
        ["model_size"] = NumOps.FromDouble(_model.Count),
        ["epsilon"] = NumOps.FromDouble(_epsilon)
    };

    public override void ResetEpisode() { }
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _qTable.Count * _options.ActionSize;
    public override int FeatureCount => _options.StateSize;
    public override byte[] Serialize() => throw new NotImplementedException();
    public override void Deserialize(byte[] data) => throw new NotImplementedException();

    public override Vector<T> GetParameters()
    {
        var p = new List<T>();
        foreach (var s in _qTable)
            foreach (var a in s.Value)
                p.Add(a.Value);
        if (p.Count == 0) p.Add(NumOps.Zero);
        var v = new Vector<T>(p.Count);
        for (int i = 0; i < p.Count; i++) v[i] = p[i];
        return new Matrix<T>(new[] { v });
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        foreach (var s in _qTable.ToList())
            for (int a = 0; a < _options.ActionSize; a++)
                if (idx < parameters.Columns)
                    _qTable[s.Key][a] = parameters[0, idx++];
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new DynaQAgent<T>(_options);

    public override (Vector<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var pred = Predict(input);
        var lf = lossFunction ?? LossFunction;
        var loss = lf.CalculateLoss(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target }));
        var grad = lf.CalculateDerivative(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target }));
        return (grad, loss);
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
