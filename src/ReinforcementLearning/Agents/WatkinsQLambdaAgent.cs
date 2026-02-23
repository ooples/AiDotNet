using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.EligibilityTraces;

public class WatkinsQLambdaAgent<T> : ReinforcementLearningAgentBase<T>
{
    private WatkinsQLambdaOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, T>> _eligibilityTraces;
    private double _epsilon;

    public WatkinsQLambdaAgent(WatkinsQLambdaOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _eligibilityTraces = new Dictionary<string, Dictionary<int, T>>();
        _epsilon = options.EpsilonStart;
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
        // Ensure states exist BEFORE accessing Q-table
        EnsureStateExists(state);
        EnsureStateExists(nextState);

        string stateKey = GetStateKey(state);
        string nextStateKey = GetStateKey(nextState);
        int actionIndex = ArgMax(action);
        int greedyCurrentAction = GetGreedyAction(stateKey);
        int greedyNextAction = GetGreedyAction(nextStateKey);

        T currentQ = _qTable[stateKey][actionIndex];
        T maxNextQ = _qTable[nextStateKey][greedyNextAction];
        T delta = NumOps.Subtract(NumOps.Add(reward, NumOps.Multiply(DiscountFactor, maxNextQ)), currentQ);

        _eligibilityTraces[stateKey][actionIndex] = NumOps.Add(_eligibilityTraces[stateKey][actionIndex], NumOps.One);

        // Watkins's Q(λ): Check if current action was greedy
        bool actionWasGreedy = (actionIndex == greedyCurrentAction);

        foreach (var s in _qTable.Keys.ToList())
        {
            for (int a = 0; a < _options.ActionSize; a++)
            {
                T update = NumOps.Multiply(LearningRate, NumOps.Multiply(delta, _eligibilityTraces[s][a]));
                _qTable[s][a] = NumOps.Add(_qTable[s][a], update);

                // Watkins's Q(λ): reset ALL traces if action was non-greedy (exploratory)
                if (!actionWasGreedy)
                {
                    _eligibilityTraces[s][a] = NumOps.Zero;
                }
                else
                {
                    T decayFactor = NumOps.Multiply(DiscountFactor, NumOps.FromDouble(_options.Lambda));
                    _eligibilityTraces[s][a] = NumOps.Multiply(_eligibilityTraces[s][a], decayFactor);
                }
            }
        }

        if (done)
        {
            ResetEpisode();
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        }
    }

    private void EnsureStateExists(Vector<T> state)
    {
        string stateKey = GetStateKey(state);
        if (!_qTable.ContainsKey(stateKey))
        {
            _qTable[stateKey] = new Dictionary<int, T>();
            _eligibilityTraces[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[stateKey][a] = NumOps.Zero;
                _eligibilityTraces[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private string GetStateKey(Vector<T> state) => string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    private int GetGreedyAction(string stateKey) { int best = 0; T bestVal = _qTable[stateKey][0]; for (int a = 1; a < _options.ActionSize; a++) if (NumOps.GreaterThan(_qTable[stateKey][a], bestVal)) { bestVal = _qTable[stateKey][a]; best = a; } return best; }
    private int ArgMax(Vector<T> values) { int maxIndex = 0; T maxValue = values[0]; for (int i = 1; i < values.Length; i++) if (NumOps.GreaterThan(values[i], maxValue)) { maxValue = values[i]; maxIndex = i; } return maxIndex; }

    public override T Train() => NumOps.Zero;
    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T> { ["states_visited"] = NumOps.FromDouble(_qTable.Count), ["epsilon"] = NumOps.FromDouble(_epsilon) };
    public override void ResetEpisode() { foreach (var s in _eligibilityTraces.Keys.ToList()) for (int a = 0; a < _options.ActionSize; a++) _eligibilityTraces[s][a] = NumOps.Zero; }
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
            EligibilityTraces = _eligibilityTraces,
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
        _eligibilityTraces = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, T>>>(state.EligibilityTraces.ToString()) ?? new Dictionary<string, Dictionary<int, T>>();
        _epsilon = state.Epsilon;
    }
    public override Vector<T> GetParameters()
    {
        int paramCount = _qTable.Count > 0 ? _qTable.Count * _options.ActionSize : 1;
        var v = new Vector<T>(paramCount);
        int idx = 0;

        foreach (var s in _qTable)
            foreach (var a in s.Value)
                v[idx++] = a.Value;

        if (idx == 0)
            v[0] = NumOps.Zero;

        return v;
    }
    public override void SetParameters(Vector<T> parameters) { int idx = 0; foreach (var s in _qTable.ToList()) for (int a = 0; a < _options.ActionSize; a++) if (idx < parameters.Length) _qTable[s.Key][a] = parameters[idx++]; }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new WatkinsQLambdaAgent<T>(_options);

        // Deep copy Q-table to preserve learned state
        foreach (var kvp in _qTable)
        {
            clone._qTable[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        // Deep copy eligibility traces
        foreach (var kvp in _eligibilityTraces)
        {
            clone._eligibilityTraces[kvp.Key] = new Dictionary<int, T>(kvp.Value);
        }

        clone._epsilon = _epsilon;
        return clone;
    }
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.CalculateLoss(pred, target); var grad = lf.CalculateDerivative(pred, target); return grad; }
    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
