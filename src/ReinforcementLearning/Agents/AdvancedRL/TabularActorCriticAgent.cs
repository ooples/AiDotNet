using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.AdvancedRL;

/// <summary>
/// Tabular Actor-Critic agent combining policy and value learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabularActorCriticAgent<T> : ReinforcementLearningAgentBase<T>
{
    private TabularActorCriticOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _policy;  // Actor: π(a|s)
    private Dictionary<string, T> _valueTable;  // Critic: V(s)

    public TabularActorCriticAgent(TabularActorCriticOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _policy = new Dictionary<string, Dictionary<int, T>>();
        _valueTable = new Dictionary<string, T>();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

        // Sample from policy distribution
        var probs = ComputeSoftmax(_policy[stateKey]);
        double r = Random.NextDouble();
        double cumulative = 0.0;
        int selectedAction = 0;

        for (int a = 0; a < _options.ActionSize; a++)
        {
            cumulative += NumOps.ToDouble(probs[a]);
            if (r <= cumulative)
            {
                selectedAction = a;
                break;
            }
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

        // Compute TD error: δ = r + γV(s') - V(s)
        T currentValue = _valueTable[stateKey];
        T nextValue = done ? NumOps.Zero : _valueTable[nextStateKey];
        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextValue));
        T tdError = NumOps.Subtract(target, currentValue);

        // Critic update: V(s) ← V(s) + α_c * δ
        T criticUpdate = NumOps.Multiply(NumOps.FromDouble(_options.CriticLearningRate), tdError);
        _valueTable[stateKey] = NumOps.Add(_valueTable[stateKey], criticUpdate);

        // Actor update: θ(s,a) ← θ(s,a) + α_a * δ
        T actorUpdate = NumOps.Multiply(NumOps.FromDouble(_options.ActorLearningRate), tdError);
        _policy[stateKey][actionIndex] = NumOps.Add(_policy[stateKey][actionIndex], actorUpdate);
    }

    public override T Train() => NumOps.Zero;

    private void EnsureStateExists(Vector<T> state)
    {
        string stateKey = GetStateKey(state);
        if (!_policy.ContainsKey(stateKey))
        {
            _policy[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _policy[stateKey][a] = NumOps.Zero;  // Preferences
            }
            _valueTable[stateKey] = NumOps.Zero;
        }
    }

    private Vector<T> ComputeSoftmax(Dictionary<int, T> preferences)
    {
        T maxPref = preferences[0];
        for (int i = 1; i < preferences.Count; i++)
        {
            if (NumOps.Compare(preferences[i], maxPref) > 0)
            {
                maxPref = preferences[i];
            }
        }

        var expValues = new Vector<T>(preferences.Count);
        T sumExp = NumOps.Zero;
        for (int i = 0; i < preferences.Count; i++)
        {
            T expVal = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(preferences[i], maxPref))));
            expValues[i] = expVal;
            sumExp = NumOps.Add(sumExp, expVal);
        }

        var probs = new Vector<T>(preferences.Count);
        for (int i = 0; i < preferences.Count; i++)
        {
            probs[i] = NumOps.Divide(expValues[i], sumExp);
        }

        return probs;
    }

    private string GetStateKey(Vector<T> state) => string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    private int ArgMax(Vector<T> values) { int maxIndex = 0; T maxValue = values[0]; for (int i = 1; i < values.Length; i++) if (NumOps.Compare(values[i], maxValue) > 0) { maxValue = values[i]; maxIndex = i; } return maxIndex; }

    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T> { ["states_visited"] = NumOps.FromDouble(_valueTable.Count) };
    public override void ResetEpisode() { }
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public override Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public override Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = "TabularActorCritic", InputSize = _options.StateSize, OutputSize = _options.ActionSize, ParameterCount = ParameterCount };
    public override int ParameterCount => _valueTable.Count + (_policy.Count * _options.ActionSize);
    public override int FeatureCount => _options.StateSize;
    public override byte[] Serialize() => throw new NotImplementedException();
    public override void Deserialize(byte[] data) => throw new NotImplementedException();
    public override Matrix<T> GetParameters() { var p = new List<T>(); foreach (var v in _valueTable.Values) p.Add(v); foreach (var s in _policy) foreach (var a in s.Value) p.Add(a.Value); if (p.Count == 0) p.Add(NumOps.Zero); var v = new Vector<T>(p.Count); for (int i = 0; i < p.Count; i++) v[i] = p[i]; return new Matrix<T>(new[] { v }); }
    public override void SetParameters(Matrix<T> parameters) { int idx = 0; foreach (var s in _valueTable.Keys.ToList()) if (idx < parameters.Columns) _valueTable[s] = parameters[0, idx++]; foreach (var s in _policy.ToList()) for (int a = 0; a < _options.ActionSize; a++) if (idx < parameters.Columns) _policy[s.Key][a] = parameters[0, idx++]; }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new TabularActorCriticAgent<T>(_options);
    public override (Matrix<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.ComputeLoss(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); var grad = lf.ComputeDerivative(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); return (grad, loss); }
    public override void ApplyGradients(Matrix<T> gradients, T learningRate) { }
    public override void Save(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void Load(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
