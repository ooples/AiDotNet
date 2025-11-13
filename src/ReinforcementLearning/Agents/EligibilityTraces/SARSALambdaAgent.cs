using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Agents.EligibilityTraces;

public class SARSALambdaAgent<T> : ReinforcementLearningAgentBase<T>
{
    private SARSALambdaOptions<T> _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, T>> _eligibilityTraces;
    private double _epsilon;
    private Vector<T> _lastState;
    private int _lastAction;

    public SARSALambdaAgent(SARSALambdaOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _eligibilityTraces = new Dictionary<string, Dictionary<int, T>>();
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
        if (_lastState != null)
        {
            string stateKey = GetStateKey(_lastState);
            string nextStateKey = GetStateKey(state);
            int nextAction = ArgMax(action);

            EnsureStateExists(_lastState);
            EnsureStateExists(state);

            T currentQ = _qTable[stateKey][_lastAction];
            T nextQ = _qTable[nextStateKey][nextAction];
            T delta = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextQ));
            delta = NumOps.Subtract(delta, currentQ);

            _eligibilityTraces[stateKey][_lastAction] = NumOps.Add(_eligibilityTraces[stateKey][_lastAction], NumOps.One);

            foreach (var s in _qTable.Keys.ToList())
            {
                for (int a = 0; a < _options.ActionSize; a++)
                {
                    T update = NumOps.Multiply(LearningRate, NumOps.Multiply(delta, _eligibilityTraces[s][a]));
                    _qTable[s][a] = NumOps.Add(_qTable[s][a], update);
                    
                    T decayFactor = NumOps.Multiply(DiscountFactor, NumOps.FromDouble(_options.Lambda));
                    _eligibilityTraces[s][a] = NumOps.Multiply(_eligibilityTraces[s][a], decayFactor);
                }
            }
        }

        _lastState = state;
        _lastAction = ArgMax(action);

        if (done)
        {
            ResetEpisode();
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
            _eligibilityTraces[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[stateKey][a] = NumOps.Zero;
                _eligibilityTraces[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private string GetStateKey(Vector<T> state) => string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    
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

    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T> { ["states_visited"] = NumOps.FromDouble(_qTable.Count), ["epsilon"] = NumOps.FromDouble(_epsilon) };
    public override void ResetEpisode() { _lastState = null; foreach (var s in _eligibilityTraces.Keys.ToList()) { for (int a = 0; a < _options.ActionSize; a++) _eligibilityTraces[s][a] = NumOps.Zero; } }
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public override Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public override Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _qTable.Count * _options.ActionSize;
    public override int FeatureCount => _options.StateSize;
    public override byte[] Serialize() => throw new NotImplementedException();
    public override void Deserialize(byte[] data) => throw new NotImplementedException();
    public override Matrix<T> GetParameters() { var p = new List<T>(); foreach (var s in _qTable) foreach (var a in s.Value) p.Add(a.Value); if (p.Count == 0) p.Add(NumOps.Zero); var v = new Vector<T>(p.Count); for (int i = 0; i < p.Count; i++) v[i] = p[i]; return new Matrix<T>(new[] { v }); }
    public override void SetParameters(Matrix<T> parameters) { int idx = 0; foreach (var s in _qTable.ToList()) for (int a = 0; a < _options.ActionSize; a++) if (idx < parameters.Columns) _qTable[s.Key][a] = parameters[0, idx++]; }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone() => new SARSALambdaAgent<T>(_options);
    public override (Matrix<T> Gradients, T Loss) ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.CalculateLoss(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); var grad = lf.CalculateDerivative(new Matrix<T>(new[] { pred }), new Matrix<T>(new[] { target })); return (grad, loss); }
    public override void ApplyGradients(Matrix<T> gradients, T learningRate) { }
    public override void Save(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void Load(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
