using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.EligibilityTraces;

public class QLambdaAgent<T> : ReinforcementLearningAgentBase<T>
{
    private QLambdaOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, T>> _eligibilityTraces;
    private HashSet<string> _activeTraceStates;
    private double _epsilon;
    private Random _random;
    private const double TraceThreshold = 1e-10;

    public QLambdaAgent(QLambdaOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _eligibilityTraces = new Dictionary<string, Dictionary<int, T>>();
        _activeTraceStates = new HashSet<string>();
        _epsilon = options.EpsilonStart;
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
        if (state is null)
        {
            throw new ArgumentNullException(nameof(state), "State vector cannot be null.");
        }

        if (action is null)
        {
            throw new ArgumentNullException(nameof(action), "Action vector cannot be null.");
        }

        if (action.Length == 0)
        {
            throw new ArgumentException("Action vector cannot be empty.", nameof(action));
        }

        if (nextState is null)
        {
            throw new ArgumentNullException(nameof(nextState), "Next state vector cannot be null.");
        }

        string stateKey = GetStateKey(state);
        string nextStateKey = GetStateKey(nextState);
        int actionIndex = ArgMax(action);

        EnsureStateExists(state);
        EnsureStateExists(nextState);

        T currentQ = _qTable[stateKey][actionIndex];
        T maxNextQ = GetMaxQValue(nextStateKey);
        T delta = NumOps.Subtract(NumOps.Add(reward, NumOps.Multiply(DiscountFactor, maxNextQ)), currentQ);

        // Update eligibility trace for current state-action and mark as active
        _eligibilityTraces[stateKey][actionIndex] = NumOps.Add(_eligibilityTraces[stateKey][actionIndex], NumOps.One);
        _activeTraceStates.Add(stateKey);

        // Only iterate over states with active traces (performance optimization)
        var statesToRemove = new List<string>();
        foreach (var s in _activeTraceStates.ToList())
        {
            bool hasActiveTrace = false;
            for (int a = 0; a < _options.ActionSize; a++)
            {
                T traceValue = _eligibilityTraces[s][a];
                double traceDouble = NumOps.ToDouble(traceValue);

                // Update Q-value using the trace
                T update = NumOps.Multiply(LearningRate, NumOps.Multiply(delta, traceValue));
                _qTable[s][a] = NumOps.Add(_qTable[s][a], update);

                // Decay the trace
                T decayFactor = NumOps.Multiply(DiscountFactor, NumOps.FromDouble(_options.Lambda));
                _eligibilityTraces[s][a] = NumOps.Multiply(traceValue, decayFactor);

                // Check if trace is still active after decay
                if (Math.Abs(NumOps.ToDouble(_eligibilityTraces[s][a])) > TraceThreshold)
                {
                    hasActiveTrace = true;
                }
            }

            // Remove state from active set if all traces decayed to near-zero
            if (!hasActiveTrace)
            {
                statesToRemove.Add(s);
            }
        }

        // Clean up inactive trace states
        foreach (var s in statesToRemove)
        {
            _activeTraceStates.Remove(s);
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

    private string GetStateKey(Vector<T> state)
    {
        if (state is null)
        {
            throw new ArgumentNullException(nameof(state), "State vector cannot be null.");
        }

        if (state.Length != _options.StateSize)
        {
            throw new ArgumentException($"State dimension mismatch. Expected {_options.StateSize} but got {state.Length}.", nameof(state));
        }

        return string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    }
    private int GetGreedyAction(string stateKey) { int best = 0; T bestVal = _qTable[stateKey][0]; for (int a = 1; a < _options.ActionSize; a++) if (NumOps.GreaterThan(_qTable[stateKey][a], bestVal)) { bestVal = _qTable[stateKey][a]; best = a; } return best; }
    private T GetMaxQValue(string stateKey) { T max = _qTable[stateKey][0]; for (int a = 1; a < _options.ActionSize; a++) if (NumOps.GreaterThan(_qTable[stateKey][a], max)) max = _qTable[stateKey][a]; return max; }
    private int ArgMax(Vector<T> values) { int maxIndex = 0; T maxValue = values[0]; for (int i = 1; i < values.Length; i++) if (NumOps.GreaterThan(values[i], maxValue)) { maxValue = values[i]; maxIndex = i; } return maxIndex; }

    public override T Train() => NumOps.Zero;
    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T> { ["states_visited"] = NumOps.FromDouble(_qTable.Count), ["epsilon"] = NumOps.FromDouble(_epsilon) };
    public override void ResetEpisode()
    {
        foreach (var s in _eligibilityTraces.Keys.ToList())
            for (int a = 0; a < _options.ActionSize; a++)
                _eligibilityTraces[s][a] = NumOps.Zero;
        _activeTraceStates.Clear();
    }
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
            ActiveTraceStates = _activeTraceStates,
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
        _activeTraceStates = JsonConvert.DeserializeObject<HashSet<string>>(state.ActiveTraceStates.ToString()) ?? new HashSet<string>();
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
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        int expectedSize = _qTable.Count * _options.ActionSize;

        if (expectedSize == 0)
        {
            // Q-table is empty, nothing to set
            return;
        }

        if (parameters.Length != expectedSize)
        {
            throw new ArgumentException($"Parameter vector size mismatch. Expected {expectedSize} parameters (states: {_qTable.Count}, actions: {_options.ActionSize}), but got {parameters.Length}.", nameof(parameters));
        }

        int idx = 0;
        foreach (var s in _qTable.ToList())
        {
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[s.Key][a] = parameters[idx++];
            }
        }
    }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new QLambdaAgent<T>(_options);

        // Deep-copy Q-table
        foreach (var stateEntry in _qTable)
        {
            clone._qTable[stateEntry.Key] = new Dictionary<int, T>();
            foreach (var actionEntry in stateEntry.Value)
            {
                clone._qTable[stateEntry.Key][actionEntry.Key] = actionEntry.Value;
            }
        }

        // Deep-copy eligibility traces
        foreach (var stateEntry in _eligibilityTraces)
        {
            clone._eligibilityTraces[stateEntry.Key] = new Dictionary<int, T>();
            foreach (var actionEntry in stateEntry.Value)
            {
                clone._eligibilityTraces[stateEntry.Key][actionEntry.Key] = actionEntry.Value;
            }
        }

        // Copy active trace states
        foreach (var stateKey in _activeTraceStates)
        {
            clone._activeTraceStates.Add(stateKey);
        }

        // Copy epsilon value
        clone._epsilon = _epsilon;

        return clone;
    }
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.CalculateLoss(pred, target); var grad = lf.CalculateDerivative(pred, target); return grad; }
    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }
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
