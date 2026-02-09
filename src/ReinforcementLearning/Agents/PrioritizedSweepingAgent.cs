using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.Planning;

/// <summary>
/// Prioritized Sweeping agent that focuses planning on high-priority state-actions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PrioritizedSweepingAgent<T> : ReinforcementLearningAgentBase<T>
{
    private PrioritizedSweepingOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, (string nextState, T reward)>> _model;
    private Dictionary<string, List<(string predecessor, int action)>> _predecessors;
    private SortedSet<(double priority, string state, int action)> _priorityQueue;
    private double _epsilon;

    public PrioritizedSweepingAgent(PrioritizedSweepingOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _model = new Dictionary<string, Dictionary<int, (string, T)>>();
        _predecessors = new Dictionary<string, List<(string, int)>>();
        _priorityQueue = new SortedSet<(double, string, int)>(Comparer<(double, string, int)>.Create((a, b) =>
        {
            int cmp = b.Item1.CompareTo(a.Item1);  // Descending priority
            if (cmp != 0) return cmp;
            cmp = string.CompareOrdinal(a.Item2, b.Item2);
            if (cmp != 0) return cmp;
            return a.Item3.CompareTo(b.Item3);
        }));
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
        string stateKey = GetStateKey(state);
        string nextStateKey = GetStateKey(nextState);
        int actionIndex = ArgMax(action);

        EnsureStateExists(state);
        EnsureStateExists(nextState);

        // Model learning
        if (!_model.ContainsKey(stateKey))
        {
            _model[stateKey] = new Dictionary<int, (string, T)>();
        }
        _model[stateKey][actionIndex] = (nextStateKey, reward);

        // Track predecessors
        if (!_predecessors.ContainsKey(nextStateKey))
        {
            _predecessors[nextStateKey] = new List<(string, int)>();
        }
        var pred = (stateKey, actionIndex);
        if (!_predecessors[nextStateKey].Contains(pred))
        {
            _predecessors[nextStateKey].Add(pred);
        }

        // Compute priority (TD error)
        T currentQ = _qTable[stateKey][actionIndex];
        T maxNextQ = GetMaxQValue(nextStateKey);
        T target = done ? reward : NumOps.Add(reward, NumOps.Multiply(DiscountFactor, maxNextQ));
        T delta = NumOps.Subtract(target, currentQ);
        double priority = Math.Abs(NumOps.ToDouble(delta));

        // Add to priority queue if above threshold
        if (priority > _options.PriorityThreshold)
        {
            _priorityQueue.Add((priority, stateKey, actionIndex));
        }

        // Planning: process high-priority updates
        int plannedUpdates = 0;
        while (_priorityQueue.Count > 0 && plannedUpdates < _options.PlanningSteps)
        {
            // Store Min value once to avoid double access
            var highestPriority = _priorityQueue.Min;
            _priorityQueue.Remove(highestPriority);
            var (p, s, a) = highestPriority;

            if (_model.ContainsKey(s) && _model[s].ContainsKey(a))
            {
                var (nextS, r) = _model[s][a];

                // Update Q-value
                T q = _qTable[s][a];
                T maxQ = GetMaxQValue(nextS);
                T t = NumOps.Add(r, NumOps.Multiply(DiscountFactor, maxQ));
                T d = NumOps.Subtract(t, q);
                _qTable[s][a] = NumOps.Add(q, NumOps.Multiply(LearningRate, d));

                // Update predecessors
                if (_predecessors.ContainsKey(s))
                {
                    foreach (var (predState, predAction) in _predecessors[s])
                    {
                        if (_model.ContainsKey(predState) && _model[predState].ContainsKey(predAction))
                        {
                            var (predNextState, predReward) = _model[predState][predAction];
                            T predQ = _qTable[predState][predAction];
                            T predMaxQ = GetMaxQValue(predNextState);
                            T predTarget = NumOps.Add(predReward, NumOps.Multiply(DiscountFactor, predMaxQ));
                            T predDelta = NumOps.Subtract(predTarget, predQ);
                            double predPriority = Math.Abs(NumOps.ToDouble(predDelta));

                            if (predPriority > _options.PriorityThreshold)
                            {
                                _priorityQueue.Add((predPriority, predState, predAction));
                            }
                        }
                    }
                }
            }

            plannedUpdates++;
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

    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T> { ["states_visited"] = NumOps.FromDouble(_qTable.Count), ["model_size"] = NumOps.FromDouble(_model.Count), ["queue_size"] = NumOps.FromDouble(_priorityQueue.Count), ["epsilon"] = NumOps.FromDouble(_epsilon) };
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
            Predecessors = _predecessors,
            PriorityQueue = _priorityQueue.ToList(),
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
        _model = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<int, (string, T)>>>(state.Model.ToString()) ?? new Dictionary<string, Dictionary<int, (string, T)>>();
        _predecessors = JsonConvert.DeserializeObject<Dictionary<string, List<(string, int)>>>(state.Predecessors.ToString()) ?? new Dictionary<string, List<(string, int)>>();

        var priorityList = JsonConvert.DeserializeObject<List<(double, string, int)>>(state.PriorityQueue.ToString()) ?? new List<(double, string, int)>();
        _priorityQueue.Clear();
        foreach (var item in priorityList)
        {
            _priorityQueue.Add(item);
        }

        _epsilon = state.Epsilon;
    }
    public override Vector<T> GetParameters()
    {
        int paramCount = _qTable.Count > 0 ? _qTable.Count * _options.ActionSize : 1;
        var v = new Vector<T>(paramCount);
        int idx = 0;

        // Sort states by key for deterministic ordering
        var sortedStates = _qTable.OrderBy(kvp => kvp.Key);
        foreach (var stateEntry in sortedStates)
        {
            // Actions are already in deterministic order (0 to ActionSize-1)
            for (int a = 0; a < _options.ActionSize; a++)
            {
                v[idx++] = stateEntry.Value[a];
            }
        }

        if (idx == 0)
            v[0] = NumOps.Zero;

        return v;
    }
    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        // Sort states by key for deterministic ordering
        var sortedStates = _qTable.Keys.OrderBy(k => k).ToList();
        foreach (var stateKey in sortedStates)
        {
            for (int a = 0; a < _options.ActionSize; a++)
            {
                if (idx < parameters.Length)
                {
                    _qTable[stateKey][a] = parameters[idx++];
                }
            }
        }
    }
    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var cloned = new PrioritizedSweepingAgent<T>(_options);

        // Deep copy Q-table
        foreach (var stateEntry in _qTable)
        {
            cloned._qTable[stateEntry.Key] = new Dictionary<int, T>();
            foreach (var actionEntry in stateEntry.Value)
            {
                cloned._qTable[stateEntry.Key][actionEntry.Key] = actionEntry.Value;
            }
        }

        // Deep copy model
        foreach (var stateEntry in _model)
        {
            cloned._model[stateEntry.Key] = new Dictionary<int, (string, T)>();
            foreach (var actionEntry in stateEntry.Value)
            {
                cloned._model[stateEntry.Key][actionEntry.Key] = actionEntry.Value;
            }
        }

        // Deep copy predecessors
        foreach (var stateEntry in _predecessors)
        {
            cloned._predecessors[stateEntry.Key] = new List<(string, int)>(stateEntry.Value);
        }

        // Deep copy priority queue
        foreach (var item in _priorityQueue)
        {
            cloned._priorityQueue.Add(item);
        }

        // Copy epsilon value
        cloned._epsilon = _epsilon;

        return cloned;
    }
    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null) { var pred = Predict(input); var lf = lossFunction ?? LossFunction; var loss = lf.CalculateLoss(pred, target); var grad = lf.CalculateDerivative(pred, target); return grad; }
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        throw new NotSupportedException("Gradient-based updates are not supported for tabular reinforcement learning agents. Q-values are updated directly through temporal difference learning in StoreExperience().");
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
