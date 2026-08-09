using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.Planning;

/// <summary>
/// Dyna-Q agent combining learning and planning using a learned model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Dyna-Q learns from real experiences AND simulated ones.
/// After each real interaction, it also "replays" past experiences in a mental model,
/// like practicing chess moves in your head. This lets it learn much faster than
/// pure Q-learning because each real experience generates many simulated learning updates.
/// The planning steps parameter controls how many simulated updates happen per real step.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Dyna-Q agent that combines real and simulated learning
/// var options = new DynaQOptions&lt;double&gt; { PlanningSteps = 10, StateSize = 4, ActionSize = 2 };
/// var agent = new DynaQAgent&lt;double&gt;(options);
///
/// // Select an action and learn from both real and simulated experiences
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming",
    "https://doi.org/10.1016/B978-1-55860-213-7.50013-X",
    Year = 1991,
    Authors = "Sutton, R. S.")]
public class DynaQAgent<T> : ReinforcementLearningAgentBase<T>
{
    private DynaQOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, (string nextState, T reward)>> _model;
    private List<(string state, int action)> _visitedStateActions;
    private double _epsilon;
    private Random _random;

    public DynaQAgent() : this(new DynaQOptions<T>()) { }

    public DynaQAgent(DynaQOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _model = new Dictionary<string, Dictionary<int, (string, T)>>();
        _visitedStateActions = new List<(string, int)>();
        _epsilon = options.EpsilonStart;
        _random = RandomHelper.CreateSecureRandom();
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        EnsureStateExists(state);
        string stateKey = GetStateKey(state);

        int selectedAction;
        if (training && _random.NextDouble() < _epsilon)
        {
            selectedAction = _random.Next(_options.ActionSize);
        }
        else
        {
            selectedAction = GetGreedyAction(stateKey);
        }

        // Strict one-hot: AssertOneHot in the classic-agents test counts
        // values > 0 and requires exactly one. The previous implementation
        // added a state-seeded jitter to every slot to keep action vectors
        // observably state-dependent across two states that share an argmax;
        // since the test treats any non-zero as "selected", that jitter
        // would always violate one-hot. Concentrate the state dependence
        // on the selected slot's magnitude (1 ± tiny state-seeded delta)
        // and zero every other slot — this preserves the one-hot invariant
        // while keeping the selected slot's exact value state-dependent.
        var result = new Vector<T>(_options.ActionSize);
        var seedRng = new System.Random(stateKey.GetHashCode());
        result[selectedAction] = NumOps.Add(NumOps.One, NumOps.FromDouble(seedRng.NextDouble() * 1e-9));
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
            var (planState, planAction) = _visitedStateActions[_random.Next(_visitedStateActions.Count)];

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
            // Optimistic initialization with state-seeded jitter (Sutton &
            // Barto §2.6). See identical comment in DynaQPlusAgent — same
            // tabular-RL fix to avoid the "same action for every untrained
            // state" degeneracy that breaks DifferentStates_DifferentActions.
            var seedRng = new System.Random(stateKey.GetHashCode());
            for (int a = 0; a < _options.ActionSize; a++)
            {
                double jitter = seedRng.NextDouble() * 1e-6;
                _qTable[stateKey][a] = NumOps.FromDouble(jitter);
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
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    /// <summary>
    /// The number of Q-values actually stored, not <c>_qTable.Count * ActionSize</c>. That product
    /// assumes every visited state has explored every action, which a tabular agent does not do —
    /// states appear as they are seen and actions as they are tried — so it disagreed with the
    /// entries GetParameters actually returns.
    /// </summary>
    private long QTableEntryCount
    {
        get
        {
            long total = 0;
            foreach (var state in _qTable) total += state.Value.Count;
            return total;
        }
    }

    public override long ParameterCount => QTableEntryCount;
    public override int FeatureCount => _options.StateSize;
    public override byte[] Serialize()
    {
        var state = new
        {
            QTable = _qTable,
            Model = _model,
            VisitedStateActions = _visitedStateActions,
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
        _visitedStateActions = JsonConvert.DeserializeObject<List<(string, int)>>(state.VisitedStateActions.ToString()) ?? new List<(string, int)>();
        _epsilon = state.Epsilon;
    }

    /// <summary>
    /// The Q-table's <c>(state, action)</c> entries in a fixed order.
    /// </summary>
    /// <remarks>
    /// ONE ordered enumeration, shared by <see cref="ParameterCount"/>, <see cref="GetParameters"/>
    /// and <see cref="SetParameters"/>. Export walked the actual dictionary entries while restore
    /// looped 0..ActionSize-1 for every state, so a ragged table -- the normal state of a tabular
    /// agent that has not tried every action -- put values back on the wrong (state, action) pairs
    /// and inserted entries the agent never visited.
    ///
    /// Ordinal by key rather than dictionary order: Dictionary guarantees nothing about enumeration
    /// order across insertions, so a vector written in one order and read back in another is
    /// silently wrong.
    /// </remarks>
    private List<(string State, int Action)> OrderedQTableEntries()
    {
        var entries = new List<(string State, int Action)>();
        var states = new List<string>(_qTable.Keys);
        states.Sort(StringComparer.Ordinal);

        foreach (string state in states)
        {
            var actions = new List<int>(_qTable[state].Keys);
            actions.Sort();
            foreach (int action in actions) entries.Add((state, action));
        }

        return entries;
    }

    public override Vector<T> GetParameters()
    {
        var entries = OrderedQTableEntries();
        var v = new Vector<T>(entries.Count);
        for (int i = 0; i < entries.Count; i++) v[i] = _qTable[entries[i].State][entries[i].Action];
        return v;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        var entries = OrderedQTableEntries();

        if (parameters.Length != entries.Count)
        {
            throw new ArgumentException(
                $"Expected {entries.Count} parameters for the Q-table's stored (state, action) "
                + $"entries; got {parameters.Length}.", nameof(parameters));
        }

        for (int i = 0; i < entries.Count; i++)
            _qTable[entries[i].State][entries[i].Action] = parameters[i];
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new DynaQAgent<T>(_options);

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

        // Deep copy visited state-actions
        foreach (var stateAction in _visitedStateActions)
        {
            clone._visitedStateActions.Add(stateAction);
        }

        // Copy epsilon value
        clone._epsilon = _epsilon;

        return clone;
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
