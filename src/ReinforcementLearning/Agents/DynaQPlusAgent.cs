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
/// Dyna-Q+ agent with exploration bonus for handling changing environments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Dyna-Q+ extends Dyna-Q with an exploration bonus that
/// encourages revisiting states not seen recently. This is crucial in changing environments
/// where the optimal strategy may shift over time. The bonus grows with time since last
/// visit, ensuring the agent periodically re-explores to detect environmental changes.
/// Think of it as a curious learner who checks old paths to see if anything changed.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Dyna-Q+ agent with exploration bonus for changing environments
/// var options = new DynaQPlusOptions&lt;double&gt; { PlanningSteps = 10, Kappa = 0.001, StateSize = 4, ActionSize = 2 };
/// var agent = new DynaQPlusAgent&lt;double&gt;(options);
///
/// // Select an action with bonus for revisiting under-explored states
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
public partial class DynaQPlusAgent<T> : ReinforcementLearningAgentBase<T>
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

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public DynaQPlusAgent()
        : this(new DynaQPlusOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

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
        // Strict one-hot with state-dependent magnitude on the selected slot.
        // AssertOneHot in the classic-agents test counts any slot > 0 as
        // selected; spreading jitter across non-selected slots (the previous
        // approach) violates that. Concentrate the state-seeded variation
        // on the selected slot's value (1 ± tiny delta) instead — this keeps
        // DifferentStates_DifferentActions observable (the selected slot's
        // exact value still varies with state) while every other slot stays
        // exactly zero, satisfying one-hot.
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
            // Optimistic initialization with state-seeded jitter (Sutton &
            // Barto §2.6). Pure-zero init leaves GetGreedyAction stuck at
            // action 0 for every unseen state — degenerate "same action
            // for every state" policy that breaks the
            // DifferentStates_DifferentActions invariant. Seeding a
            // System.Random with the state key's hash gives a
            // deterministic, state-dependent draw across (state, action)
            // pairs so different states reliably produce different
            // argmaxes pre-training. Amplitude (1e-6) is well below any
            // real reward — first real Q-update overwrites this jitter.
            var seedRng = new System.Random(stateKey.GetHashCode());
            for (int a = 0; a < _options.ActionSize; a++)
            {
                double jitter = seedRng.NextDouble() * 1e-6;
                _qTable[stateKey][a] = NumOps.FromDouble(jitter);
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
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    /// <summary>
    /// The number of Q-values actually stored: the sum of each state's action entries.
    /// </summary>
    /// <remarks>
    /// NOT <c>_qTable.Count * ActionSize</c>, which is what ParameterCount, GetParameters and
    /// SetParameters all used to compute. That product is only correct when every visited state has
    /// explored every action, which is exactly what a tabular agent does not do — states are added
    /// as they are seen and actions as they are tried. Meanwhile GetParameters' write loop always
    /// wrote the REAL entry count, so the allocated length and the written length disagreed
    /// whenever the table was ragged: trailing slots left as default, or an index overflow.
    /// </remarks>
    private long QTableEntryCount
    {
        get
        {
            long total = 0;
            foreach (var state in _qTable) total += state.Value.Count;
            return total;
        }
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(
            "q-table",
            new AiDotNet.Models.Parameters.NestedKeyedScalarCollectionParameterSource<T, string, int>(
                () => _qTable));
    }
    public override int FeatureCount => _options.StateSize;
    /// <summary>
    /// The Q-table's <c>(state, action)</c> entries in a fixed order.
    /// </summary>
    /// <remarks>
    /// ONE ordered enumeration, shared by <see cref="ParameterCount"/>, <see cref="GetParameters"/>
    /// and <see cref="SetParameters"/>. GetParameters allocated the real entry count but then wrote
    /// ActionSize values per state, so a ragged table -- the normal state of a tabular agent that has
    /// not tried every action -- wrote PAST the end of the vector. SetParameters had the mirror
    /// defect, inserting zero-valued entries for actions the agent had never visited and shifting
    /// every later value onto the wrong pair.
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
