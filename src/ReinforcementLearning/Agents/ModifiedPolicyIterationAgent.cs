using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.DynamicProgramming;

/// <summary>
/// Helper class for serializing model transition data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TransitionData<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string NextState { get; set; } = string.Empty;
    public T Reward { get; set; }
    public T Probability { get; set; }

    public TransitionData()
    {
        Reward = NumOps.Zero;
        Probability = NumOps.Zero;
    }
}

/// <summary>
/// Modified Policy Iteration agent - hybrid of Policy Iteration and Value Iteration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Modified PI performs limited policy evaluation sweeps before improvement,
/// trading off between the efficiency of VI and the stability of PI.</para>
/// <para><b>For Beginners:</b> Modified Policy Iteration is a middle ground between two classic
/// algorithms: Value Iteration (fast but less stable) and Policy Iteration (stable but slow).
/// Instead of fully evaluating a policy before improving it, it does a limited number of
/// evaluation sweeps. Think of it like proofreading a draft: you do a few passes (not infinite)
/// before revising. The number of evaluation sweeps controls the speed-stability trade-off.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Modified Policy Iteration agent balancing speed and stability
/// var options = new ModifiedPolicyIterationOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, MaxEvaluationSweeps = 5 };
/// var agent = new ModifiedPolicyIterationAgent&lt;double&gt;(options);
///
/// // Select an action using the partially-evaluated policy
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Reinforcement Learning: An Introduction",
    "https://incompleteideas.net/book/the-book-2nd.html",
    Year = 2018,
    Authors = "Sutton, R. S. & Barto, A. G.")]
public partial class ModifiedPolicyIterationAgent<T> : ReinforcementLearningAgentBase<T>
{
    private ModifiedPolicyIterationOptions<T> _options;

    private Dictionary<string, T> _valueTable;
    private Dictionary<string, int> _policy;
    private Dictionary<string, Dictionary<int, List<(string nextState, T reward, T probability)>>> _model;

    /// <summary>
    /// Initializes a new instance with default options (StateSize=4, ActionSize=2).
    /// </summary>
    public ModifiedPolicyIterationAgent()
        : this(new ModifiedPolicyIterationOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public ModifiedPolicyIterationAgent(ModifiedPolicyIterationOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        Options = _options;
        _valueTable = new Dictionary<string, T>();
        _policy = new Dictionary<string, int>();
        _model = new Dictionary<string, Dictionary<int, List<(string, T, T)>>>();
    }

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = GetStateKey(state);

        if (!_policy.ContainsKey(stateKey))
        {
            // Deterministic initial policy: the greedy action over the zero-initialized value table is
            // action 0 (all action-values equal). Modified Policy Iteration (Puterman & Shin 1978) starts
            // from an arbitrary DETERMINISTIC policy; a random initial action makes SelectAction/Predict
            // non-reproducible and clone-variant (the clone's independent RNG picks a different action for
            // the same unseen state). Policy Improvement refines this from the collected model.
            _policy[stateKey] = 0;
            _valueTable[stateKey] = NumOps.Zero;
        }

        int selectedAction = _policy[stateKey];

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = GetStateKey(state);
        string nextStateKey = GetStateKey(nextState);
        int actionIndex = ArgMax(action);

        if (!_model.ContainsKey(stateKey))
        {
            _model[stateKey] = new Dictionary<int, List<(string, T, T)>>();
        }

        if (!_model[stateKey].ContainsKey(actionIndex))
        {
            _model[stateKey][actionIndex] = new List<(string, T, T)>();
        }

        // Store transition with count of 1 initially
        // Probabilities will be normalized when computing expected values
        _model[stateKey][actionIndex].Add((nextStateKey, reward, NumOps.One));
    }

    /// <inheritdoc />
    public override T Train()
    {
        if (_model.Count == 0)
        {
            return NumOps.Zero;
        }

        bool policyStable = false;
        int iterations = 0;

        while (!policyStable && iterations < 100)
        {
            // Modified Policy Evaluation (limited sweeps)
            ModifiedPolicyEvaluation();

            // Policy Improvement
            policyStable = PolicyImprovement();

            iterations++;
        }

        return NumOps.FromDouble(iterations);
    }

    private void ModifiedPolicyEvaluation()
    {
        // At most MaxEvaluationSweeps sweeps (Puterman & Shin 1978's partial evaluation), stopping early once the
        // largest value change in a sweep falls below Theta, the evaluation tolerance.
        for (int sweep = 0; sweep < _options.MaxEvaluationSweeps; sweep++)
        {
            var oldValues = new List<T>();
            var newValues = new List<T>();
            foreach (var stateKey in _valueTable.Keys.ToList())
            {
                if (!_policy.ContainsKey(stateKey))
                {
                    continue;
                }

                int action = _policy[stateKey];
                T newValue = ComputeActionValue(stateKey, action);
                oldValues.Add(_valueTable[stateKey]);
                newValues.Add(newValue);
                _valueTable[stateKey] = newValue;
            }

            if (SweepConverged(oldValues, newValues))
            {
                break;
            }
        }
    }

    /// <summary>The value table's estimate for a state, or zero for a state the agent has not seen.</summary>
    internal T StateValue(Vector<T> state) => _valueTable.TryGetValue(GetStateKey(state), out var value) ? value : NumOps.Zero;

    /// <summary>
    /// True when the sweep's largest change max(0, max |new - old|) is below Theta. A NaN change
    /// poisons the running Math.Max, so it never converges, exactly like the scalar tracker.
    /// </summary>
    private bool SweepConverged(List<T> oldValues, List<T> newValues)
    {
        T theta = NumOps.FromDouble(_options.Theta);
        if (oldValues.Count == 0)
        {
            return NumOps.LessThan(NumOps.Zero, theta);
        }

        var engine = AiDotNetEngine.Current;
        var change = engine.Abs(engine.Subtract(Vector<T>.FromEnumerable(newValues), Vector<T>.FromEnumerable(oldValues)));
        var changeTensor = change.AsTensor();
        var nanFlags = engine.TensorWhere(engine.TensorIsNan(changeTensor),
            engine.TensorFullLike(changeTensor, NumOps.One), engine.TensorFullLike(changeTensor, NumOps.Zero));
        if (RLTensorOps.AnyGreaterThan(nanFlags.ToVector(), NumOps.Zero))
        {
            return false;
        }

        return NumOps.LessThan(RLTensorOps.MaxAbove(change, NumOps.Zero), theta);
    }

    private bool PolicyImprovement()
    {
        bool policyStable = true;

        foreach (var stateKey in _policy.Keys.ToList())
        {
            int oldAction = _policy[stateKey];

            int bestAction = RLTensorOps.ArgMaxAbove(ActionValues(stateKey), NumOps.MinValue, out _);

            _policy[stateKey] = bestAction;

            if (oldAction != bestAction)
            {
                policyStable = false;
            }
        }

        return policyStable;
    }

    /// <summary>The one-step look-ahead value of every action in a state, as a vector.</summary>
    private Vector<T> ActionValues(string stateKey)
    {
        return Vector<T>.FromEnumerable(Enumerable.Range(0, _options.ActionSize).Select(a => ComputeActionValue(stateKey, a)));
    }

    private T ComputeActionValue(string stateKey, int action)
    {
        if (!_model.ContainsKey(stateKey) || !_model[stateKey].ContainsKey(action))
        {
            return NumOps.Zero;
        }

        // Normalize probabilities by total count to prevent blow-up
        var transitions = _model[stateKey][action];
        if (transitions.Count == 0)
        {
            return NumOps.Zero;
        }

        T totalCount = NumOps.FromDouble(transitions.Count);

        // Expected value sum_s' (p(s') / N) * (r + gamma * V(s')), with V = 0 for unseen states;
        // each transition's probability is normalized by the transition count N.
        var engine = AiDotNetEngine.Current;
        var zero = NumOps.Zero;
        var rewards = Vector<T>.FromEnumerable(transitions.Select(transition => transition.Item2));
        var probabilities = Vector<T>.FromEnumerable(transitions.Select(transition => transition.Item3));
        var nextValues = Vector<T>.FromEnumerable(transitions.Select(
            transition => _valueTable.TryGetValue(transition.Item1, out var value) ? value : zero));
        var normalized = engine.Divide(probabilities, totalCount);
        var transitionValues = engine.Add(rewards, engine.Multiply(nextValues, DiscountFactor));
        return engine.DotProduct(normalized, transitionValues);
    }

    private string GetStateKey(Vector<T> state)
    {
        return string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    }

    private int ArgMax(Vector<T> values)
    {
        return RLTensorOps.ArgMax(values);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["states_visited"] = NumOps.FromDouble(_valueTable.Count),
            ["model_transitions"] = NumOps.FromDouble(_model.Count)
        };
    }

    /// <inheritdoc />
    public override void ResetEpisode()
    {
        // No episode-specific state
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Vector<T> input)
    {
        return SelectAction(input, training: false);
    }

    public Task<Vector<T>> PredictAsync(Vector<T> input)
    {
        return Task.FromResult(Predict(input));
    }

    public Task TrainAsync()
    {
        Train();
        return Task.CompletedTask;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
        };
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(
            "value-table",
            new AiDotNet.Models.Parameters.KeyedScalarCollectionParameterSource<T, string>(
                () => _valueTable));
    }

    /// <inheritdoc />
    public override int FeatureCount => _options.StateSize;

    /// <inheritdoc />
    public override void SaveModel(string filepath)
    {
        if (string.IsNullOrWhiteSpace(filepath))
        {
            throw new ArgumentException("File path cannot be null or whitespace", nameof(filepath));
        }

        var data = Serialize();
        System.IO.File.WriteAllBytes(filepath, data);
    }

    /// <inheritdoc />
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
