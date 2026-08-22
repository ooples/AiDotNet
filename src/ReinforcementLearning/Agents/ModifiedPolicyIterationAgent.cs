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
/// var options = new ModifiedPolicyIterationOptions&lt;double&gt; { StateSize = 4, ActionSize = 2, EvaluationSweeps = 5 };
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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
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
        _valueTable = new Dictionary<string, T>();
        _policy = new Dictionary<string, int>();
        _model = new Dictionary<string, Dictionary<int, List<(string, T, T)>>>();
    }

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
        // Only do k sweeps instead of iterating to convergence
        for (int sweep = 0; sweep < _options.MaxEvaluationSweeps; sweep++)
        {
            foreach (var stateKey in _valueTable.Keys.ToList())
            {
                if (!_policy.ContainsKey(stateKey))
                {
                    continue;
                }

                int action = _policy[stateKey];
                T newValue = ComputeActionValue(stateKey, action);
                _valueTable[stateKey] = newValue;
            }
        }
    }

    private bool PolicyImprovement()
    {
        bool policyStable = true;

        foreach (var stateKey in _policy.Keys.ToList())
        {
            int oldAction = _policy[stateKey];

            int bestAction = 0;
            T bestValue = NumOps.MinValue;

            for (int a = 0; a < _options.ActionSize; a++)
            {
                T actionValue = ComputeActionValue(stateKey, a);

                if (NumOps.GreaterThan(actionValue, bestValue))
                {
                    bestValue = actionValue;
                    bestAction = a;
                }
            }

            _policy[stateKey] = bestAction;

            if (oldAction != bestAction)
            {
                policyStable = false;
            }
        }

        return policyStable;
    }

    private T ComputeActionValue(string stateKey, int action)
    {
        if (!_model.ContainsKey(stateKey) || !_model[stateKey].ContainsKey(action))
        {
            return NumOps.Zero;
        }

        T expectedValue = NumOps.Zero;

        // Normalize probabilities by total count to prevent blow-up
        var transitions = _model[stateKey][action];
        T totalCount = NumOps.FromDouble(transitions.Count);

        foreach (var (nextStateKey, reward, probability) in transitions)
        {
            T nextValue = NumOps.Zero;
            if (_valueTable.ContainsKey(nextStateKey))
            {
                nextValue = _valueTable[nextStateKey];
            }

            // Normalize probability: each transition gets weight 1/N
            T normalizedProb = NumOps.Divide(probability, totalCount);
            T transitionValue = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextValue));
            expectedValue = NumOps.Add(expectedValue, NumOps.Multiply(normalizedProb, transitionValue));
        }

        return expectedValue;
    }

    private string GetStateKey(Vector<T> state)
    {
        return string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
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

    public override Dictionary<string, T> GetMetrics()
    {
        return new Dictionary<string, T>
        {
            ["states_visited"] = NumOps.FromDouble(_valueTable.Count),
            ["model_transitions"] = NumOps.FromDouble(_model.Count)
        };
    }

    public override void ResetEpisode()
    {
        // No episode-specific state
    }

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

    public override int FeatureCount => _options.StateSize;

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
