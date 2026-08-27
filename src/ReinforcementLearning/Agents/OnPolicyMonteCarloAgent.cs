using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.MonteCarlo;

/// <summary>
/// On-Policy Monte Carlo Control agent with epsilon-greedy exploration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>On-Policy MC Control uses epsilon-greedy policy for both behavior and target,
/// ensuring exploration while learning the optimal policy.</para>
/// <para><b>For Beginners:</b> On-policy MC learns by evaluating the same policy it uses
/// to collect data. The agent follows an epsilon-greedy strategy (mostly best action,
/// sometimes random) and improves that exact strategy over time. Think of learning to cook
/// by actually cooking with your current recipe, then adjusting based on results. Simpler
/// than off-policy methods but cannot reuse data from previous policies.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an on-policy Monte Carlo agent with epsilon-greedy exploration
/// var options = new OnPolicyMonteCarloOptions&lt;double&gt; { StateSize = 4, ActionSize = 2 };
/// var agent = new OnPolicyMonteCarloAgent&lt;double&gt;(options);
///
/// // Select an action following the epsilon-greedy policy
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
public partial class OnPolicyMonteCarloAgent<T> : ReinforcementLearningAgentBase<T>
{
    private OnPolicyMonteCarloOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, List<T>>> _returns;
    private List<(Vector<T> state, int action, T reward)> _episode;
    private double _epsilon;
    private Random _random;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public OnPolicyMonteCarloAgent()
        : this(new OnPolicyMonteCarloOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public OnPolicyMonteCarloAgent(OnPolicyMonteCarloOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;
        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _returns = new Dictionary<string, Dictionary<int, List<T>>>();
        _episode = new List<(Vector<T>, int, T)>();
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
            // Explore: random action
            selectedAction = _random.Next(_options.ActionSize);
        }
        else
        {
            // Exploit: greedy action with Sutton & Barto §2.3 tie-break.
            // On a fresh state where all Q(s, ·) are still at their
            // initialization values (zero), every action ties and a naive
            // "pick index 0" rule produces a degenerate constant policy
            // that maps every state to action 0 — DifferentStates_DifferentActions
            // catches this. When all Q-values are equal, hash the state
            // key to a deterministic-but-state-dependent action so
            // distinct unvisited states get distinct initial choices.
            selectedAction = 0;
            T bestValue = _qTable[stateKey][0];
            bool allEqual = true;

            for (int a = 1; a < _options.ActionSize; a++)
            {
                if (NumOps.GreaterThan(_qTable[stateKey][a], bestValue))
                {
                    bestValue = _qTable[stateKey][a];
                    selectedAction = a;
                    allEqual = false;
                }
                else if (!NumOps.Equals(_qTable[stateKey][a], bestValue))
                {
                    allEqual = false;
                }
            }

            if (allEqual)
            {
                // string.GetHashCode is randomized per-process in .NET Core+
                // (documented at learn.microsoft.com/.../system.string.gethashcode),
                // so the tie-breaking action would change across runs and
                // make the policy non-reproducible even with a fixed RNG
                // seed. Route through the inherited stable SHA1-based
                // HashStateToAction helper so the same state always maps
                // to the same tie-broken action.
                selectedAction = HashStateToAction(stateKey, _options.ActionSize);
            }
        }

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int actionIndex = ArgMax(action);
        _episode.Add((state, actionIndex, reward));

        if (done)
        {
            UpdateFromEpisode();
            _episode.Clear();

            // Decay epsilon
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        }
    }

    public override T Train()
    {
        // Training happens during episode completion in StoreExperience
        return NumOps.Zero;
    }

    private void UpdateFromEpisode()
    {
        T G = NumOps.Zero;
        var visited = new HashSet<string>();

        // Process episode backward (first-visit MC)
        for (int t = _episode.Count - 1; t >= 0; t--)
        {
            var (state, action, reward) = _episode[t];
            G = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, G));

            string stateKey = GetStateKey(state);
            string stateActionKey = $"{stateKey}_{action}";

            // First-visit: only update first occurrence
            if (!visited.Contains(stateActionKey))
            {
                visited.Add(stateActionKey);

                EnsureStateExists(state);
                if (!_returns.ContainsKey(stateKey))
                {
                    _returns[stateKey] = new Dictionary<int, List<T>>();
                }
                if (!_returns[stateKey].ContainsKey(action))
                {
                    _returns[stateKey][action] = new List<T>();
                }

                _returns[stateKey][action].Add(G);
                _qTable[stateKey][action] = ComputeAverage(_returns[stateKey][action]);
            }
        }
    }

    private void EnsureStateExists(Vector<T> state)
    {
        string stateKey = GetStateKey(state);

        if (!_qTable.ContainsKey(stateKey))
        {
            // Sutton & Barto §5.4: "Initialize Q(s,a) arbitrarily." The
            // canonical choice (kept here) is exact zeros — the greedy
            // tie-break in SelectAction reads from the state key when all
            // Q-values are equal, so distinct states still produce
            // distinct argmax actions even before any returns arrive.
            _qTable[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private string GetStateKey(Vector<T> state)
    {
        return string.Join(",", Enumerable.Range(0, state.Length).Select(i => NumOps.ToDouble(state[i]).ToString("F4")));
    }

    private T ComputeAverage(List<T> values)
    {
        if (values.Count == 0)
        {
            return NumOps.Zero;
        }

        T sum = NumOps.Zero;
        foreach (var value in values)
        {
            sum = NumOps.Add(sum, value);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(values.Count));
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
            ["states_visited"] = NumOps.FromDouble(_qTable.Count),
            ["episode_length"] = NumOps.FromDouble(_episode.Count),
            ["epsilon"] = NumOps.FromDouble(_epsilon)
        };
    }

    public override void ResetEpisode()
    {
        _episode.Clear();
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

    /// <summary>
    /// The number of Q-values actually stored, not <c>_qTable.Count * ActionSize</c>. That product
    /// assumes every visited state has explored every action, which a tabular agent does not do --
    /// states appear as they are seen and actions as they are tried.
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

    public override void SaveModel(string filepath)
    {
        throw new NotSupportedException(
            "OnPolicyMonteCarlo persistence is not yet fully supported. " +
            "Use Serialize() for manual serialization or GetParameters() to extract Q-values.");
    }

    public override void LoadModel(string filepath)
    {
        throw new NotSupportedException(
            "OnPolicyMonteCarlo persistence is not yet fully supported. " +
            "Use Deserialize() for manual deserialization or SetParameters() for parameter restoration.");
    }
}
