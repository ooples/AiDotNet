using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;

using AiDotNet.ReinforcementLearning.Parameters;

namespace AiDotNet.ReinforcementLearning.Agents.MonteCarlo;

/// <summary>
/// Every-Visit Monte Carlo agent that updates all visits to states in an episode.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Every-Visit Monte Carlo learns by playing complete episodes
/// (from start to finish) and then averaging the total reward received. Unlike First-Visit MC
/// which only counts the first time a state is seen, this counts every visit. Think of it
/// like a student who reviews every practice problem, not just the first attempt. This gives
/// more data points per episode but with potentially correlated samples. Good for episodic
/// tasks like board games where you learn from complete games.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an Every-Visit Monte Carlo agent for episodic tasks
/// var options = new MonteCarloOptions&lt;double&gt; { StateSize = 4, ActionSize = 2 };
/// var agent = new EveryVisitMonteCarloAgent&lt;double&gt;(options);
///
/// // Select an action using epsilon-greedy over learned Q-values
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
public partial class EveryVisitMonteCarloAgent<T> : ReinforcementLearningAgentBase<T>, IGradientComputable<T, Vector<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>Entry-based like its siblings, but this one appended a single zero when the table was empty so the surface is never zero-length.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new QTableEntriesParameterSource<T>(_qTable, padEmptyToOne: true));
    }
    private MonteCarloOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Dictionary<string, Dictionary<int, T>> _qTable;
    private Dictionary<string, Dictionary<int, List<T>>> _returns;
    private List<(string state, int action, T reward)> _episode;
    private double _epsilon;
    private Random _random;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public EveryVisitMonteCarloAgent()
        : this(new MonteCarloOptions<T> { StateSize = 4, ActionSize = 2 })
    {
    }

    public EveryVisitMonteCarloAgent(MonteCarloOptions<T> options)
        : base(options)
    {
        if (options == null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        _options = options;

        // Validate EpsilonDecay is in (0, 1] range (1.0 means no decay, which is valid)
        if (_options.EpsilonDecay <= 0.0 || _options.EpsilonDecay > 1.0)
        {
            throw new ArgumentException("EpsilonDecay must be in the range (0, 1] for proper decay behavior.", nameof(options));
        }

        _qTable = new Dictionary<string, Dictionary<int, T>>();
        _returns = new Dictionary<string, Dictionary<int, List<T>>>();
        _episode = new List<(string, int, T)>();
        _epsilon = _options.EpsilonStart;
        _random = Random;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        string stateKey = VectorToStateKey(state);
        int actionIndex;
        if (training && _random.NextDouble() < _epsilon)
        {
            actionIndex = _random.Next(_options.ActionSize);
        }
        else
        {
            actionIndex = GetBestAction(stateKey);
        }
        var action = new Vector<T>(_options.ActionSize);
        action[actionIndex] = NumOps.One;
        return action;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        string stateKey = VectorToStateKey(state);
        int actionIndex = GetActionIndex(action);
        _episode.Add((stateKey, actionIndex, reward));

        if (done)
        {
            UpdateFromEpisode();
            _episode.Clear();
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        }
    }

    private void UpdateFromEpisode()
    {
        T G = NumOps.Zero;

        for (int t = _episode.Count - 1; t >= 0; t--)
        {
            var (state, action, reward) = _episode[t];
            G = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, G));

            EnsureStateExists(state);
            if (!_returns.ContainsKey(state))
            {
                _returns[state] = new Dictionary<int, List<T>>();
            }
            if (!_returns[state].ContainsKey(action))
            {
                _returns[state][action] = new List<T>();
            }

            _returns[state][action].Add(G);
            _qTable[state][action] = ComputeAverage(_returns[state][action]);
        }
    }

    public override T Train() { return NumOps.Zero; }

    /// <summary>
    /// Converts a state vector to a string key for the Q-table.
    /// Uses F8 precision (8 decimal places) to minimize state collisions.
    /// Note: States differing only beyond 8 decimal places will be treated as identical.
    /// </summary>
    private string VectorToStateKey(Vector<T> state)
    {
        var parts = new string[state.Length];
        for (int i = 0; i < state.Length; i++)
        {
            parts[i] = NumOps.ToDouble(state[i]).ToString("F8");
        }
        return string.Join(",", parts);
    }

    /// <summary>
    /// Gets the index of the selected action from a one-hot encoded action vector.
    /// </summary>
    /// <param name="action">One-hot encoded action vector.</param>
    /// <returns>Index of the action with value greater than zero.</returns>
    /// <exception cref="ArgumentException">Thrown when action vector is invalid (all elements &lt;= 0).</exception>
    private int GetActionIndex(Vector<T> action)
    {
        if (action == null)
        {
            throw new ArgumentNullException(nameof(action));
        }

        for (int i = 0; i < action.Length; i++)
        {
            if (NumOps.GreaterThan(action[i], NumOps.Zero))
            {
                return i;
            }
        }

        // Invalid action vector - all elements are <= 0
        throw new ArgumentException("Invalid action vector: all elements are <= 0. Expected one-hot encoded vector with exactly one positive element.", nameof(action));
    }

    private void EnsureStateExists(string stateKey)
    {
        if (!_qTable.ContainsKey(stateKey))
        {
            _qTable[stateKey] = new Dictionary<int, T>();
            for (int a = 0; a < _options.ActionSize; a++)
            {
                _qTable[stateKey][a] = NumOps.Zero;
            }
        }
    }

    private int GetBestAction(string stateKey)
    {
        EnsureStateExists(stateKey);
        int bestAction = 0;
        T bestValue = _qTable[stateKey][0];
        bool allEqual = true;
        for (int a = 1; a < _options.ActionSize; a++)
        {
            if (NumOps.GreaterThan(_qTable[stateKey][a], bestValue))
            {
                bestValue = _qTable[stateKey][a];
                bestAction = a;
                allEqual = false;
            }
            else if (!NumOps.Equals(_qTable[stateKey][a], bestValue))
            {
                allEqual = false;
            }
        }
        if (allEqual)
            bestAction = HashStateToAction(stateKey, _options.ActionSize);
        return bestAction;
    }

    /// <summary>
    /// Computes the average of a list of returns.
    /// </summary>
    /// <param name="returns">List of return values.</param>
    /// <returns>The average return value.</returns>
    private T ComputeAverage(List<T> returns)
    {
        if (returns == null || returns.Count == 0)
        {
            return NumOps.Zero;
        }

        T sum = NumOps.Zero;
        foreach (T value in returns)
        {
            sum = NumOps.Add(sum, value);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(returns.Count));
    }

    public override void ResetEpisode() { _episode.Clear(); base.ResetEpisode(); }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    }

    public override int FeatureCount => _options.StateSize;

    public Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Returned GetParameters() -- the WEIGHTS -- where this interface promises
        // "gradients with respect to all model parameters". The vector length matches, so
        // nothing downstream could detect the substitution: ApplyGradients would subtract the
        // weights from themselves, and Elastic Weight Consolidation / Gradient Episodic Memory /
        // Memory Aware Synapses would build Fisher-information estimates out of parameter
        // magnitudes. A distributed trainer averaging these across workers was averaging
        // parameters and calling the result a gradient.
        //
        // This agent is tabular: it owns no network and its update is a value backup, not a
        // differentiable loss, so no parameter gradient exists here to return.
        throw new NotSupportedException(
            "EveryVisitMonteCarloAgent is a tabular agent whose update is a value backup rather than a "
            + "differentiable loss, so it has no parameter gradients for this interface to "
            + "return. Call Train() instead.");
    }

    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // An empty body silently accepted a gradient vector and did nothing with it, so a
        // caller applying gradients believed the update landed. Refusing is the honest
        // contract, and matches ComputeGradients above.
        throw new NotSupportedException(
            "EveryVisitMonteCarloAgent is a tabular agent and does not apply gradient updates. "
            + "Call Train() instead.");
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
