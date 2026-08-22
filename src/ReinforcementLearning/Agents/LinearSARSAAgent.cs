using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

using AiDotNet.ReinforcementLearning.Parameters;

namespace AiDotNet.ReinforcementLearning.Agents.AdvancedRL;

/// <summary>
/// Linear SARSA agent using linear function approximation with on-policy learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Linear SARSA is like Linear Q-Learning but learns on-policy,
/// meaning it evaluates and improves the policy it is actually following. The name SARSA comes
/// from the update sequence: State, Action, Reward, next State, next Action. This makes it
/// safer for real-world applications because it accounts for the exploration the agent is doing,
/// unlike Q-Learning which assumes optimal future behavior.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a linear SARSA agent for on-policy learning with function approximation
/// var options = new LinearSARSAOptions&lt;double&gt; { FeatureSize = 4, ActionSize = 2, LearningRate = 0.01 };
/// var agent = new LinearSARSAAgent&lt;double&gt;(options);
///
/// // Select an action using linear value function
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Reinforcement Learning: An Introduction",
    "https://incompleteideas.net/book/the-book-2nd.html",
    Year = 2018,
    Authors = "Sutton, R. S. & Barto, A. G.")]
public partial class LinearSARSAAgent<T> : ReinforcementLearningAgentBase<T>
{

    /// <inheritdoc />
    /// <remarks>The linear weight matrix, row-major, which is what the hand-written loop over
    /// [action, feature] produced. Registered through an accessor because this agent can
    /// REPLACE the matrix rather than mutate it.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new MatrixParameterSource<T>(() => _weights));
    }
    private LinearSARSAOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Matrix<T> _weights;  // Weight matrix: [ActionSize x FeatureSize]
    private double _epsilon;
    private int _lastAction = -1;
    [Scratch]
    private Vector<T>? _lastState = null;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public LinearSARSAAgent()
        : this(new LinearSARSAOptions<T> { ActionSize = 2, FeatureSize = 4 })
    {
    }

    public LinearSARSAAgent(LinearSARSAOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _weights = new Matrix<T>(_options.ActionSize, _options.FeatureSize);

        // Initialize weights to zero
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                _weights[a, f] = NumOps.Zero;
            }
        }

        _epsilon = options.EpsilonStart;
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        int selectedAction;
        if (training && Random.NextDouble() < _epsilon)
        {
            selectedAction = Random.Next(_options.ActionSize);
        }
        else
        {
            selectedAction = GetGreedyAction(state);
        }

        _lastState = state;
        _lastAction = selectedAction;

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        if (_lastState == null || _lastAction < 0) return;

        // Compute current Q-value: Q(s,a) = w_a^T * φ(s)
        T currentQ = ComputeQValue(_lastState, _lastAction);

        // Compute next Q-value using the action that will be taken (on-policy)
        T nextQ = NumOps.Zero;
        if (!done)
        {
            // Select next action using current policy
            int nextAction;
            if (Random.NextDouble() < _epsilon)
            {
                nextAction = Random.Next(_options.ActionSize);
            }
            else
            {
                nextAction = GetGreedyAction(nextState);
            }
            nextQ = ComputeQValue(nextState, nextAction);
        }

        // Compute TD target and error
        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, nextQ));
        T tdError = NumOps.Subtract(target, currentQ);

        // Update weights: w_a ← w_a + α * δ * φ(s)
        T learningRateT = NumOps.Multiply(LearningRate, tdError);
        for (int f = 0; f < _options.FeatureSize; f++)
        {
            T update = NumOps.Multiply(learningRateT, _lastState[f]);
            _weights[_lastAction, f] = NumOps.Add(_weights[_lastAction, f], update);
        }

        if (done)
        {
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
            _lastAction = -1;
            _lastState = null;
        }
    }

    public override T Train() => NumOps.Zero;

    private T ComputeQValue(Vector<T> features, int actionIndex)
    {
        T qValue = NumOps.Zero;
        for (int f = 0; f < _options.FeatureSize; f++)
        {
            T weightedFeature = NumOps.Multiply(_weights[actionIndex, f], features[f]);
            qValue = NumOps.Add(qValue, weightedFeature);
        }
        return qValue;
    }

    private int GetGreedyAction(Vector<T> state)
    {
        int bestAction = 0;
        T bestValue = ComputeQValue(state, 0);
        bool allEqual = true;

        for (int a = 1; a < _options.ActionSize; a++)
        {
            T value = ComputeQValue(state, a);
            if (NumOps.GreaterThan(value, bestValue))
            {
                bestValue = value;
                bestAction = a;
                allEqual = false;
            }
            else if (!NumOps.Equals(value, bestValue))
            {
                allEqual = false;
            }
        }

        // Sutton & Barto §2.3 tie-break — break ties using a state-dependent
        // hash so untrained agents don't return action 0 for every input.
        if (allEqual)
        {
            // Build a key from the actual numeric state so close-but-distinct
            // states still produce distinct keys.
            var sb = new System.Text.StringBuilder(state.Length * 8);
            for (int i = 0; i < state.Length; i++)
            {
                sb.Append(NumOps.ToDouble(state[i]).ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                sb.Append(',');
            }
            bestAction = HashStateToAction(sb.ToString(), _options.ActionSize);
        }

        return bestAction;
    }

    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T>
    {
        ["epsilon"] = NumOps.FromDouble(_epsilon),
        ["weight_norm"] = ComputeWeightNorm()
    };

    private T ComputeWeightNorm()
    {
        T sumSquares = NumOps.Zero;
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                T squared = NumOps.Multiply(_weights[a, f], _weights[a, f]);
                sumSquares = NumOps.Add(sumSquares, squared);
            }
        }
        return NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSquares)));
    }

    public override void ResetEpisode()
    {
        _lastAction = -1;
        _lastState = null;
    }

    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int FeatureCount => _options.FeatureSize;
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
