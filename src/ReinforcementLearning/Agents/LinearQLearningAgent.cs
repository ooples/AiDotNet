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
/// Linear Q-Learning agent using linear function approximation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Linear Q-Learning replaces the Q-table with a linear function
/// Q(s,a) = w dot phi(s,a), where phi extracts features from state-action pairs. This allows
/// handling continuous or large state spaces that would be impossible with tables. Think of it
/// like using a formula instead of a lookup table. The trade-off is that it can only represent
/// linear relationships, but it scales to much larger problems than tabular Q-learning.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a linear Q-Learning agent for continuous state spaces
/// var options = new LinearQLearningOptions&lt;double&gt; { FeatureSize = 4, ActionSize = 2, LearningRate = 0.01 };
/// var agent = new LinearQLearningAgent&lt;double&gt;(options);
///
/// // Select an action using linear function approximation
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
public partial class LinearQLearningAgent<T> : ReinforcementLearningAgentBase<T>
{

    /// <inheritdoc />
    /// <remarks>The linear weight matrix, row-major, which is what the hand-written loop over
    /// [action, feature] produced. Registered through an accessor because this agent can
    /// REPLACE the matrix rather than mutate it.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new MatrixParameterSource<T>(() => _weights));
    }
    private LinearQLearningOptions<T> _options;

    private Matrix<T> _weights;  // Weight matrix: [ActionSize x FeatureSize]
    private double _epsilon;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public LinearQLearningAgent()
        : this(new LinearQLearningOptions<T> { ActionSize = 2, FeatureSize = 4 })
    {
    }

    public LinearQLearningAgent(LinearQLearningOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        // Initialize weights to zero
        _weights = Matrix<T>.CreateDefault(_options.ActionSize, _options.FeatureSize, NumOps.Zero);

        _epsilon = options.EpsilonStart;
    }

    /// <inheritdoc />
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

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int actionIndex = ArgMax(action);

        // Compute current Q-value: Q(s,a) = w_a^T * φ(s)
        T currentQ = ComputeQValue(state, actionIndex);

        // Compute max Q-value for next state
        T maxNextQ = NumOps.Zero;
        if (!done)
        {
            int bestNextAction = GetGreedyAction(nextState);
            maxNextQ = ComputeQValue(nextState, bestNextAction);
        }

        // Compute TD target and error
        T target = NumOps.Add(reward, NumOps.Multiply(DiscountFactor, maxNextQ));
        T tdError = NumOps.Subtract(target, currentQ);

        // Update weights: w_a ← w_a + α * δ * φ(s)
        T learningRateT = NumOps.Multiply(LearningRate, tdError);
        RLTensorOps.AddScaledToRow(_weights, actionIndex, state, learningRateT);

        if (done)
        {
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
        }
    }

    /// <inheritdoc />
    public override T Train() => NumOps.Zero;

    private T ComputeQValue(Vector<T> features, int actionIndex)
    {
        // Q(s,a) = w_a . phi(s)
        return RLTensorOps.LinearQValue(_weights, actionIndex, features);
    }

    private int GetGreedyAction(Vector<T> state)
    {
        // First arg-max of W * phi(s) over the actions.
        return RLTensorOps.ArgMax(RLTensorOps.LinearQValues(_weights, state));
    }

    private int ArgMax(Vector<T> values)
    {
        return RLTensorOps.ArgMax(values);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T>
    {
        ["epsilon"] = NumOps.FromDouble(_epsilon),
        ["weight_norm"] = ComputeWeightNorm()
    };

    private T ComputeWeightNorm()
    {
        return RLTensorOps.FrobeniusNorm(_weights);
    }

    /// <inheritdoc />
    public override void ResetEpisode() { }
    /// <inheritdoc />
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    /// <inheritdoc />
    public override int FeatureCount => _options.FeatureSize;
    /// <inheritdoc />
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    /// <inheritdoc />
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
