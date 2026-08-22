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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
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

        var result = new Vector<T>(_options.ActionSize);
        result[selectedAction] = NumOps.One;
        return result;
    }

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
        for (int f = 0; f < _options.FeatureSize; f++)
        {
            T update = NumOps.Multiply(learningRateT, state[f]);
            _weights[actionIndex, f] = NumOps.Add(_weights[actionIndex, f], update);
        }

        if (done)
        {
            _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);
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

        for (int a = 1; a < _options.ActionSize; a++)
        {
            T value = ComputeQValue(state, a);
            if (NumOps.GreaterThan(value, bestValue))
            {
                bestValue = value;
                bestAction = a;
            }
        }

        return bestAction;
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

    public override void ResetEpisode() { }
    public override Vector<T> Predict(Vector<T> input) => SelectAction(input, false);
    public Task<Vector<T>> PredictAsync(Vector<T> input) => Task.FromResult(Predict(input));
    public Task TrainAsync() { Train(); return Task.CompletedTask; }
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int FeatureCount => _options.FeatureSize;
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
