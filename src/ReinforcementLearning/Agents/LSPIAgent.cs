using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

using AiDotNet.ReinforcementLearning.Parameters;

namespace AiDotNet.ReinforcementLearning.Agents.AdvancedRL;

/// <summary>
/// LSPI (Least-Squares Policy Iteration) agent using iterative policy improvement with LSTDQ.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> LSPI is a batch RL algorithm that makes the most efficient use
/// of collected data. Instead of learning from one experience at a time, it collects a batch
/// of experiences and uses linear algebra to find the best policy in one shot. Think of it
/// like studying all past exam questions at once rather than one at a time. This makes it
/// very sample-efficient but requires storing all experiences in memory.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an LSPI agent for batch policy iteration
/// var options = new LSPIOptions&lt;double&gt; { FeatureSize = 4, ActionSize = 2 };
/// var agent = new LSPIAgent&lt;double&gt;(options);
///
/// // Select an action using the current policy weights
/// var state = new Vector&lt;double&gt;(new double[] { 0.5, -0.3, 1.0, 0.2 });
/// var action = agent.SelectAction(state);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Least-Squares Policy Iteration",
    "https://www.jmlr.org/papers/v4/lagoudakis03a.html",
    Year = 2003,
    Authors = "Lagoudakis, M. G. & Parr, R.")]
public partial class LSPIAgent<T> : ReinforcementLearningAgentBase<T>
{

    /// <inheritdoc />
    /// <remarks>The linear weight matrix, row-major, which is what the hand-written loop over
    /// [action, feature] produced. Registered through an accessor because this agent can
    /// REPLACE the matrix rather than mutate it.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new MatrixParameterSource<T>(() => _weights));
    }
    private LSPIOptions<T> _options;

    private Matrix<T> _weights;  // Weight matrix: [ActionSize x FeatureSize]
    private List<(Vector<T> state, int action, T reward, Vector<T> nextState, bool done)> _samples;
    private int _iterations;

    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public LSPIAgent()
        // FeatureSize must match the length of the state vectors fed in (LSPI uses raw-state
        // features, phi(s) = s). The previous parameterless default left FeatureSize at 0, so the
        // weight matrix was [ActionSize x 0] — empty parameters, all-zero Q-values, and no learning.
        // Default to 4 features, matching the documented StateSize = 4 example; callers with a
        // different state dimension set FeatureSize explicitly.
        : this(new LSPIOptions<T> { ActionSize = 2, FeatureSize = 4 })
    {
    }

    public LSPIAgent(LSPIOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        Options = _options;
        _samples = new List<(Vector<T>, int, T, Vector<T>, bool)>();
        _iterations = 0;

        // Initialize weights to zero
        _weights = Matrix<T>.CreateDefault(_options.ActionSize, _options.FeatureSize, NumOps.Zero);
    }

    /// <inheritdoc />
    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Greedy action selection based on current Q-values
        int bestAction = GetGreedyAction(state);

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    /// <inheritdoc />
    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int actionIndex = ArgMax(action);
        _samples.Add((state, actionIndex, reward, nextState, done));
    }

    /// <inheritdoc />
    public override T Train()
    {
        if (_samples.Count == 0) return NumOps.Zero;

        // LSPI (Lagoudakis & Parr 2003, Figure 8): repeat LSTDQ policy evaluation of the policy greedy in the
        // current weights, one joint solve over phi(s, a), until ||w - w'|| < epsilon.
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            _iterations = iter + 1;

            var previousWeights = CloneWeights(_weights);
            var updatedWeights = Lstdq(previousWeights);
            _weights = updatedWeights;

            if (HasConverged(previousWeights, updatedWeights))
            {
                break;
            }
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// LSTDQ (Lagoudakis &amp; Parr 2003, Figure 5): evaluates the policy greedy in <paramref name="policyWeights"/> on
    /// every stored sample with one joint system over the state-action features,
    /// A = sum phi(s, a) (phi(s, a) - gamma phi(s', pi(s')))^T + lambda I and b = sum phi(s, a) r, and returns
    /// w = A^-1 b laid out as the [ActionSize x FeatureSize] weight matrix (row a = block a of w).
    /// </summary>
    private Matrix<T> Lstdq(Matrix<T> policyWeights)
    {
        var engine = AiDotNetEngine.Current;
        int jointSize = JointFeatureSize;
        var flatPolicyWeights = Flatten(policyWeights);

        var phiRows = _samples.Select(sample => StateActionFeaturesChecked(sample.state, sample.action)).ToList();
        // phi(s, a) - gamma phi(s', pi(s')); a terminal step has no successor (phi(s', .) = 0).
        var diffRows = _samples.Select((sample, i) => sample.done
                ? phiRows[i]
                : engine.Subtract(phiRows[i], engine.Multiply(
                    StateActionFeaturesChecked(sample.nextState, GreedyAction(flatPolicyWeights, sample.nextState)), DiscountFactor)))
            .ToList();
        var rewards = _samples.Select(sample => sample.reward).ToList();

        var (a, b) = RLTensorOps.LeastSquaresTdSystem(phiRows, diffRows, rewards, jointSize);
        a = RLTensorOps.AddToDiagonal(a, NumOps.FromDouble(_options.RegularizationParam));

        var solution = AiDotNet.Helpers.LinalgDispatch<T>.Solve(Tensor<T>.FromMatrix(a), new Tensor<T>(new[] { jointSize, 1 }, b));
        var weights = new Matrix<T>(_options.ActionSize, _options.FeatureSize);
        solution.AsSpan().CopyTo(weights.AsWritableSpan());
        return weights;
    }

    /// <summary>Length of the joint state-action feature vector phi(s, a): ActionSize * FeatureSize.</summary>
    protected int JointFeatureSize => _options.ActionSize * _options.FeatureSize;

    /// <summary>
    /// The state-action basis phi(s, a) that LSTDQ regresses Q(s, a) = w . phi(s, a) on. The default is the
    /// paper's block basis: the first FeatureSize state entries placed in action a's block, zeros elsewhere, so
    /// w's block a is the weight row for action a. Override to supply another basis of the same length
    /// (<see cref="JointFeatureSize"/>).
    /// </summary>
    protected virtual Vector<T> StateActionFeatures(Vector<T> state, int action)
    {
        Guard.NotNull(state);
        int featureSize = _options.FeatureSize;
        return RLTensorOps.Concat(
            new Vector<T>(action * featureSize),
            RLTensorOps.Head(state, featureSize),
            new Vector<T>((_options.ActionSize - action - 1) * featureSize));
    }

    private Vector<T> StateActionFeaturesChecked(Vector<T> state, int action)
    {
        var features = StateActionFeatures(state, action);
        if (features is null || features.Length != JointFeatureSize)
            throw new InvalidOperationException(
                $"StateActionFeatures must return {JointFeatureSize} entries (ActionSize * FeatureSize).");
        return features;
    }

    private static Vector<T> Flatten(Matrix<T> weights) => Vector<T>.FromSpan(weights.AsSpan());

    /// <summary>First arg-max over a of w . phi(s, a).</summary>
    private int GreedyAction(Vector<T> flatWeights, Vector<T> state)
    {
        var engine = AiDotNetEngine.Current;
        var qValues = Vector<T>.FromEnumerable(Enumerable.Range(0, _options.ActionSize)
            .Select(action => engine.DotProduct(flatWeights, StateActionFeaturesChecked(state, action))).ToList());
        return RLTensorOps.ArgMax(qValues);
    }

    private Matrix<T> CloneWeights(Matrix<T> weights)
    {
        return weights.Clone();
    }

    private T ComputeWeightChange(Matrix<T> w1, Matrix<T> w2)
    {
        // ||w1 - w2||_F
        return RLTensorOps.FrobeniusNorm(AiDotNetEngine.Current.MatrixSubtract(w1, w2));
    }

    /// <summary>The paper's stopping rule ||w - w'|| &lt; epsilon.</summary>
    private bool HasConverged(Matrix<T> previousWeights, Matrix<T> updatedWeights)
        => NumOps.ToDouble(ComputeWeightChange(previousWeights, updatedWeights)) < _options.ConvergenceThreshold;

    private int GetGreedyAction(Vector<T> state)
    {
        // First arg-max of Q(s, a) = w . phi(s, a) over the actions.
        return GreedyAction(Flatten(_weights), state);
    }

    private int ArgMax(Vector<T> values)
    {
        return RLTensorOps.ArgMax(values);
    }

    /// <inheritdoc />
    public override Dictionary<string, T> GetMetrics() => new Dictionary<string, T>
    {
        ["samples_collected"] = NumOps.FromDouble(_samples.Count),
        ["iterations"] = NumOps.FromDouble(_iterations),
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
