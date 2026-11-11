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

        Matrix<T> previousWeights = CloneWeights(_weights);

        // LSPI iterations
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            _iterations = iter + 1;

            // LSTDQ: Solve for Q-function weights for each action
            for (int targetAction = 0; targetAction < _options.ActionSize; targetAction++)
            {
                var (A, b) = ComputeLSTDQMatrices(targetAction);

                // Add regularization: A += λI
                T regParam = NumOps.FromDouble(_options.RegularizationParam);
                A = RLTensorOps.AddToDiagonal(A, regParam);

                // Solve: w = A^-1 * b
                Vector<T> w = SolveLinearSystem(A, b);

                // Update weights for this action
                AiDotNetEngine.Current.SetRow(_weights, targetAction, w);
            }

            // Check convergence
            T weightChange = ComputeWeightChange(previousWeights, _weights);
            if (NumOps.ToDouble(weightChange) < _options.ConvergenceThreshold)
            {
                break;
            }

            previousWeights = CloneWeights(_weights);
        }

        return NumOps.Zero;
    }

    private (Matrix<T> A, Vector<T> b) ComputeLSTDQMatrices(int targetAction)
    {
        // Accumulate A and b from samples where target action was taken
        var engine = AiDotNetEngine.Current;
        var phiRows = new List<Vector<T>>();
        var diffRows = new List<Vector<T>>();
        var sampleRewards = new List<T>();
        foreach (var (state, action, reward, nextState, done) in _samples)
        {
            if (action != targetAction) continue;

            // Find best next action
            int nextAction = done ? 0 : GetGreedyAction(nextState);

            // phi(s,a) and the TD difference phi(s,a) - gamma * phi(s',a') (just phi at a terminal step)
            Vector<T> phi = RLTensorOps.Head(state, _options.FeatureSize);
            phiRows.Add(phi);
            diffRows.Add(done
                ? phi
                : engine.Subtract(phi, engine.Multiply(RLTensorOps.Head(nextState, _options.FeatureSize), DiscountFactor)));
            sampleRewards.Add(reward);
        }

        // A = sum phi (phi - gamma phi')^T and b = sum phi r over those samples
        return RLTensorOps.LeastSquaresTdSystem(phiRows, diffRows, sampleRewards, _options.FeatureSize);
    }

    private Vector<T> SolveLinearSystem(Matrix<T> A, Vector<T> b)
    {
        int n = _options.FeatureSize;
        var augmented = new Matrix<T>(n, n + 1);

        // Create augmented matrix [A|b]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Gaussian elimination with partial pivoting
        for (int k = 0; k < n; k++)
        {
            // Find pivot
            int maxRow = k;
            T maxVal = augmented[k, k];
            for (int i = k + 1; i < n; i++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(augmented[i, k]), NumOps.Abs(maxVal)))
                {
                    maxVal = augmented[i, k];
                    maxRow = i;
                }
            }

            // Swap rows
            if (maxRow != k)
            {
                for (int j = 0; j <= n; j++)
                {
                    T temp = augmented[k, j];
                    augmented[k, j] = augmented[maxRow, j];
                    augmented[maxRow, j] = temp;
                }
            }

            // Forward elimination
            for (int i = k + 1; i < n; i++)
            {
                T factor = NumOps.Divide(augmented[i, k], augmented[k, k]);
                for (int j = k; j <= n; j++)
                {
                    augmented[i, j] = NumOps.Subtract(augmented[i, j], NumOps.Multiply(factor, augmented[k, j]));
                }
            }
        }

        // Back substitution
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            T sum = augmented[i, n];
            for (int j = i + 1; j < n; j++)
            {
                sum = NumOps.Subtract(sum, NumOps.Multiply(augmented[i, j], x[j]));
            }
            x[i] = NumOps.Divide(sum, augmented[i, i]);
        }

        return x;
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

    private T ComputeQValue(Vector<T> features, int actionIndex)
    {
        // Q(s,a) = w_a . phi(s)
        return RLTensorOps.LinearQValue(_weights, actionIndex, features);
    }

    private int GetGreedyAction(Vector<T> state)
    {
        // First arg-max of W * phi(s) over the actions.
        int bestAction = RLTensorOps.ArgMax(RLTensorOps.LinearQValues(_weights, state));

        return bestAction;
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
