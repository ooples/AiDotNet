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
/// var options = new LSPIOptions&lt;double&gt; { StateSize = 4, ActionSize = 2 };
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

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
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
        _weights = new Matrix<T>(_options.ActionSize, _options.FeatureSize);
        _samples = new List<(Vector<T>, int, T, Vector<T>, bool)>();
        _iterations = 0;

        // Initialize weights to zero
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                _weights[a, f] = NumOps.Zero;
            }
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Greedy action selection based on current Q-values
        int bestAction = GetGreedyAction(state);

        var result = new Vector<T>(_options.ActionSize);
        result[bestAction] = NumOps.One;
        return result;
    }

    public override void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
    {
        int actionIndex = ArgMax(action);
        _samples.Add((state, actionIndex, reward, nextState, done));
    }

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
                for (int i = 0; i < _options.FeatureSize; i++)
                {
                    A[i, i] = NumOps.Add(A[i, i], regParam);
                }

                // Solve: w = A^-1 * b
                Vector<T> w = SolveLinearSystem(A, b);

                // Update weights for this action
                for (int f = 0; f < _options.FeatureSize; f++)
                {
                    _weights[targetAction, f] = w[f];
                }
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
        var A = new Matrix<T>(_options.FeatureSize, _options.FeatureSize);
        var b = new Vector<T>(_options.FeatureSize);

        // Initialize to zero
        for (int i = 0; i < _options.FeatureSize; i++)
        {
            b[i] = NumOps.Zero;
            for (int j = 0; j < _options.FeatureSize; j++)
            {
                A[i, j] = NumOps.Zero;
            }
        }

        // Accumulate A and b from samples where target action was taken
        foreach (var (state, action, reward, nextState, done) in _samples)
        {
            if (action != targetAction) continue;

            // Find best next action using current policy
            int nextAction = done ? 0 : GetGreedyAction(nextState);

            // Compute φ(s,a) and φ(s',a')
            Vector<T> phi = state;
            Vector<T> phiNext = done ? new Vector<T>(_options.FeatureSize) : nextState;

            // A += φ(s,a)(φ(s,a) - γφ(s',a'))^T
            for (int i = 0; i < _options.FeatureSize; i++)
            {
                T diff = done ? phi[i] : NumOps.Subtract(phi[i], NumOps.Multiply(DiscountFactor, phiNext[i]));
                for (int j = 0; j < _options.FeatureSize; j++)
                {
                    T increment = NumOps.Multiply(phi[j], diff);
                    A[j, i] = NumOps.Add(A[j, i], increment);
                }
            }

            // b += φ(s,a)r
            for (int i = 0; i < _options.FeatureSize; i++)
            {
                T increment = NumOps.Multiply(phi[i], reward);
                b[i] = NumOps.Add(b[i], increment);
            }
        }

        return (A, b);
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
        var clone = new Matrix<T>(_options.ActionSize, _options.FeatureSize);
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                clone[a, f] = weights[a, f];
            }
        }
        return clone;
    }

    private T ComputeWeightChange(Matrix<T> w1, Matrix<T> w2)
    {
        T sumSquaredDiff = NumOps.Zero;
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                T diff = NumOps.Subtract(w1[a, f], w2[a, f]);
                T squared = NumOps.Multiply(diff, diff);
                sumSquaredDiff = NumOps.Add(sumSquaredDiff, squared);
            }
        }
        return NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSquaredDiff)));
    }

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
        ["samples_collected"] = NumOps.FromDouble(_samples.Count),
        ["iterations"] = NumOps.FromDouble(_iterations),
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
