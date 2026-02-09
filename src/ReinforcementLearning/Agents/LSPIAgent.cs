using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;

namespace AiDotNet.ReinforcementLearning.Agents.AdvancedRL;

/// <summary>
/// LSPI (Least-Squares Policy Iteration) agent using iterative policy improvement with LSTDQ.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LSPIAgent<T> : ReinforcementLearningAgentBase<T>
{
    private LSPIOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Matrix<T> _weights;  // Weight matrix: [ActionSize x FeatureSize]
    private List<(Vector<T> state, int action, T reward, Vector<T> nextState, bool done)> _samples;
    private int _iterations;

    public LSPIAgent(LSPIOptions<T> options) : base(options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
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
    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T> { ModelType = ModelType.ReinforcementLearning, FeatureCount = this.FeatureCount, Complexity = ParameterCount };
    public override int ParameterCount => _options.ActionSize * _options.FeatureSize;
    public override int FeatureCount => _options.FeatureSize;
    public override byte[] Serialize()
    {
        var state = new
        {
            Weights = _weights,
            Samples = _samples,
            Iterations = _iterations,
            Options = _options
        };
        string json = JsonConvert.SerializeObject(state);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }

    public override void Deserialize(byte[] data)
    {
        if (data is null || data.Length == 0)
        {
            throw new ArgumentException("Serialized data cannot be null or empty", nameof(data));
        }

        string json = System.Text.Encoding.UTF8.GetString(data);
        var state = JsonConvert.DeserializeObject<dynamic>(json);
        if (state is null)
        {
            throw new InvalidOperationException("Deserialization returned null");
        }

        // Create matrix with correct dimensions from options
        _weights = new Matrix<T>(_options.ActionSize, _options.FeatureSize);

        // Parse weights matrix from JArray structure
        var weightsObj = state.Weights;
        if (weightsObj is Newtonsoft.Json.Linq.JArray jArray)
        {
            for (int r = 0; r < _options.ActionSize && r < jArray.Count; r++)
            {
                var rowArray = jArray[r] as Newtonsoft.Json.Linq.JArray;
                if (rowArray is not null)
                {
                    for (int c = 0; c < _options.FeatureSize && c < rowArray.Count; c++)
                    {
                        _weights[r, c] = NumOps.FromDouble((double)rowArray[c]);
                    }
                }
            }
        }

        // Deserialize samples list
        _samples = new List<(Vector<T>, int, T, Vector<T>, bool)>();
        var samplesObj = state.Samples;
        if (samplesObj is Newtonsoft.Json.Linq.JArray samplesArray)
        {
            foreach (var sample in samplesArray.OfType<Newtonsoft.Json.Linq.JObject>())
            {
                // Deserialize and validate state vector (Item1)
                var stateArray = sample["Item1"] as Newtonsoft.Json.Linq.JArray;
                if (stateArray is null || stateArray.Count != _options.FeatureSize)
                {
                    throw new InvalidOperationException(
                        $"Sample state vector dimension mismatch: expected {_options.FeatureSize}, " +
                        $"got {stateArray?.Count ?? 0}.");
                }

                var stateVec = new Vector<T>(stateArray.Count);
                for (int i = 0; i < stateArray.Count; i++)
                {
                    stateVec[i] = NumOps.FromDouble(Convert.ToDouble(stateArray[i]));
                }

                // Deserialize and validate action (Item2)
                int action = sample["Item2"] is not null ? Convert.ToInt32(sample["Item2"]) : 0;
                if (action < 0 || action >= _options.ActionSize)
                {
                    throw new InvalidOperationException(
                        $"Sample action index out of range: {action} (valid range: 0-{_options.ActionSize - 1}).");
                }

                // Deserialize reward (Item3)
                T reward = NumOps.FromDouble(sample["Item3"] is not null ? Convert.ToDouble(sample["Item3"]) : 0.0);

                // Deserialize and validate next state vector (Item4)
                var nextStateArray = sample["Item4"] as Newtonsoft.Json.Linq.JArray;
                if (nextStateArray is null || nextStateArray.Count != _options.FeatureSize)
                {
                    throw new InvalidOperationException(
                        $"Sample next state vector dimension mismatch: expected {_options.FeatureSize}, " +
                        $"got {nextStateArray?.Count ?? 0}.");
                }

                var nextStateVec = new Vector<T>(nextStateArray.Count);
                for (int i = 0; i < nextStateArray.Count; i++)
                {
                    nextStateVec[i] = NumOps.FromDouble(Convert.ToDouble(nextStateArray[i]));
                }

                // Deserialize done flag (Item5)
                bool done = sample["Item5"] is not null && Convert.ToBoolean(sample["Item5"]);

                _samples.Add((stateVec, action, reward, nextStateVec, done));
            }
        }

        _iterations = Convert.ToInt32(state.Iterations);
    }

    public override Vector<T> GetParameters()
    {
        int paramCount = _options.ActionSize * _options.FeatureSize;
        var vector = new Vector<T>(paramCount);
        int idx = 0;

        for (int a = 0; a < _options.ActionSize; a++)
            for (int f = 0; f < _options.FeatureSize; f++)
                vector[idx++] = _weights[a, f];

        return vector;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        for (int a = 0; a < _options.ActionSize; a++)
            for (int f = 0; f < _options.FeatureSize; f++)
                if (idx < parameters.Length)
                    _weights[a, f] = parameters[idx++];
    }

    public override IFullModel<T, Vector<T>, Vector<T>> Clone()
    {
        var clone = new LSPIAgent<T>(_options);
        // Copy learned weights
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                clone._weights[a, f] = _weights[a, f];
            }
        }
        // Copy samples and iterations
        clone._samples.AddRange(_samples);
        clone._iterations = _iterations;
        return clone;
    }

    public override Vector<T> ComputeGradients(Vector<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var pred = Predict(input);
        var lf = lossFunction ?? LossFunction;
        var predMatrix = new Matrix<T>(new[] { pred });
        var targetMatrix = new Matrix<T>(new[] { target });
        var loss = lf.CalculateLoss(predMatrix.GetRow(0), targetMatrix.GetRow(0));
        var grad = lf.CalculateDerivative(predMatrix.GetRow(0), targetMatrix.GetRow(0));
        return grad;
    }

    public override void ApplyGradients(Vector<T> gradients, T learningRate) { }
    public override void SaveModel(string filepath) { var data = Serialize(); System.IO.File.WriteAllBytes(filepath, data); }
    public override void LoadModel(string filepath) { var data = System.IO.File.ReadAllBytes(filepath); Deserialize(data); }
}
