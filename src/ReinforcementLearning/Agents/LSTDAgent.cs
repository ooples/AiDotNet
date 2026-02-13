using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Agents.AdvancedRL;

/// <summary>
/// LSTD (Least-Squares Temporal Difference) agent using direct solution for value function weights.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LSTDAgent<T> : ReinforcementLearningAgentBase<T>
{
    private LSTDOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private Matrix<T> _weights;  // Weight matrix: [ActionSize x FeatureSize]
    private Matrix<T> _A;  // A matrix for least-squares: [FeatureSize x FeatureSize]
    private Vector<T> _b;  // b vector for least-squares: [FeatureSize]
    private List<(Vector<T> state, int action, T reward, Vector<T> nextState, bool done)> _samples;
    private int _currentAction;

    public LSTDAgent(LSTDOptions<T> options) : base(options)
    {
        Guard.NotNull(options);
        _options = options;
        _weights = new Matrix<T>(_options.ActionSize, _options.FeatureSize);
        _A = new Matrix<T>(_options.FeatureSize, _options.FeatureSize);
        _b = new Vector<T>(_options.FeatureSize);
        _samples = new List<(Vector<T>, int, T, Vector<T>, bool)>();
        _currentAction = 0;

        // Initialize weights and matrices to zero
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                _weights[a, f] = NumOps.Zero;
            }
        }

        for (int i = 0; i < _options.FeatureSize; i++)
        {
            _b[i] = NumOps.Zero;
            for (int j = 0; j < _options.FeatureSize; j++)
            {
                _A[i, j] = NumOps.Zero;
            }
        }
    }

    public override Vector<T> SelectAction(Vector<T> state, bool training = true)
    {
        // Greedy action selection based on current Q-values
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

        _currentAction = bestAction;
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

        // Solve LSTD for each action separately
        for (int targetAction = 0; targetAction < _options.ActionSize; targetAction++)
        {
            // Reset A and b for this action
            for (int i = 0; i < _options.FeatureSize; i++)
            {
                _b[i] = NumOps.Zero;
                for (int j = 0; j < _options.FeatureSize; j++)
                {
                    _A[i, j] = NumOps.Zero;
                }
            }

            // Accumulate A and b from samples where action was taken
            foreach (var (state, action, reward, nextState, done) in _samples)
            {
                if (action != targetAction) continue;

                // Find best next action
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
                        _A[j, i] = NumOps.Add(_A[j, i], increment);
                    }
                }

                // b += φ(s,a)r
                for (int i = 0; i < _options.FeatureSize; i++)
                {
                    T increment = NumOps.Multiply(phi[i], reward);
                    _b[i] = NumOps.Add(_b[i], increment);
                }
            }

            // Add regularization: A += λI
            T regParam = NumOps.FromDouble(_options.RegularizationParam);
            for (int i = 0; i < _options.FeatureSize; i++)
            {
                _A[i, i] = NumOps.Add(_A[i, i], regParam);
            }

            // Solve: w = A^-1 * b using Gaussian elimination
            Vector<T> w = SolveLinearSystem(_A, _b);

            // Update weights for this action
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                _weights[targetAction, f] = w[f];
            }
        }

        return NumOps.Zero;
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
            Weights = GetParameters(),  // Serialize as flat vector for consistency
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

        // Parse and validate weights as flat vector
        var weightsObj = state.Weights;
        if (weightsObj is null)
        {
            throw new InvalidOperationException("Failed to deserialize agent state: Weights property is missing or null.");
        }

        if (weightsObj is not Newtonsoft.Json.Linq.JArray jArray)
        {
            throw new InvalidOperationException($"Failed to deserialize agent state: Weights must be a JSON array, got {weightsObj.GetType().Name}.");
        }

        int expectedCount = _options.ActionSize * _options.FeatureSize;
        if (jArray.Count != expectedCount)
        {
            throw new InvalidOperationException($"Weight count mismatch: expected {expectedCount} (ActionSize={_options.ActionSize} × FeatureSize={_options.FeatureSize}), got {jArray.Count}.");
        }

        var weights = new Vector<T>(jArray.Count);
        for (int i = 0; i < jArray.Count; i++)
        {
            weights[i] = NumOps.FromDouble((double)jArray[i]);
        }
        SetParameters(weights);
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
        var clone = new LSTDAgent<T>(_options);

        // Deep copy weights matrix
        for (int a = 0; a < _options.ActionSize; a++)
        {
            for (int f = 0; f < _options.FeatureSize; f++)
            {
                clone._weights[a, f] = _weights[a, f];
            }
        }

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
