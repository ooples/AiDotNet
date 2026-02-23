using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Interacting Layer for AutoInt architecture.
/// </summary>
/// <remarks>
/// <para>
/// The interacting layer is the core component of AutoInt that learns high-order feature
/// interactions through multi-head self-attention. Each layer captures different orders
/// of interactions between features.
/// </para>
/// <para>
/// <b>For Beginners:</b> The interacting layer helps discover relationships between features:
/// - 1st layer: "age relates to income"
/// - 2nd layer: "age + income together relate to credit score"
/// - 3rd layer: "age + income + credit score relate to loan approval"
///
/// Each layer builds on the previous to capture more complex patterns.
/// The attention mechanism learns which feature combinations are important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class InteractingLayer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _attentionDim;
    private readonly bool _useResidual;

    // Multi-head attention parameters
    private Tensor<T> _queryWeights;   // [embeddingDim, attentionDim]
    private Tensor<T> _keyWeights;     // [embeddingDim, attentionDim]
    private Tensor<T> _valueWeights;   // [embeddingDim, attentionDim]

    // Output projection (combines heads)
    private Tensor<T> _outputWeights;  // [attentionDim, embeddingDim]

    // Residual projection (if dimensions don't match)
    private Tensor<T>? _residualWeights;  // [embeddingDim, embeddingDim] if needed

    // Gradients
    private Tensor<T> _queryWeightsGrad;
    private Tensor<T> _keyWeightsGrad;
    private Tensor<T> _valueWeightsGrad;
    private Tensor<T> _outputWeightsGrad;
    private Tensor<T>? _residualWeightsGrad;

    // Cached values
    private Tensor<T>? _inputCache;
    private Tensor<T>? _queriesCache;
    private Tensor<T>? _keysCache;
    private Tensor<T>? _valuesCache;
    private Tensor<T>? _attentionScoresCache;
    private Tensor<T>? _attendedCache;
    private Tensor<T>? _preActivationCache;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _outputWeights.Length + (_residualWeights?.Length ?? 0);

    /// <summary>
    /// Initializes an interacting layer.
    /// </summary>
    /// <param name="embeddingDim">Input/output embedding dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="attentionDim">Dimension for attention (default: embeddingDim).</param>
    /// <param name="useResidual">Whether to use residual connections.</param>
    /// <param name="initScale">Initialization scale.</param>
    public InteractingLayer(
        int embeddingDim,
        int numHeads = 2,
        int? attentionDim = null,
        bool useResidual = true,
        double initScale = 0.02)
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _attentionDim = attentionDim ?? embeddingDim;
        _headDim = _attentionDim / numHeads;
        _useResidual = useResidual;
        _random = RandomHelper.CreateSecureRandom();

        // Initialize attention weights
        _queryWeights = new Tensor<T>([embeddingDim, _attentionDim]);
        _keyWeights = new Tensor<T>([embeddingDim, _attentionDim]);
        _valueWeights = new Tensor<T>([embeddingDim, _attentionDim]);
        _outputWeights = new Tensor<T>([_attentionDim, embeddingDim]);

        // Initialize gradients
        _queryWeightsGrad = new Tensor<T>([embeddingDim, _attentionDim]);
        _keyWeightsGrad = new Tensor<T>([embeddingDim, _attentionDim]);
        _valueWeightsGrad = new Tensor<T>([embeddingDim, _attentionDim]);
        _outputWeightsGrad = new Tensor<T>([_attentionDim, embeddingDim]);

        // Initialize residual projection if needed
        if (_useResidual && _attentionDim != embeddingDim)
        {
            _residualWeights = new Tensor<T>([embeddingDim, embeddingDim]);
            _residualWeightsGrad = new Tensor<T>([embeddingDim, embeddingDim]);
        }

        InitializeWeights(initScale);
    }

    private void InitializeWeights(double scale)
    {
        // Xavier/Glorot initialization
        double queryKeyScale = scale / Math.Sqrt(_embeddingDim);
        double outputScale = scale / Math.Sqrt(_attentionDim);

        for (int i = 0; i < _queryWeights.Length; i++)
        {
            _queryWeights[i] = NumOps.FromDouble(_random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _keyWeights.Length; i++)
        {
            _keyWeights[i] = NumOps.FromDouble(_random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _valueWeights.Length; i++)
        {
            _valueWeights[i] = NumOps.FromDouble(_random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _outputWeights.Length; i++)
        {
            _outputWeights[i] = NumOps.FromDouble(_random.NextGaussian() * outputScale);
        }

        if (_residualWeights != null)
        {
            for (int i = 0; i < _residualWeights.Length; i++)
            {
                _residualWeights[i] = NumOps.FromDouble(_random.NextGaussian() * scale);
            }
        }
    }

    /// <summary>
    /// Forward pass through the interacting layer.
    /// </summary>
    /// <param name="input">Input embeddings [batchSize, numFeatures, embeddingDim].</param>
    /// <returns>Feature interactions [batchSize, numFeatures, embeddingDim].</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        _inputCache = input;

        int batchSize = input.Shape[0];
        int numFeatures = input.Shape[1];
        int embDim = input.Shape[2];

        // Project to queries, keys, values
        var queries = ProjectInput(input, _queryWeights, batchSize, numFeatures, embDim, _attentionDim);
        var keys = ProjectInput(input, _keyWeights, batchSize, numFeatures, embDim, _attentionDim);
        var values = ProjectInput(input, _valueWeights, batchSize, numFeatures, embDim, _attentionDim);

        _queriesCache = queries;
        _keysCache = keys;
        _valuesCache = values;

        // Multi-head self-attention
        var attended = MultiHeadAttention(queries, keys, values, batchSize, numFeatures);
        _attendedCache = attended;

        // Output projection
        var output = ProjectOutput(attended, batchSize, numFeatures);

        // Residual connection
        if (_useResidual)
        {
            output = AddResidual(input, output, batchSize, numFeatures, embDim);
        }

        _preActivationCache = output;

        // Apply ReLU activation
        output = ApplyReLU(output);

        return output;
    }

    private Tensor<T> ProjectInput(Tensor<T> input, Tensor<T> weights,
        int batchSize, int numFeatures, int inputDim, int outputDim)
    {
        var projected = new Tensor<T>([batchSize, numFeatures, outputDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                for (int o = 0; o < outputDim; o++)
                {
                    var sum = NumOps.Zero;
                    for (int i = 0; i < inputDim; i++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(
                            input[b * numFeatures * inputDim + f * inputDim + i],
                            weights[i * outputDim + o]));
                    }
                    projected[b * numFeatures * outputDim + f * outputDim + o] = sum;
                }
            }
        }

        return projected;
    }

    private Tensor<T> MultiHeadAttention(Tensor<T> queries, Tensor<T> keys, Tensor<T> values,
        int batchSize, int numFeatures)
    {
        var output = new Tensor<T>([batchSize, numFeatures, _attentionDim]);
        var attentionScores = new Tensor<T>([batchSize, numFeatures, numFeatures]);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));

        for (int b = 0; b < batchSize; b++)
        {
            // Compute attention scores [numFeatures, numFeatures]
            var scores = new T[numFeatures, numFeatures];

            for (int i = 0; i < numFeatures; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    var dot = NumOps.Zero;
                    for (int d = 0; d < _attentionDim; d++)
                    {
                        dot = NumOps.Add(dot, NumOps.Multiply(
                            queries[b * numFeatures * _attentionDim + i * _attentionDim + d],
                            keys[b * numFeatures * _attentionDim + j * _attentionDim + d]));
                    }
                    scores[i, j] = NumOps.Multiply(dot, scale);
                }
            }

            // Softmax over features (column dimension)
            for (int i = 0; i < numFeatures; i++)
            {
                var maxScore = scores[i, 0];
                for (int j = 1; j < numFeatures; j++)
                {
                    if (NumOps.Compare(scores[i, j], maxScore) > 0)
                        maxScore = scores[i, j];
                }

                var sumExp = NumOps.Zero;
                for (int j = 0; j < numFeatures; j++)
                {
                    scores[i, j] = NumOps.Exp(NumOps.Subtract(scores[i, j], maxScore));
                    sumExp = NumOps.Add(sumExp, scores[i, j]);
                }

                for (int j = 0; j < numFeatures; j++)
                {
                    scores[i, j] = NumOps.Divide(scores[i, j], sumExp);
                    attentionScores[b * numFeatures * numFeatures + i * numFeatures + j] = scores[i, j];
                }
            }

            // Apply attention to values
            for (int i = 0; i < numFeatures; i++)
            {
                for (int d = 0; d < _attentionDim; d++)
                {
                    var sum = NumOps.Zero;
                    for (int j = 0; j < numFeatures; j++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(
                            scores[i, j],
                            values[b * numFeatures * _attentionDim + j * _attentionDim + d]));
                    }
                    output[b * numFeatures * _attentionDim + i * _attentionDim + d] = sum;
                }
            }
        }

        _attentionScoresCache = attentionScores;
        return output;
    }

    private Tensor<T> ProjectOutput(Tensor<T> attended, int batchSize, int numFeatures)
    {
        var projected = new Tensor<T>([batchSize, numFeatures, _embeddingDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                for (int o = 0; o < _embeddingDim; o++)
                {
                    var sum = NumOps.Zero;
                    for (int i = 0; i < _attentionDim; i++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(
                            attended[b * numFeatures * _attentionDim + f * _attentionDim + i],
                            _outputWeights[i * _embeddingDim + o]));
                    }
                    projected[b * numFeatures * _embeddingDim + f * _embeddingDim + o] = sum;
                }
            }
        }

        return projected;
    }

    private Tensor<T> AddResidual(Tensor<T> input, Tensor<T> output,
        int batchSize, int numFeatures, int embDim)
    {
        var result = new Tensor<T>(output.Shape);

        if (_residualWeights != null)
        {
            // Project residual if dimensions differ
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < numFeatures; f++)
                {
                    for (int o = 0; o < embDim; o++)
                    {
                        var residual = NumOps.Zero;
                        for (int i = 0; i < embDim; i++)
                        {
                            residual = NumOps.Add(residual, NumOps.Multiply(
                                input[b * numFeatures * embDim + f * embDim + i],
                                _residualWeights[i * embDim + o]));
                        }
                        result[b * numFeatures * embDim + f * embDim + o] = NumOps.Add(
                            output[b * numFeatures * embDim + f * embDim + o],
                            residual);
                    }
                }
            }
        }
        else
        {
            // Direct addition
            for (int i = 0; i < output.Length; i++)
            {
                result[i] = NumOps.Add(output[i], input[i]);
            }
        }

        return result;
    }

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Length; i++)
        {
            output[i] = NumOps.Compare(input[i], NumOps.Zero) > 0 ? input[i] : NumOps.Zero;
        }

        return output;
    }

    /// <summary>
    /// Backward pass through the interacting layer.
    /// </summary>
    /// <param name="gradient">Gradient from upstream.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        if (_inputCache == null
            || _queriesCache == null
            || _keysCache == null
            || _valuesCache == null
            || _attentionScoresCache == null
            || _attendedCache == null
            || _preActivationCache == null)
        {
            throw new InvalidOperationException("Forward must be called before backward");
        }

        int batchSize = _inputCache.Shape[0];
        int numFeatures = _inputCache.Shape[1];
        int embDim = _inputCache.Shape[2];

        var inputGrad = new Tensor<T>([batchSize, numFeatures, embDim]);

        // Reset gradients
        for (int i = 0; i < _queryWeightsGrad.Length; i++)
            _queryWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _keyWeightsGrad.Length; i++)
            _keyWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _valueWeightsGrad.Length; i++)
            _valueWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _outputWeightsGrad.Length; i++)
            _outputWeightsGrad[i] = NumOps.Zero;
        if (_residualWeightsGrad != null)
        {
            for (int i = 0; i < _residualWeightsGrad.Length; i++)
            {
                _residualWeightsGrad[i] = NumOps.Zero;
            }
        }

        // ReLU backward
        var gradPreActivation = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradPreActivation[i] = NumOps.Compare(_preActivationCache[i], NumOps.Zero) > 0
                ? gradient[i]
                : NumOps.Zero;
        }

        // Residual backward
        if (_useResidual)
        {
            if (_residualWeights != null && _residualWeightsGrad != null)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    for (int f = 0; f < numFeatures; f++)
                    {
                        for (int o = 0; o < embDim; o++)
                        {
                            int outIdx = b * numFeatures * embDim + f * embDim + o;
                            var gradVal = gradPreActivation[outIdx];
                            for (int i = 0; i < embDim; i++)
                            {
                                int wIdx = i * embDim + o;
                                int inIdx = b * numFeatures * embDim + f * embDim + i;
                                _residualWeightsGrad[wIdx] = NumOps.Add(
                                    _residualWeightsGrad[wIdx],
                                    NumOps.Multiply(_inputCache[inIdx], gradVal));
                            }
                        }
                    }
                }

                for (int b = 0; b < batchSize; b++)
                {
                    for (int f = 0; f < numFeatures; f++)
                    {
                        for (int i = 0; i < embDim; i++)
                        {
                            var sum = NumOps.Zero;
                            for (int o = 0; o < embDim; o++)
                            {
                                int outIdx = b * numFeatures * embDim + f * embDim + o;
                                int wIdx = i * embDim + o;
                                sum = NumOps.Add(sum, NumOps.Multiply(
                                    gradPreActivation[outIdx],
                                    _residualWeights[wIdx]));
                            }
                            int inIdx = b * numFeatures * embDim + f * embDim + i;
                            inputGrad[inIdx] = NumOps.Add(inputGrad[inIdx], sum);
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < inputGrad.Length; i++)
                {
                    inputGrad[i] = NumOps.Add(inputGrad[i], gradPreActivation[i]);
                }
            }
        }

        // Output projection backward
        var gradAttended = new Tensor<T>([batchSize, numFeatures, _attentionDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                for (int a = 0; a < _attentionDim; a++)
                {
                    var sum = NumOps.Zero;
                    int attIdx = b * numFeatures * _attentionDim + f * _attentionDim + a;
                    for (int e = 0; e < embDim; e++)
                    {
                        int outIdx = b * numFeatures * embDim + f * embDim + e;
                        int wIdx = a * embDim + e;
                        var gradOut = gradPreActivation[outIdx];
                        _outputWeightsGrad[wIdx] = NumOps.Add(
                            _outputWeightsGrad[wIdx],
                            NumOps.Multiply(_attendedCache[attIdx], gradOut));
                        sum = NumOps.Add(sum, NumOps.Multiply(gradOut, _outputWeights[wIdx]));
                    }
                    gradAttended[attIdx] = sum;
                }
            }
        }

        // Attention backward
        var gradValues = new Tensor<T>([batchSize, numFeatures, _attentionDim]);
        var gradScores = new Tensor<T>([batchSize, numFeatures, numFeatures]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numFeatures; i++)
            {
                var rowDot = new T[numFeatures];
                for (int j = 0; j < numFeatures; j++)
                {
                    var sum = NumOps.Zero;
                    for (int d = 0; d < _attentionDim; d++)
                    {
                        int attIdx = b * numFeatures * _attentionDim + i * _attentionDim + d;
                        int valIdx = b * numFeatures * _attentionDim + j * _attentionDim + d;
                        sum = NumOps.Add(sum, NumOps.Multiply(gradAttended[attIdx], _valuesCache[valIdx]));
                    }
                    rowDot[j] = sum;
                }

                var rowSum = NumOps.Zero;
                for (int j = 0; j < numFeatures; j++)
                {
                    int scoreIdx = b * numFeatures * numFeatures + i * numFeatures + j;
                    rowSum = NumOps.Add(rowSum, NumOps.Multiply(rowDot[j], _attentionScoresCache[scoreIdx]));
                }

                for (int j = 0; j < numFeatures; j++)
                {
                    int scoreIdx = b * numFeatures * numFeatures + i * numFeatures + j;
                    gradScores[scoreIdx] = NumOps.Multiply(
                        _attentionScoresCache[scoreIdx],
                        NumOps.Subtract(rowDot[j], rowSum));

                    for (int d = 0; d < _attentionDim; d++)
                    {
                        int attIdx = b * numFeatures * _attentionDim + i * _attentionDim + d;
                        int valIdx = b * numFeatures * _attentionDim + j * _attentionDim + d;
                        gradValues[valIdx] = NumOps.Add(
                            gradValues[valIdx],
                            NumOps.Multiply(_attentionScoresCache[scoreIdx], gradAttended[attIdx]));
                    }
                }
            }
        }

        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));
        var gradQueries = new Tensor<T>([batchSize, numFeatures, _attentionDim]);
        var gradKeys = new Tensor<T>([batchSize, numFeatures, _attentionDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numFeatures; i++)
            {
                for (int d = 0; d < _attentionDim; d++)
                {
                    var sum = NumOps.Zero;
                    for (int j = 0; j < numFeatures; j++)
                    {
                        int scoreIdx = b * numFeatures * numFeatures + i * numFeatures + j;
                        int keyIdx = b * numFeatures * _attentionDim + j * _attentionDim + d;
                        sum = NumOps.Add(sum, NumOps.Multiply(gradScores[scoreIdx], _keysCache[keyIdx]));
                    }
                    int qIdx = b * numFeatures * _attentionDim + i * _attentionDim + d;
                    gradQueries[qIdx] = NumOps.Multiply(sum, scale);
                }
            }

            for (int j = 0; j < numFeatures; j++)
            {
                for (int d = 0; d < _attentionDim; d++)
                {
                    var sum = NumOps.Zero;
                    for (int i = 0; i < numFeatures; i++)
                    {
                        int scoreIdx = b * numFeatures * numFeatures + i * numFeatures + j;
                        int queryIdx = b * numFeatures * _attentionDim + i * _attentionDim + d;
                        sum = NumOps.Add(sum, NumOps.Multiply(gradScores[scoreIdx], _queriesCache[queryIdx]));
                    }
                    int kIdx = b * numFeatures * _attentionDim + j * _attentionDim + d;
                    gradKeys[kIdx] = NumOps.Multiply(sum, scale);
                }
            }
        }

        // Parameter gradients for projections
        for (int i = 0; i < embDim; i++)
        {
            for (int a = 0; a < _attentionDim; a++)
            {
                var sumQ = NumOps.Zero;
                var sumK = NumOps.Zero;
                var sumV = NumOps.Zero;

                for (int b = 0; b < batchSize; b++)
                {
                    for (int f = 0; f < numFeatures; f++)
                    {
                        int inIdx = b * numFeatures * embDim + f * embDim + i;
                        int attIdx = b * numFeatures * _attentionDim + f * _attentionDim + a;
                        var inputVal = _inputCache[inIdx];
                        sumQ = NumOps.Add(sumQ, NumOps.Multiply(inputVal, gradQueries[attIdx]));
                        sumK = NumOps.Add(sumK, NumOps.Multiply(inputVal, gradKeys[attIdx]));
                        sumV = NumOps.Add(sumV, NumOps.Multiply(inputVal, gradValues[attIdx]));
                    }
                }

                int wIdx = i * _attentionDim + a;
                _queryWeightsGrad[wIdx] = sumQ;
                _keyWeightsGrad[wIdx] = sumK;
                _valueWeightsGrad[wIdx] = sumV;
            }
        }

        // Input gradients from Q, K, V projections
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                for (int i = 0; i < embDim; i++)
                {
                    var sum = NumOps.Zero;
                    for (int a = 0; a < _attentionDim; a++)
                    {
                        int wIdx = i * _attentionDim + a;
                        int attIdx = b * numFeatures * _attentionDim + f * _attentionDim + a;
                        sum = NumOps.Add(sum, NumOps.Multiply(gradQueries[attIdx], _queryWeights[wIdx]));
                        sum = NumOps.Add(sum, NumOps.Multiply(gradKeys[attIdx], _keyWeights[wIdx]));
                        sum = NumOps.Add(sum, NumOps.Multiply(gradValues[attIdx], _valueWeights[wIdx]));
                    }
                    int inIdx = b * numFeatures * embDim + f * embDim + i;
                    inputGrad[inIdx] = NumOps.Add(inputGrad[inIdx], sum);
                }
            }
        }

        return inputGrad;
    }

    /// <summary>
    /// Gets attention scores for interpretability.
    /// </summary>
    public Tensor<T>? GetAttentionScores() => _attentionScoresCache;

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < _queryWeights.Length; i++)
        {
            _queryWeights[i] = NumOps.Subtract(_queryWeights[i],
                NumOps.Multiply(learningRate, _queryWeightsGrad[i]));
        }

        for (int i = 0; i < _keyWeights.Length; i++)
        {
            _keyWeights[i] = NumOps.Subtract(_keyWeights[i],
                NumOps.Multiply(learningRate, _keyWeightsGrad[i]));
        }

        for (int i = 0; i < _valueWeights.Length; i++)
        {
            _valueWeights[i] = NumOps.Subtract(_valueWeights[i],
                NumOps.Multiply(learningRate, _valueWeightsGrad[i]));
        }

        for (int i = 0; i < _outputWeights.Length; i++)
        {
            _outputWeights[i] = NumOps.Subtract(_outputWeights[i],
                NumOps.Multiply(learningRate, _outputWeightsGrad[i]));
        }

        if (_residualWeights != null && _residualWeightsGrad != null)
        {
            for (int i = 0; i < _residualWeights.Length; i++)
            {
                _residualWeights[i] = NumOps.Subtract(_residualWeights[i],
                    NumOps.Multiply(learningRate, _residualWeightsGrad[i]));
            }
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public void ResetState()
    {
        _inputCache = null;
        _queriesCache = null;
        _keysCache = null;
        _valuesCache = null;
        _attentionScoresCache = null;
        _attendedCache = null;
        _preActivationCache = null;

        for (int i = 0; i < _queryWeightsGrad.Length; i++)
            _queryWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _keyWeightsGrad.Length; i++)
            _keyWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _valueWeightsGrad.Length; i++)
            _valueWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _outputWeightsGrad.Length; i++)
            _outputWeightsGrad[i] = NumOps.Zero;
        if (_residualWeightsGrad != null)
        {
            for (int i = 0; i < _residualWeightsGrad.Length; i++)
            {
                _residualWeightsGrad[i] = NumOps.Zero;
            }
        }
    }
}
