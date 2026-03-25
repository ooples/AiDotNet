using AiDotNet.Autodiff;
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
public class InteractingLayer<T> : LayerBase<T>
{
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

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public override int ParameterCount =>
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
        : base([embeddingDim], [embeddingDim])
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _attentionDim = attentionDim ?? embeddingDim;
        _headDim = _attentionDim / numHeads;
        _useResidual = useResidual;

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
            _queryWeights[i] = NumOps.FromDouble(Random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _keyWeights.Length; i++)
        {
            _keyWeights[i] = NumOps.FromDouble(Random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _valueWeights.Length; i++)
        {
            _valueWeights[i] = NumOps.FromDouble(Random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _outputWeights.Length; i++)
        {
            _outputWeights[i] = NumOps.FromDouble(Random.NextGaussian() * outputScale);
        }

        if (_residualWeights != null)
        {
            for (int i = 0; i < _residualWeights.Length; i++)
            {
                _residualWeights[i] = NumOps.FromDouble(Random.NextGaussian() * scale);
            }
        }
    }

    /// <summary>
    /// Forward pass through the interacting layer.
    /// </summary>
    /// <param name="input">Input embeddings [batchSize, numFeatures, embeddingDim].</param>
    /// <returns>Feature interactions [batchSize, numFeatures, embeddingDim].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _inputCache = input;

        int batchSize = input.Shape[0];
        int numFeatures = input.Shape[1];
        int embDim = input.Shape[2];

        // Project to queries, keys, values using Engine.TensorMatMul
        var queries = ProjectInput(input, _queryWeights, batchSize, numFeatures, embDim, _attentionDim);
        var keys = ProjectInput(input, _keyWeights, batchSize, numFeatures, embDim, _attentionDim);
        var values = ProjectInput(input, _valueWeights, batchSize, numFeatures, embDim, _attentionDim);

        _queriesCache = queries;
        _keysCache = keys;
        _valuesCache = values;

        // Multi-head self-attention using Engine ops
        var attended = MultiHeadAttention(queries, keys, values, batchSize, numFeatures);
        _attendedCache = attended;

        // Output projection using Engine.TensorMatMul
        var output = ProjectOutput(attended, batchSize, numFeatures);

        // Residual connection
        if (_useResidual)
        {
            output = AddResidual(input, output, batchSize, numFeatures, embDim);
        }

        _preActivationCache = output;

        // Apply ReLU activation using Engine
        output = Engine.ReLU(output);

        return output;
    }

    private Tensor<T> ProjectInput(Tensor<T> input, Tensor<T> weights,
        int batchSize, int numFeatures, int inputDim, int outputDim)
    {
        // Reshape [batchSize, numFeatures, inputDim] -> [batchSize * numFeatures, inputDim]
        var flat = input.Reshape(batchSize * numFeatures, inputDim);
        // MatMul: [batchSize * numFeatures, inputDim] x [inputDim, outputDim] -> [batchSize * numFeatures, outputDim]
        var projected = Engine.TensorMatMul(flat, weights);
        // Reshape back to [batchSize, numFeatures, outputDim]
        return projected.Reshape(batchSize, numFeatures, outputDim);
    }

    private Tensor<T> MultiHeadAttention(Tensor<T> queries, Tensor<T> keys, Tensor<T> values,
        int batchSize, int numFeatures)
    {
        var output = new Tensor<T>([batchSize, numFeatures, _attentionDim]);
        var attentionScores = new Tensor<T>([batchSize, numFeatures, numFeatures]);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));

        for (int b = 0; b < batchSize; b++)
        {
            // Extract batch slices [numFeatures, attentionDim]
            var qSlice = ExtractBatchSlice(queries, b, numFeatures, _attentionDim);
            var kSlice = ExtractBatchSlice(keys, b, numFeatures, _attentionDim);
            var vSlice = ExtractBatchSlice(values, b, numFeatures, _attentionDim);

            // Q * K^T -> [numFeatures, numFeatures]
            var kT = kSlice.Transpose(new[] { 1, 0 });
            var scores = Engine.TensorMatMul(qSlice, kT);
            scores = Engine.TensorMultiplyScalar(scores, scale);

            // Softmax over features
            scores = Engine.Softmax(scores);

            // Store attention scores
            int scoreOffset = b * numFeatures * numFeatures;
            for (int i = 0; i < numFeatures * numFeatures; i++)
            {
                attentionScores[scoreOffset + i] = scores[i];
            }

            // Apply attention: scores * V -> [numFeatures, attentionDim]
            var attended = Engine.TensorMatMul(scores, vSlice);

            // Store into output
            int outOffset = b * numFeatures * _attentionDim;
            for (int i = 0; i < numFeatures * _attentionDim; i++)
            {
                output[outOffset + i] = attended[i];
            }
        }

        _attentionScoresCache = attentionScores;
        return output;
    }

    private Tensor<T> ExtractBatchSlice(Tensor<T> tensor, int batchIdx, int rows, int cols)
    {
        var slice = new Tensor<T>([rows, cols]);
        int offset = batchIdx * rows * cols;
        for (int i = 0; i < rows * cols; i++)
        {
            slice[i] = tensor[offset + i];
        }
        return slice;
    }

    private Tensor<T> ProjectOutput(Tensor<T> attended, int batchSize, int numFeatures)
    {
        // Reshape [batchSize, numFeatures, attentionDim] -> [batchSize * numFeatures, attentionDim]
        var flat = attended.Reshape(batchSize * numFeatures, _attentionDim);
        // MatMul: [batchSize * numFeatures, attentionDim] x [attentionDim, embeddingDim]
        var projected = Engine.TensorMatMul(flat, _outputWeights);
        // Reshape back to [batchSize, numFeatures, embeddingDim]
        return projected.Reshape(batchSize, numFeatures, _embeddingDim);
    }

    private Tensor<T> AddResidual(Tensor<T> input, Tensor<T> output,
        int batchSize, int numFeatures, int embDim)
    {
        if (_residualWeights != null)
        {
            // Project residual if dimensions differ
            var flat = input.Reshape(batchSize * numFeatures, embDim);
            var projected = Engine.TensorMatMul(flat, _residualWeights);
            var residual = projected.Reshape(batchSize, numFeatures, embDim);
            return Engine.TensorAdd(output, residual);
        }

        // Direct addition
        return Engine.TensorAdd(output, input);
    }

    /// <summary>
    /// Backward pass through the interacting layer.
    /// </summary>
    /// <param name="gradient">Gradient from upstream.</param>
    /// <returns>Gradient with respect to input.</returns>
    public override Tensor<T> Backward(Tensor<T> gradient)
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
        Engine.TensorFill(_queryWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_keyWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_valueWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_outputWeightsGrad, NumOps.Zero);
        if (_residualWeightsGrad != null)
        {
            Engine.TensorFill(_residualWeightsGrad, NumOps.Zero);
        }

        // ReLU backward
        var gradPreActivation = new Tensor<T>(gradient.Shape.ToArray());
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
                inputGrad = Engine.TensorAdd(inputGrad, gradPreActivation);
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
    /// Updates parameters using gradient descent.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        // w = w - lr * grad  =>  Engine.TensorSubtract(w, Engine.TensorMultiplyScalar(grad, lr))
        _queryWeights = Engine.TensorSubtract(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGrad, learningRate));
        _keyWeights = Engine.TensorSubtract(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGrad, learningRate));
        _valueWeights = Engine.TensorSubtract(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGrad, learningRate));
        _outputWeights = Engine.TensorSubtract(_outputWeights, Engine.TensorMultiplyScalar(_outputWeightsGrad, learningRate));

        if (_residualWeights != null && _residualWeightsGrad != null)
        {
            _residualWeights = Engine.TensorSubtract(_residualWeights, Engine.TensorMultiplyScalar(_residualWeightsGrad, learningRate));
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int total = ParameterCount;
        var result = new Vector<T>(total);
        int offset = 0;

        CopyTensorToVector(_queryWeights, result, ref offset);
        CopyTensorToVector(_keyWeights, result, ref offset);
        CopyTensorToVector(_valueWeights, result, ref offset);
        CopyTensorToVector(_outputWeights, result, ref offset);
        if (_residualWeights != null)
        {
            CopyTensorToVector(_residualWeights, result, ref offset);
        }

        return result;
    }

    private static void CopyTensorToVector(Tensor<T> tensor, Vector<T> vector, ref int offset)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            vector[offset++] = tensor[i];
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        _inputCache = null;
        _queriesCache = null;
        _keysCache = null;
        _valuesCache = null;
        _attentionScoresCache = null;
        _attendedCache = null;
        _preActivationCache = null;

        Engine.TensorFill(_queryWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_keyWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_valueWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_outputWeightsGrad, NumOps.Zero);
        if (_residualWeightsGrad != null)
        {
            Engine.TensorFill(_residualWeightsGrad, NumOps.Zero);
        }
    }

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Export Q, K, V weight projections
        var qWeightsNode = TensorOperations<T>.Constant(_queryWeights, "queryWeights");
        var kWeightsNode = TensorOperations<T>.Constant(_keyWeights, "keyWeights");
        var vWeightsNode = TensorOperations<T>.Constant(_valueWeights, "valueWeights");
        var oWeightsNode = TensorOperations<T>.Constant(_outputWeights, "outputWeights");

        var queryNode = TensorOperations<T>.MatrixMultiply(inputNode, qWeightsNode);
        var keyNode = TensorOperations<T>.MatrixMultiply(inputNode, kWeightsNode);
        var valueNode = TensorOperations<T>.MatrixMultiply(inputNode, vWeightsNode);

        // Attention: softmax(Q * K^T) * V
        var attentionScores = TensorOperations<T>.MatrixMultiply(queryNode, TensorOperations<T>.Transpose(keyNode));
        var attentionWeights = TensorOperations<T>.Softmax(attentionScores);
        var attended = TensorOperations<T>.MatrixMultiply(attentionWeights, valueNode);

        return TensorOperations<T>.MatrixMultiply(attended, oWeightsNode);
    }
}
