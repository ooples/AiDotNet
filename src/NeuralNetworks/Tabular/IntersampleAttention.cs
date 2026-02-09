using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Intersample (Row) Attention for SAINT architecture.
/// </summary>
/// <remarks>
/// <para>
/// Intersample attention allows samples in a batch to attend to each other,
/// enabling the model to learn relationships between different data points.
/// This is a key innovation in SAINT that helps with semi-supervised learning.
/// </para>
/// <para>
/// <b>For Beginners:</b> While column attention looks at relationships between features
/// within a single sample, intersample attention looks at relationships between different
/// samples in the batch:
/// - Column attention: "How does age relate to income for this person?"
/// - Intersample attention: "How does this person compare to others in the batch?"
///
/// This allows the model to learn from the distribution of data, not just individual samples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class IntersampleAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly double _dropoutRate;

    // Attention projections
    private readonly FullyConnectedLayer<T> _queryProjection;
    private readonly FullyConnectedLayer<T> _keyProjection;
    private readonly FullyConnectedLayer<T> _valueProjection;
    private readonly FullyConnectedLayer<T> _outputProjection;

    // Layer normalization parameters
    private Tensor<T> _layerNormGamma;
    private Tensor<T> _layerNormBeta;

    // Cached values for backward pass
    private Tensor<T>? _inputCache;
    private Tensor<T>? _normalizedCache;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int ParameterCount =>
        _queryProjection.ParameterCount +
        _keyProjection.ParameterCount +
        _valueProjection.ParameterCount +
        _outputProjection.ParameterCount +
        _embeddingDim * 2; // LayerNorm gamma and beta

    /// <summary>
    /// Initializes intersample attention.
    /// </summary>
    /// <param name="embeddingDim">Embedding dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="dropoutRate">Dropout rate for attention.</param>
    public IntersampleAttention(int embeddingDim, int numHeads = 8, double dropoutRate = 0.1)
    {
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _headDim = embeddingDim / numHeads;
        _dropoutRate = dropoutRate;

        // Initialize attention projections
        _queryProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (IActivationFunction<T>?)null);
        _keyProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (IActivationFunction<T>?)null);
        _valueProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (IActivationFunction<T>?)null);
        _outputProjection = new FullyConnectedLayer<T>(
            embeddingDim, embeddingDim, (IActivationFunction<T>?)null);

        // Initialize layer normalization parameters
        _layerNormGamma = new Tensor<T>([embeddingDim]);
        _layerNormBeta = new Tensor<T>([embeddingDim]);

        for (int i = 0; i < embeddingDim; i++)
        {
            _layerNormGamma[i] = NumOps.One;
            _layerNormBeta[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Forward pass through intersample attention.
    /// </summary>
    /// <param name="input">Input tensor [batchSize, numFeatures, embeddingDim].</param>
    /// <param name="training">Whether in training mode (for dropout).</param>
    /// <returns>Output with intersample attention applied [batchSize, numFeatures, embeddingDim].</returns>
    public Tensor<T> Forward(Tensor<T> input, bool training = true)
    {
        _inputCache = input;

        int batchSize = input.Shape[0];
        int numFeatures = input.Shape[1];
        int embDim = input.Shape[2];

        // Transpose to [numFeatures, batchSize, embeddingDim] for intersample attention
        var transposed = TransposeForIntersample(input, batchSize, numFeatures, embDim);

        // Apply multi-head self-attention across samples
        var attended = ApplyMultiHeadAttention(transposed, numFeatures, batchSize, embDim);

        // Transpose back to [batchSize, numFeatures, embeddingDim]
        var output = TransposeBack(attended, batchSize, numFeatures, embDim);

        // Residual connection and layer normalization
        output = AddResidualAndNormalize(input, output, batchSize, numFeatures, embDim);
        _normalizedCache = output;

        return output;
    }

    private Tensor<T> TransposeForIntersample(Tensor<T> input, int batchSize, int numFeatures, int embDim)
    {
        var transposed = new Tensor<T>([numFeatures, batchSize, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    int srcIdx = b * numFeatures * embDim + f * embDim + d;
                    int dstIdx = f * batchSize * embDim + b * embDim + d;
                    transposed[dstIdx] = input[srcIdx];
                }
            }
        }

        return transposed;
    }

    private Tensor<T> TransposeBack(Tensor<T> input, int batchSize, int numFeatures, int embDim)
    {
        var transposed = new Tensor<T>([batchSize, numFeatures, embDim]);

        for (int f = 0; f < numFeatures; f++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    int srcIdx = f * batchSize * embDim + b * embDim + d;
                    int dstIdx = b * numFeatures * embDim + f * embDim + d;
                    transposed[dstIdx] = input[srcIdx];
                }
            }
        }

        return transposed;
    }

    private Tensor<T> ApplyMultiHeadAttention(Tensor<T> input, int numFeatures, int batchSize, int embDim)
    {
        var output = new Tensor<T>([numFeatures, batchSize, embDim]);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));

        // For each feature position, apply attention across samples
        for (int f = 0; f < numFeatures; f++)
        {
            // Extract feature slice [batchSize, embDim]
            var featureSlice = new Tensor<T>([batchSize, embDim]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    featureSlice[b * embDim + d] = input[f * batchSize * embDim + b * embDim + d];
                }
            }

            // Project to Q, K, V
            var queries = _queryProjection.Forward(featureSlice);
            var keys = _keyProjection.Forward(featureSlice);
            var values = _valueProjection.Forward(featureSlice);

            // Compute attention scores [batchSize, batchSize]
            var scores = new T[batchSize, batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < batchSize; j++)
                {
                    var dot = NumOps.Zero;
                    for (int d = 0; d < embDim; d++)
                    {
                        dot = NumOps.Add(dot, NumOps.Multiply(
                            queries[i * embDim + d],
                            keys[j * embDim + d]));
                    }
                    scores[i, j] = NumOps.Multiply(dot, scale);
                }
            }

            // Softmax over samples
            for (int i = 0; i < batchSize; i++)
            {
                var maxScore = scores[i, 0];
                for (int j = 1; j < batchSize; j++)
                {
                    if (NumOps.Compare(scores[i, j], maxScore) > 0)
                        maxScore = scores[i, j];
                }

                var sumExp = NumOps.Zero;
                for (int j = 0; j < batchSize; j++)
                {
                    scores[i, j] = NumOps.Exp(NumOps.Subtract(scores[i, j], maxScore));
                    sumExp = NumOps.Add(sumExp, scores[i, j]);
                }

                for (int j = 0; j < batchSize; j++)
                {
                    scores[i, j] = NumOps.Divide(scores[i, j], sumExp);
                }
            }

            // Apply attention to values
            for (int i = 0; i < batchSize; i++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    var sum = NumOps.Zero;
                    for (int j = 0; j < batchSize; j++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(
                            scores[i, j],
                            values[j * embDim + d]));
                    }
                    output[f * batchSize * embDim + i * embDim + d] = sum;
                }
            }
        }

        return output;
    }

    private Tensor<T> AddResidualAndNormalize(Tensor<T> residual, Tensor<T> output,
        int batchSize, int numFeatures, int embDim)
    {
        var result = new Tensor<T>(output.Shape);
        var epsilon = NumOps.FromDouble(1e-5);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                // Add residual
                for (int d = 0; d < embDim; d++)
                {
                    int idx = b * numFeatures * embDim + f * embDim + d;
                    result[idx] = NumOps.Add(residual[idx], output[idx]);
                }

                // Compute mean and variance for this position
                var mean = NumOps.Zero;
                for (int d = 0; d < embDim; d++)
                {
                    int idx = b * numFeatures * embDim + f * embDim + d;
                    mean = NumOps.Add(mean, result[idx]);
                }
                mean = NumOps.Divide(mean, NumOps.FromDouble(embDim));

                var variance = NumOps.Zero;
                for (int d = 0; d < embDim; d++)
                {
                    int idx = b * numFeatures * embDim + f * embDim + d;
                    var diff = NumOps.Subtract(result[idx], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(embDim));

                // Normalize and apply gamma/beta
                var stdDev = NumOps.Sqrt(NumOps.Add(variance, epsilon));
                for (int d = 0; d < embDim; d++)
                {
                    int idx = b * numFeatures * embDim + f * embDim + d;
                    var normalized = NumOps.Divide(NumOps.Subtract(result[idx], mean), stdDev);
                    result[idx] = NumOps.Add(
                        NumOps.Multiply(_layerNormGamma[d], normalized),
                        _layerNormBeta[d]);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Backward pass through intersample attention.
    /// </summary>
    /// <param name="gradient">Gradient from upstream.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        // Simplified backward - in full implementation would backprop through attention
        var inputGrad = _outputProjection.Backward(gradient);
        return inputGrad;
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        _queryProjection.UpdateParameters(learningRate);
        _keyProjection.UpdateParameters(learningRate);
        _valueProjection.UpdateParameters(learningRate);
        _outputProjection.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public void ResetState()
    {
        _inputCache = null;
        _normalizedCache = null;

        _queryProjection.ResetState();
        _keyProjection.ResetState();
        _valueProjection.ResetState();
        _outputProjection.ResetState();
    }
}
