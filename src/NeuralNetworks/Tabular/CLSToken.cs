using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// CLS (Classification) Token for transformer-based tabular models.
/// </summary>
/// <remarks>
/// <para>
/// The CLS token is a learnable embedding prepended to the input sequence.
/// After transformer processing, the CLS token's representation is used
/// as an aggregate representation of the entire input for classification/regression.
/// </para>
/// <para>
/// <b>For Beginners:</b> The CLS token serves as a "summary" position:
/// - It's added to the beginning of your features
/// - The transformer allows it to attend to all features
/// - After processing, the CLS token "knows about" all features
/// - We use its final representation for prediction
///
/// This is the same approach used in BERT for text classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CLSToken<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    private Tensor<T> _clsEmbedding;
    private Tensor<T> _clsGradient;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    public int ParameterCount => EmbeddingDimension;

    /// <summary>
    /// Initializes a new CLS token with the specified embedding dimension.
    /// </summary>
    /// <param name="embeddingDimension">The dimension of the CLS embedding.</param>
    /// <param name="initScale">Initialization scale for the embedding.</param>
    public CLSToken(int embeddingDimension, double initScale = 0.02)
    {
        EmbeddingDimension = embeddingDimension;
        _random = RandomHelper.CreateSecureRandom();

        _clsEmbedding = new Tensor<T>([1, embeddingDimension]);
        _clsGradient = new Tensor<T>([1, embeddingDimension]);

        InitializeEmbedding(initScale);
    }

    private void InitializeEmbedding(double scale)
    {
        for (int i = 0; i < EmbeddingDimension; i++)
        {
            _clsEmbedding[i] = NumOps.FromDouble(_random.NextGaussian() * scale);
        }
    }

    /// <summary>
    /// Prepends the CLS token to the input embeddings.
    /// </summary>
    /// <param name="embeddings">Input embeddings with shape [batchSize, seqLen, embDim].</param>
    /// <returns>Embeddings with CLS token prepended: [batchSize, seqLen+1, embDim].</returns>
    public Tensor<T> PrependCLS(Tensor<T> embeddings)
    {
        int batchSize = embeddings.Shape[0];
        int seqLen = embeddings.Shape[1];
        int inputEmbDim = embeddings.Shape[2];

        if (inputEmbDim != EmbeddingDimension)
        {
            throw new ArgumentException(
                $"Input embedding dimension ({inputEmbDim}) does not match CLS token dimension ({EmbeddingDimension}).");
        }

        int embDim = EmbeddingDimension;

        var output = new Tensor<T>([batchSize, seqLen + 1, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            // Add CLS token at position 0
            for (int d = 0; d < embDim; d++)
            {
                output[b * (seqLen + 1) * embDim + d] = _clsEmbedding[d];
            }

            // Copy original embeddings starting at position 1
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    int srcIdx = b * seqLen * embDim + s * embDim + d;
                    int dstIdx = b * (seqLen + 1) * embDim + (s + 1) * embDim + d;
                    output[dstIdx] = embeddings[srcIdx];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Extracts the CLS token representation from transformer output.
    /// </summary>
    /// <param name="transformerOutput">Output with shape [batchSize, seqLen+1, embDim].</param>
    /// <returns>CLS representations with shape [batchSize, embDim].</returns>
    public Tensor<T> ExtractCLS(Tensor<T> transformerOutput)
    {
        int batchSize = transformerOutput.Shape[0];
        int embDim = EmbeddingDimension;

        var clsOutput = new Tensor<T>([batchSize, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < embDim; d++)
            {
                // CLS is at position 0
                int srcIdx = b * transformerOutput.Shape[1] * embDim + d;
                clsOutput[b * embDim + d] = transformerOutput[srcIdx];
            }
        }

        return clsOutput;
    }

    /// <summary>
    /// Computes gradients for the CLS token from the backward pass.
    /// </summary>
    /// <param name="gradient">Gradient with respect to CLS output [batchSize, embDim].</param>
    public void Backward(Tensor<T> gradient)
    {
        int batchSize = gradient.Shape[0];
        int embDim = EmbeddingDimension;

        // Accumulate gradients from all batch samples
        for (int d = 0; d < embDim; d++)
        {
            var gradSum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                gradSum = NumOps.Add(gradSum, gradient[b * embDim + d]);
            }
            _clsGradient[d] = gradSum;
        }
    }

    /// <summary>
    /// Updates the CLS token embedding.
    /// </summary>
    /// <param name="learningRate">The learning rate.</param>
    public void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < EmbeddingDimension; i++)
        {
            _clsEmbedding[i] = NumOps.Subtract(
                _clsEmbedding[i],
                NumOps.Multiply(learningRate, _clsGradient[i]));
        }
    }

    /// <summary>
    /// Resets gradients to zero.
    /// </summary>
    public void ResetGradients()
    {
        for (int i = 0; i < EmbeddingDimension; i++)
        {
            _clsGradient[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Gets the current CLS embedding values.
    /// </summary>
    public Tensor<T> GetEmbedding() => _clsEmbedding;
}
