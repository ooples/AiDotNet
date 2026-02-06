using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Column (positional) embedding for tabular transformers like TabTransformer.
/// </summary>
/// <remarks>
/// <para>
/// Column embeddings provide position information to the transformer, telling it
/// which column/feature each embedding came from. This is analogous to positional
/// encodings in NLP transformers.
/// </para>
/// <para>
/// <b>For Beginners:</b> Column embeddings help the model understand:
/// - "This is feature #1 (age)"
/// - "This is feature #2 (income)"
///
/// Without column embeddings, the model wouldn't know which feature is which
/// after the attention layers mix them together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ColumnEmbedding<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    private readonly int _numColumns;
    private readonly int _embeddingDim;
    private readonly bool _learnable;

    private Tensor<T> _embeddings;
    private Tensor<T> _embeddingGradients;

    /// <summary>
    /// Gets the number of columns.
    /// </summary>
    public int NumColumns => _numColumns;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDim => _embeddingDim;

    /// <summary>
    /// Gets the number of parameters (0 if using sinusoidal).
    /// </summary>
    public int ParameterCount => _learnable ? _numColumns * _embeddingDim : 0;

    /// <summary>
    /// Initializes column embeddings.
    /// </summary>
    /// <param name="numColumns">Number of columns/features.</param>
    /// <param name="embeddingDim">Embedding dimension.</param>
    /// <param name="learnable">Whether embeddings are learnable (true) or sinusoidal (false).</param>
    /// <param name="initScale">Initialization scale for learnable embeddings.</param>
    public ColumnEmbedding(int numColumns, int embeddingDim, bool learnable = true, double initScale = 0.02)
    {
        _numColumns = numColumns;
        _embeddingDim = embeddingDim;
        _learnable = learnable;
        _random = RandomHelper.CreateSecureRandom();

        _embeddings = new Tensor<T>([numColumns, embeddingDim]);
        _embeddingGradients = new Tensor<T>([numColumns, embeddingDim]);

        if (learnable)
        {
            InitializeLearnableEmbeddings(initScale);
        }
        else
        {
            InitializeSinusoidalEmbeddings();
        }
    }

    private void InitializeLearnableEmbeddings(double scale)
    {
        for (int i = 0; i < _embeddings.Length; i++)
        {
            _embeddings[i] = NumOps.FromDouble(_random.NextGaussian() * scale);
        }
    }

    private void InitializeSinusoidalEmbeddings()
    {
        for (int col = 0; col < _numColumns; col++)
        {
            for (int d = 0; d < _embeddingDim; d++)
            {
                double angle = col / Math.Pow(10000, (2.0 * (d / 2)) / _embeddingDim);
                double value = d % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
                _embeddings[col * _embeddingDim + d] = NumOps.FromDouble(value);
            }
        }
    }

    /// <summary>
    /// Adds column embeddings to feature embeddings.
    /// </summary>
    /// <param name="featureEmbeddings">Feature embeddings [batchSize, numColumns, embeddingDim].</param>
    /// <returns>Embeddings with column information added.</returns>
    public Tensor<T> AddColumnEmbeddings(Tensor<T> featureEmbeddings)
    {
        int batchSize = featureEmbeddings.Shape[0];
        int numCols = featureEmbeddings.Shape[1];
        int embDim = featureEmbeddings.Shape[2];

        if (numCols > _numColumns)
        {
            throw new ArgumentException($"Input has {numCols} columns but only {_numColumns} column embeddings");
        }

        if (embDim != _embeddingDim)
        {
            throw new ArgumentException(
                $"Input embedding dimension ({embDim}) does not match column embedding dimension ({_embeddingDim}).");
        }

        var output = new Tensor<T>(featureEmbeddings.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numCols; c++)
            {
                for (int d = 0; d < embDim; d++)
                {
                    int inputIdx = b * numCols * embDim + c * embDim + d;
                    int embIdx = c * _embeddingDim + d;
                    output[inputIdx] = NumOps.Add(featureEmbeddings[inputIdx], _embeddings[embIdx]);
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Gets the column embedding for a specific column.
    /// </summary>
    /// <param name="columnIndex">Column index.</param>
    /// <returns>Column embedding [embeddingDim].</returns>
    public T[] GetColumnEmbedding(int columnIndex)
    {
        if (columnIndex < 0 || columnIndex >= _numColumns)
        {
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        }

        var embedding = new T[_embeddingDim];
        for (int d = 0; d < _embeddingDim; d++)
        {
            embedding[d] = _embeddings[columnIndex * _embeddingDim + d];
        }
        return embedding;
    }

    /// <summary>
    /// Computes gradients for column embeddings.
    /// </summary>
    /// <param name="gradient">Gradient from upstream [batchSize, numColumns, embeddingDim].</param>
    public void Backward(Tensor<T> gradient)
    {
        if (!_learnable) return;

        int batchSize = gradient.Shape[0];
        int numCols = gradient.Shape[1];
        int embDim = gradient.Shape[2];

        if (embDim != _embeddingDim)
        {
            throw new ArgumentException(
                $"Gradient embedding dimension ({embDim}) does not match column embedding dimension ({_embeddingDim}).");
        }

        // Accumulate gradients from all batch samples
        for (int c = 0; c < numCols; c++)
        {
            for (int d = 0; d < embDim; d++)
            {
                var gradSum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    int gradIdx = b * numCols * embDim + c * embDim + d;
                    gradSum = NumOps.Add(gradSum, gradient[gradIdx]);
                }
                _embeddingGradients[c * _embeddingDim + d] = gradSum;
            }
        }
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        if (!_learnable) return;

        for (int i = 0; i < _embeddings.Length; i++)
        {
            _embeddings[i] = NumOps.Subtract(_embeddings[i],
                NumOps.Multiply(learningRate, _embeddingGradients[i]));
        }
    }

    /// <summary>
    /// Resets gradients.
    /// </summary>
    public void ResetGradients()
    {
        for (int i = 0; i < _embeddingGradients.Length; i++)
        {
            _embeddingGradients[i] = NumOps.Zero;
        }
    }
}
