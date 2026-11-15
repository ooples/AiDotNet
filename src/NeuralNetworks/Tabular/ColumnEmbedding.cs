using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public partial class ColumnEmbedding<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private IEngine Engine => AiDotNetEngine.Current;
    private readonly Random _random;

    private readonly int _numColumns;
    private readonly int _embeddingDim;
    private readonly bool _learnable;

    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _embeddings;
    [AiDotNet.Attributes.TrainableParameter]
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
    public long ParameterCount => _learnable ? _numColumns * _embeddingDim : 0;

    /// <summary>
    /// Initializes column embeddings.
    /// </summary>
    /// <param name="numColumns">Number of columns/features.</param>
    /// <param name="embeddingDim">Embedding dimension.</param>
    /// <param name="learnable">Whether embeddings are learnable (true) or sinusoidal (false).</param>
    /// <param name="initScale">Initialization scale for learnable embeddings.</param>
    public ColumnEmbedding(int numColumns, int embeddingDim, bool learnable = true, double initScale = 0.02)
    {
        if (numColumns <= 0)
            throw new ArgumentOutOfRangeException(nameof(numColumns), "Number of columns must be positive.");
        if (embeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

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
        // N(0, scale^2) for every slot in one engine call, seeded from this instance's Random.
        Engine.TensorNormalInPlace(_embeddings, 0.0, scale, _random.Next());
    }

    private void InitializeSinusoidalEmbeddings()
    {
        // angle[col, d] = col / 10000^(2 * floor(d / 2) / D); even d -> sin, odd d -> cos.
        var two = NumOps.FromDouble(2.0);
        var colIdx = Engine.TensorArange<T>(0, _numColumns, 1);
        var dimIdx = Engine.TensorArange<T>(0, _embeddingDim, 1);
        var pairIdx = Engine.TensorFloor(Engine.TensorDivideScalar(dimIdx, two));
        var invFreq = Engine.TensorExp(Engine.TensorMultiplyScalar(
            pairIdx, NumOps.FromDouble(-2.0 * Math.Log(10000.0) / _embeddingDim)));
        var angle = Engine.TensorOuterProduct(colIdx, invFreq);
        var oddMask = Engine.TensorSubtract(dimIdx, Engine.TensorMultiplyScalar(pairIdx, two));
        var evenMask = Engine.TensorAddScalar(Engine.TensorNegate(oddMask), NumOps.One);
        _embeddings = Engine.TensorAdd(
            Engine.TensorMultiply(Engine.TensorSin(angle), evenMask),
            Engine.TensorMultiply(Engine.TensorCos(angle), oddMask));
    }

    /// <summary>
    /// Adds column embeddings to feature embeddings.
    /// </summary>
    /// <param name="featureEmbeddings">Feature embeddings [batchSize, numColumns, embeddingDim].</param>
    /// <returns>Embeddings with column information added.</returns>
    public Tensor<T> AddColumnEmbeddings(Tensor<T> featureEmbeddings)
    {
        if (featureEmbeddings.Shape.Length < 3)
            throw new ArgumentException(
                $"Feature embeddings must be at least rank-3 [..., numColumns, embeddingDim], but got rank {featureEmbeddings.Shape.Length}.");

        int numCols = featureEmbeddings.Shape[^2];
        int embDim = featureEmbeddings.Shape[^1];

        if (numCols > _numColumns)
        {
            throw new ArgumentException($"Input has {numCols} columns but only {_numColumns} column embeddings");
        }

        if (embDim != _embeddingDim)
        {
            throw new ArgumentException(
                $"Input embedding dimension ({embDim}) does not match column embedding dimension ({_embeddingDim}).");
        }

        // Compute total batch elements (all dims except last two)
        int totalBatchElements = 1;
        for (int i = 0; i < featureEmbeddings.Shape.Length - 2; i++)
        {
            totalBatchElements *= featureEmbeddings.Shape[i];
        }

        // Fresh, detached output (as before): every batch element gets the first numCols
        // column embeddings added.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var batched = Engine.Reshape(featureEmbeddings, new[] { totalBatchElements, numCols, embDim });
            var colEmb = Engine.Reshape(
                Engine.TensorSlice(_embeddings, new[] { 0, 0 }, new[] { numCols, embDim }),
                new[] { 1, numCols, embDim });
            var sum = Engine.TensorAdd(batched,
                Engine.TensorBroadcastTo(colEmb, new[] { totalBatchElements, numCols, embDim }));
            return Engine.Reshape(sum, featureEmbeddings._shape);
        }
    }

    /// <summary>
    /// Gets the column embedding for a specific column.
    /// </summary>
    /// <param name="columnIndex">Column index.</param>
    /// <returns>Column embedding [embeddingDim].</returns>
    public Vector<T> GetColumnEmbedding(int columnIndex)
    {
        if (columnIndex < 0 || columnIndex >= _numColumns)
        {
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        }

        return Engine.TensorSlice(_embeddings, new[] { columnIndex, 0 }, new[] { 1, _embeddingDim })
            .ToVector();
    }

    /// <summary>
    /// Updates embeddings via gradient descent using the gradients computed by <see cref="Backward"/>.
    /// </summary>
    /// <param name="learningRate">Step size for the gradient descent update.</param>
    public void UpdateParameters(T learningRate)
    {
        if (!_learnable) return;

        Engine.TensorSubtractInPlace(_embeddings,
            Engine.TensorMultiplyScalar(_embeddingGradients, learningRate));
    }

    /// <summary>
    /// Zeros all stored gradients. Call between training steps to prevent stale gradient values
    /// from affecting subsequent <see cref="UpdateParameters"/> calls.
    /// </summary>
    public void ResetGradients()
    {
        Engine.TensorFill(_embeddingGradients, NumOps.Zero);
    }
}
