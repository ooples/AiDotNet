using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Implements the Feature Tokenizer that converts tabular features into embeddings for FT-Transformer.
/// </summary>
/// <remarks>
/// <para>
/// The Feature Tokenizer converts each input feature into a d-dimensional embedding:
/// - For numerical features: embedding = x * w + b (linear projection)
/// - For categorical features: embedding lookup from learned embedding table
/// A learnable [CLS] token is prepended to the sequence for classification/regression.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Feature Tokenizer is like a translator that converts your
/// spreadsheet columns into a format the Transformer can understand.
///
/// What it does:
/// 1. Takes each column value and converts it to a vector (list of numbers)
/// 2. For numbers (like age=25): Multiplies by learned weights and adds bias
/// 3. For categories (like color="red"): Looks up a learned embedding vector
/// 4. Adds a special [CLS] token that will aggregate all feature information
///
/// Why this matters:
/// - Transformers work on sequences of vectors, not raw numbers
/// - This conversion allows the model to learn rich representations of each feature
/// - The [CLS] token provides a single representation for the final prediction
///
/// Example: If you have 5 features with embedding dimension 64:
/// Input: [age, income, zip_code, gender, marital_status]
/// Output: Tensor of shape [batch, 6, 64] (5 features + 1 CLS token, each as 64-dim vector)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public partial class FeatureTokenizer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numNumericalFeatures;
    private readonly int _numCategoricalFeatures;
    private readonly int _embeddingDimension;
    private readonly int[]? _categoricalCardinalities;
    private readonly bool _useNumericalBias;

    // Numerical feature embeddings (linear projection)
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _numericalWeights;  // Shape: [numNumerical, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _numericalBias;     // Shape: [numNumerical, embeddingDim]

    // Categorical feature embeddings (lookup tables)
    private readonly List<Tensor<T>> _categoricalEmbeddings;  // Each: [cardinality, embeddingDim]

    // [CLS] token embedding
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _clsToken;  // Shape: [1, embeddingDim]

    // Gradients
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _numericalWeightsGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _numericalBiasGrad;
    private readonly List<Tensor<T>?> _categoricalEmbeddingsGrad;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _clsTokenGrad;

    /// <summary>
    /// Gets the total number of features (numerical + categorical).
    /// </summary>
    public int TotalFeatures => _numNumericalFeatures + _numCategoricalFeatures;

    /// <summary>
    /// Gets the sequence length including [CLS] token.
    /// </summary>
    public int SequenceLength => TotalFeatures + 1;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _embeddingDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public long ParameterCount
    {
        get
        {
            int count = _numericalWeights.Length;
            if (_numericalBias != null) count += _numericalBias.Length;
            count += _clsToken.Length;
            foreach (var emb in _categoricalEmbeddings)
            {
                count += emb.Length;
            }
            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the FeatureTokenizer class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical features.</param>
    /// <param name="embeddingDimension">Dimension of each feature embedding.</param>
    /// <param name="categoricalCardinalities">Array of cardinalities for categorical features (null if none).</param>
    /// <param name="useNumericalBias">Whether to use bias in numerical embeddings.</param>
    /// <param name="initScale">Initialization scale for embeddings.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creating a Feature Tokenizer:
    /// - numNumericalFeatures: How many number columns (age, income, etc.)
    /// - embeddingDimension: How rich each feature representation should be
    /// - categoricalCardinalities: For each category column, how many unique values
    /// - useNumericalBias: Usually true, adds learnable bias to number embeddings
    ///
    /// Example:
    /// If you have 3 numerical features and 2 categorical features:
    /// - categorical feature 1 has 5 unique values
    /// - categorical feature 2 has 10 unique values
    /// Then: categoricalCardinalities = [5, 10]
    /// </para>
    /// </remarks>
    public FeatureTokenizer(
        int numNumericalFeatures,
        int embeddingDimension,
        int[]? categoricalCardinalities = null,
        bool useNumericalBias = true,
        double initScale = 0.01)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numNumericalFeatures = numNumericalFeatures;
        _embeddingDimension = embeddingDimension;
        _categoricalCardinalities = categoricalCardinalities;
        _numCategoricalFeatures = categoricalCardinalities?.Length ?? 0;
        _useNumericalBias = useNumericalBias;

        // Initialize numerical feature embeddings
        _numericalWeights = new Tensor<T>([_numNumericalFeatures, embeddingDimension]);
        InitializeXavier(_numericalWeights, initScale);

        if (useNumericalBias)
        {
            _numericalBias = new Tensor<T>([_numNumericalFeatures, embeddingDimension]);
            _numericalBias.Fill(_numOps.Zero);
        }

        // Initialize categorical embeddings
        _categoricalEmbeddings = new List<Tensor<T>>();
        _categoricalEmbeddingsGrad = new List<Tensor<T>?>();
        if (categoricalCardinalities != null)
        {
            foreach (int cardinality in categoricalCardinalities)
            {
                var embedding = new Tensor<T>([cardinality, embeddingDimension]);
                InitializeXavier(embedding, initScale);
                _categoricalEmbeddings.Add(embedding);
                _categoricalEmbeddingsGrad.Add(null);
            }
        }

        // Initialize [CLS] token
        _clsToken = new Tensor<T>([1, embeddingDimension]);
        InitializeXavier(_clsToken, initScale);
    }

    /// <summary>
    /// Initializes a tensor with Xavier/Glorot initialization.
    /// </summary>
    private void InitializeXavier(Tensor<T> tensor, double scale)
    {
        var random = RandomHelper.CreateSecureRandom();
        int fanIn = tensor.Shape.Length > 1 ? tensor.Shape[0] : 1;
        int fanOut = tensor.Shape.Length > 1 ? tensor.Shape[1] : tensor.Shape[0];
        double stdDev = scale * Math.Sqrt(2.0 / (fanIn + fanOut));

        // N(0, stdDev^2) for every element in one engine call (unseeded secure source as before).
        AiDotNetEngine.Current.TensorNormalInPlace(tensor, 0.0, stdDev, random.Next());
    }

    /// <summary>
    /// Performs the forward pass to tokenize features into embeddings.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices matrix [batch_size, num_categorical] or null.</param>
    /// <returns>Embedded features tensor [batch_size, sequence_length, embedding_dim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts your raw data into embeddings:
    ///
    /// 1. For each numerical feature value x:
    ///    embedding[i] = x * weight[i] + bias[i]
    ///
    /// 2. For each categorical feature index:
    ///    embedding[i] = look up the row from the embedding table
    ///
    /// 3. Prepend the [CLS] token to the sequence
    ///
    /// The output is a sequence of embeddings that the Transformer can process.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        int numNumerical = numericalFeatures.Shape[1];

        if (numNumerical != _numNumericalFeatures)
        {
            throw new ArgumentException($"Expected {_numNumericalFeatures} numerical features, got {numNumerical}");
        }

        // Output: [batch, seq_len, embed_dim] where seq_len = 1 (CLS) + numFeatures. Built from
        // engine ops so the tokenizer's weights are on the gradient tape: the FT-Transformer
        // training step registers them via GetTrainableTensors and expects gradients for them.
        var eng = AiDotNetEngine.Current;
        int embDim = _embeddingDimension;

        // Position 0: [CLS] token, shared by every row.
        var parts = new List<Tensor<T>>
        {
            eng.TensorBroadcastTo(eng.Reshape(_clsToken, new[] { 1, 1, embDim }), new[] { batchSize, 1, embDim })
        };

        // Positions 1..numNumerical: x[b, f] * W[f, :] (+ bias[f, :]).
        var numericalShape = new[] { batchSize, _numNumericalFeatures, embDim };
        var numerical = eng.TensorMultiply(
            eng.TensorBroadcastTo(eng.Reshape(numericalFeatures, new[] { batchSize, _numNumericalFeatures, 1 }), numericalShape),
            eng.TensorBroadcastTo(eng.Reshape(_numericalWeights, new[] { 1, _numNumericalFeatures, embDim }), numericalShape));
        if (_numericalBias != null)
        {
            numerical = eng.TensorAdd(numerical,
                eng.TensorBroadcastTo(eng.Reshape(_numericalBias, new[] { 1, _numNumericalFeatures, embDim }), numericalShape));
        }

        parts.Add(numerical);

        // Remaining positions: categorical embedding rows looked up by index (zeros when no indices are given).
        if (_numCategoricalFeatures > 0)
        {
            if (categoricalIndices != null)
            {
                var indexTable = Tensor<int>.FromMatrix(categoricalIndices);
                for (int f = 0; f < _numCategoricalFeatures; f++)
                {
                    var columnIndices = eng.Reshape(
                        eng.TensorSlice(indexTable, new[] { 0, f }, new[] { batchSize, 1 }), new[] { batchSize });
                    var rows = eng.TensorEmbeddingLookup(_categoricalEmbeddings[f], columnIndices);
                    parts.Add(eng.Reshape(rows, new[] { batchSize, 1, embDim }));
                }
            }
            else
            {
                parts.Add(new Tensor<T>(new[] { batchSize, _numCategoricalFeatures, embDim }));
            }
        }

        return eng.TensorConcatenate(parts.ToArray(), 1);
    }

    /// <summary>
    /// Performs the forward pass with numerical features only.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <returns>Embedded features tensor [batch_size, sequence_length, embedding_dim].</returns>
    public Tensor<T> Forward(Tensor<T> numericalFeatures)
    {
        return Forward(numericalFeatures, null);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var eng = AiDotNetEngine.Current;

        // Order: numerical weights, numerical bias (if any), CLS token, categorical embeddings.
        var parts = new List<Tensor<T>> { eng.Reshape(_numericalWeights, new[] { _numericalWeights.Length }) };
        if (_numericalBias != null)
        {
            parts.Add(eng.Reshape(_numericalBias, new[] { _numericalBias.Length }));
        }

        parts.Add(eng.Reshape(_clsToken, new[] { _clsToken.Length }));
        foreach (var emb in _categoricalEmbeddings)
        {
            parts.Add(eng.Reshape(emb, new[] { emb.Length }));
        }

        return eng.TensorConcatenate(parts.ToArray(), 0).ToVector();
    }

    /// <summary>
    /// Returns the tokenizer's trainable weight TENSORS (the live field instances,
    /// not a flat copy) so a caller can register them with a <c>GradientTape</c> and
    /// apply gradients in place. Order matches <see cref="GetParameters"/>:
    /// numerical weights, numerical bias (if present), CLS token, then each
    /// categorical embedding. Used by the FT-Transformer tape-based training step —
    /// the tokenizer keeps its own manual gradient fields for the legacy path, but
    /// tape training updates these tensors directly.
    /// </summary>
    internal IReadOnlyList<Tensor<T>> GetTrainableTensors()
    {
        var tensors = new List<Tensor<T>> { _numericalWeights };
        if (_numericalBias is not null) tensors.Add(_numericalBias);
        tensors.Add(_clsToken);
        tensors.AddRange(_categoricalEmbeddings);
        return tensors;
    }

    /// <summary>
    /// Sets the trainable parameters from a vector.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        var eng = AiDotNetEngine.Current;
        int idx = 0;

        // Copies the next block of the flat vector into the live tensor (in place, so tensors
        // handed out by GetTrainableTensors stay the ones being updated).
        void Load(Tensor<T> target)
        {
            eng.TensorCopy(Tensor<T>.FromVector(parameters.GetSubVector(idx, target.Length)), target);
            idx += target.Length;
        }

        Load(_numericalWeights);
        if (_numericalBias != null)
        {
            Load(_numericalBias);
        }

        Load(_clsToken);
        foreach (var emb in _categoricalEmbeddings)
        {
            Load(emb);
        }
    }

    /// <summary>
    /// Gets the parameter gradients as a single vector.
    /// </summary>
    public Vector<T> GetParameterGradients()
    {
        var eng = AiDotNetEngine.Current;

        // Each block contributes its gradient when one exists, zeros of the parameter's size otherwise.
        Tensor<T> Flat(Tensor<T>? grad, Tensor<T> param) =>
            grad != null ? eng.Reshape(grad, new[] { grad.Length }) : new Tensor<T>(new[] { param.Length });

        var parts = new List<Tensor<T>> { Flat(_numericalWeightsGrad, _numericalWeights) };
        if (_numericalBias != null)
        {
            parts.Add(Flat(_numericalBiasGrad, _numericalBias));
        }

        parts.Add(Flat(_clsTokenGrad, _clsToken));
        for (int c = 0; c < _categoricalEmbeddings.Count; c++)
        {
            parts.Add(Flat(_categoricalEmbeddingsGrad[c], _categoricalEmbeddings[c]));
        }

        return eng.TensorConcatenate(parts.ToArray(), 0).ToVector();
    }

    /// <summary>
    /// Updates parameters using the calculated gradients.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        var eng = AiDotNetEngine.Current;

        if (_numericalWeightsGrad != null)
        {
            _numericalWeights = eng.TensorSubtract(_numericalWeights,
                eng.TensorMultiplyScalar(_numericalWeightsGrad, learningRate));
        }

        if (_numericalBias != null && _numericalBiasGrad != null)
        {
            _numericalBias = eng.TensorSubtract(_numericalBias,
                eng.TensorMultiplyScalar(_numericalBiasGrad, learningRate));
        }

        if (_clsTokenGrad != null)
        {
            _clsToken = eng.TensorSubtract(_clsToken,
                eng.TensorMultiplyScalar(_clsTokenGrad, learningRate));
        }

        for (int c = 0; c < _categoricalEmbeddings.Count; c++)
        {
            var embGrad = _categoricalEmbeddingsGrad[c];
            if (embGrad != null)
            {
                _categoricalEmbeddings[c] = eng.TensorSubtract(_categoricalEmbeddings[c],
                    eng.TensorMultiplyScalar(embGrad, learningRate));
            }
        }
    }

    /// <summary>
    /// Resets all gradients to zero.
    /// </summary>
    public void ResetGradients()
    {
        _numericalWeightsGrad = null;
        _numericalBiasGrad = null;
        _clsTokenGrad = null;
        for (int i = 0; i < _categoricalEmbeddingsGrad.Count; i++)
        {
            _categoricalEmbeddingsGrad[i] = null;
        }
    }

    /// <summary>
    /// Gets the [CLS] token embedding.
    /// </summary>
    public Tensor<T> GetClsToken() => _clsToken;

    /// <summary>
    /// Gets the numerical feature weights.
    /// </summary>
    public Tensor<T> GetNumericalWeights() => _numericalWeights;

    /// <summary>
    /// Gets the numerical feature biases.
    /// </summary>
    public Tensor<T>? GetNumericalBias() => _numericalBias;
}
