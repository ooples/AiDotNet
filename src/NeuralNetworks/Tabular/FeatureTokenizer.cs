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
public class FeatureTokenizer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numNumericalFeatures;
    private readonly int _numCategoricalFeatures;
    private readonly int _embeddingDimension;
    private readonly int[]? _categoricalCardinalities;
    private readonly bool _useNumericalBias;

    // Numerical feature embeddings (linear projection)
    private Tensor<T> _numericalWeights;  // Shape: [numNumerical, embeddingDim]
    private Tensor<T>? _numericalBias;     // Shape: [numNumerical, embeddingDim]

    // Categorical feature embeddings (lookup tables)
    private readonly List<Tensor<T>> _categoricalEmbeddings;  // Each: [cardinality, embeddingDim]

    // [CLS] token embedding
    private Tensor<T> _clsToken;  // Shape: [1, embeddingDim]

    // Gradients
    private Tensor<T>? _numericalWeightsGrad;
    private Tensor<T>? _numericalBiasGrad;
    private readonly List<Tensor<T>?> _categoricalEmbeddingsGrad;
    private Tensor<T>? _clsTokenGrad;

    // Cache for backward pass
    private Tensor<T>? _inputCache;
    private Matrix<int>? _categoricalIndicesCache;

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
    public int ParameterCount
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

        for (int i = 0; i < tensor.Length; i++)
        {
            // Box-Muller transform for normal distribution
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            tensor[i] = _numOps.FromDouble(normal * stdDev);
        }
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

        _inputCache = numericalFeatures;
        _categoricalIndicesCache = categoricalIndices;

        // Output: [batch, seq_len, embed_dim] where seq_len = 1 (CLS) + numFeatures
        int seqLen = SequenceLength;
        var output = new Tensor<T>([batchSize, seqLen, _embeddingDimension]);

        for (int b = 0; b < batchSize; b++)
        {
            // Position 0: [CLS] token
            for (int d = 0; d < _embeddingDimension; d++)
            {
                output[b * seqLen * _embeddingDimension + 0 * _embeddingDimension + d] = _clsToken[d];
            }

            // Positions 1 to numNumerical: numerical feature embeddings
            for (int f = 0; f < _numNumericalFeatures; f++)
            {
                var featureValue = numericalFeatures[b * numNumerical + f];
                int outputPos = 1 + f;

                for (int d = 0; d < _embeddingDimension; d++)
                {
                    int weightIdx = f * _embeddingDimension + d;
                    var embedding = _numOps.Multiply(featureValue, _numericalWeights[weightIdx]);
                    if (_numericalBias != null)
                    {
                        embedding = _numOps.Add(embedding, _numericalBias[weightIdx]);
                    }
                    output[b * seqLen * _embeddingDimension + outputPos * _embeddingDimension + d] = embedding;
                }
            }

            // Positions numNumerical+1 to end: categorical feature embeddings
            if (categoricalIndices != null && _numCategoricalFeatures > 0)
            {
                for (int f = 0; f < _numCategoricalFeatures; f++)
                {
                    int catIndex = categoricalIndices[b, f];
                    int outputPos = 1 + _numNumericalFeatures + f;
                    var embTable = _categoricalEmbeddings[f];

                    for (int d = 0; d < _embeddingDimension; d++)
                    {
                        output[b * seqLen * _embeddingDimension + outputPos * _embeddingDimension + d] =
                            embTable[catIndex * _embeddingDimension + d];
                    }
                }
            }
        }

        return output;
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
    /// Performs the backward pass to compute gradients.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer [batch_size, seq_len, embed_dim].</param>
    /// <returns>Gradient with respect to numerical input [batch_size, num_numerical].</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_inputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = outputGradient.Shape[0];
        int seqLen = outputGradient.Shape[1];
        var inputGrad = new Tensor<T>([batchSize, _numNumericalFeatures]);

        // Initialize gradients
        _numericalWeightsGrad = new Tensor<T>(_numericalWeights.Shape);
        _numericalWeightsGrad.Fill(_numOps.Zero);

        if (_numericalBias != null)
        {
            _numericalBiasGrad = new Tensor<T>(_numericalBias.Shape);
            _numericalBiasGrad.Fill(_numOps.Zero);
        }

        _clsTokenGrad = new Tensor<T>(_clsToken.Shape);
        _clsTokenGrad.Fill(_numOps.Zero);

        for (int i = 0; i < _categoricalEmbeddingsGrad.Count; i++)
        {
            _categoricalEmbeddingsGrad[i] = new Tensor<T>(_categoricalEmbeddings[i].Shape);
            _categoricalEmbeddingsGrad[i]!.Fill(_numOps.Zero);
        }

        for (int b = 0; b < batchSize; b++)
        {
            // [CLS] token gradient (position 0)
            for (int d = 0; d < _embeddingDimension; d++)
            {
                var grad = outputGradient[b * seqLen * _embeddingDimension + 0 * _embeddingDimension + d];
                _clsTokenGrad[d] = _numOps.Add(_clsTokenGrad[d], grad);
            }

            // Numerical feature gradients
            for (int f = 0; f < _numNumericalFeatures; f++)
            {
                int outputPos = 1 + f;
                var featureValue = _inputCache[b * _numNumericalFeatures + f];
                var inputGradAccum = _numOps.Zero;

                for (int d = 0; d < _embeddingDimension; d++)
                {
                    int gradIdx = b * seqLen * _embeddingDimension + outputPos * _embeddingDimension + d;
                    var grad = outputGradient[gradIdx];
                    int weightIdx = f * _embeddingDimension + d;

                    // Gradient w.r.t. weights: grad * input_value
                    _numericalWeightsGrad[weightIdx] = _numOps.Add(
                        _numericalWeightsGrad[weightIdx],
                        _numOps.Multiply(grad, featureValue));

                    // Gradient w.r.t. bias: grad
                    if (_numericalBiasGrad != null)
                    {
                        _numericalBiasGrad[weightIdx] = _numOps.Add(_numericalBiasGrad[weightIdx], grad);
                    }

                    // Gradient w.r.t. input: grad * weight
                    inputGradAccum = _numOps.Add(inputGradAccum,
                        _numOps.Multiply(grad, _numericalWeights[weightIdx]));
                }

                inputGrad[b * _numNumericalFeatures + f] = inputGradAccum;
            }

            // Categorical embedding gradients
            if (_categoricalIndicesCache != null && _numCategoricalFeatures > 0)
            {
                for (int f = 0; f < _numCategoricalFeatures; f++)
                {
                    int catIndex = _categoricalIndicesCache[b, f];
                    int outputPos = 1 + _numNumericalFeatures + f;
                    var embGrad = _categoricalEmbeddingsGrad[f]!;

                    for (int d = 0; d < _embeddingDimension; d++)
                    {
                        var grad = outputGradient[b * seqLen * _embeddingDimension + outputPos * _embeddingDimension + d];
                        embGrad[catIndex * _embeddingDimension + d] = _numOps.Add(
                            embGrad[catIndex * _embeddingDimension + d], grad);
                    }
                }
            }
        }

        return inputGrad;
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var paramsList = new List<T>();

        // Numerical weights
        for (int i = 0; i < _numericalWeights.Length; i++)
        {
            paramsList.Add(_numericalWeights[i]);
        }

        // Numerical bias
        if (_numericalBias != null)
        {
            for (int i = 0; i < _numericalBias.Length; i++)
            {
                paramsList.Add(_numericalBias[i]);
            }
        }

        // CLS token
        for (int i = 0; i < _clsToken.Length; i++)
        {
            paramsList.Add(_clsToken[i]);
        }

        // Categorical embeddings
        foreach (var emb in _categoricalEmbeddings)
        {
            for (int i = 0; i < emb.Length; i++)
            {
                paramsList.Add(emb[i]);
            }
        }

        return new Vector<T>([.. paramsList]);
    }

    /// <summary>
    /// Sets the trainable parameters from a vector.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;

        // Numerical weights
        for (int i = 0; i < _numericalWeights.Length; i++)
        {
            _numericalWeights[i] = parameters[idx++];
        }

        // Numerical bias
        if (_numericalBias != null)
        {
            for (int i = 0; i < _numericalBias.Length; i++)
            {
                _numericalBias[i] = parameters[idx++];
            }
        }

        // CLS token
        for (int i = 0; i < _clsToken.Length; i++)
        {
            _clsToken[i] = parameters[idx++];
        }

        // Categorical embeddings
        foreach (var emb in _categoricalEmbeddings)
        {
            for (int i = 0; i < emb.Length; i++)
            {
                emb[i] = parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Gets the parameter gradients as a single vector.
    /// </summary>
    public Vector<T> GetParameterGradients()
    {
        var gradsList = new List<T>();

        if (_numericalWeightsGrad != null)
        {
            for (int i = 0; i < _numericalWeightsGrad.Length; i++)
            {
                gradsList.Add(_numericalWeightsGrad[i]);
            }
        }
        else
        {
            for (int i = 0; i < _numericalWeights.Length; i++)
            {
                gradsList.Add(_numOps.Zero);
            }
        }

        if (_numericalBias != null)
        {
            if (_numericalBiasGrad != null)
            {
                for (int i = 0; i < _numericalBiasGrad.Length; i++)
                {
                    gradsList.Add(_numericalBiasGrad[i]);
                }
            }
            else
            {
                for (int i = 0; i < _numericalBias.Length; i++)
                {
                    gradsList.Add(_numOps.Zero);
                }
            }
        }

        if (_clsTokenGrad != null)
        {
            for (int i = 0; i < _clsTokenGrad.Length; i++)
            {
                gradsList.Add(_clsTokenGrad[i]);
            }
        }
        else
        {
            for (int i = 0; i < _clsToken.Length; i++)
            {
                gradsList.Add(_numOps.Zero);
            }
        }

        for (int c = 0; c < _categoricalEmbeddings.Count; c++)
        {
            var embGrad = _categoricalEmbeddingsGrad[c];
            var emb = _categoricalEmbeddings[c];
            if (embGrad != null)
            {
                for (int i = 0; i < embGrad.Length; i++)
                {
                    gradsList.Add(embGrad[i]);
                }
            }
            else
            {
                for (int i = 0; i < emb.Length; i++)
                {
                    gradsList.Add(_numOps.Zero);
                }
            }
        }

        return new Vector<T>([.. gradsList]);
    }

    /// <summary>
    /// Updates parameters using the calculated gradients.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        if (_numericalWeightsGrad != null)
        {
            for (int i = 0; i < _numericalWeights.Length; i++)
            {
                _numericalWeights[i] = _numOps.Subtract(_numericalWeights[i],
                    _numOps.Multiply(learningRate, _numericalWeightsGrad[i]));
            }
        }

        if (_numericalBias != null && _numericalBiasGrad != null)
        {
            for (int i = 0; i < _numericalBias.Length; i++)
            {
                _numericalBias[i] = _numOps.Subtract(_numericalBias[i],
                    _numOps.Multiply(learningRate, _numericalBiasGrad[i]));
            }
        }

        if (_clsTokenGrad != null)
        {
            for (int i = 0; i < _clsToken.Length; i++)
            {
                _clsToken[i] = _numOps.Subtract(_clsToken[i],
                    _numOps.Multiply(learningRate, _clsTokenGrad[i]));
            }
        }

        for (int c = 0; c < _categoricalEmbeddings.Count; c++)
        {
            var embGrad = _categoricalEmbeddingsGrad[c];
            if (embGrad != null)
            {
                var emb = _categoricalEmbeddings[c];
                for (int i = 0; i < emb.Length; i++)
                {
                    emb[i] = _numOps.Subtract(emb[i], _numOps.Multiply(learningRate, embGrad[i]));
                }
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
