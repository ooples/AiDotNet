using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to dense vectors using GloVe-style word embeddings.
/// </summary>
/// <remarks>
/// <para>
/// GloVe (Global Vectors for Word Representation) learns word vectors by factorizing
/// the word-word co-occurrence matrix. Unlike Word2Vec which uses local context windows,
/// GloVe leverages global corpus statistics for potentially better embeddings.
/// </para>
/// <para>
/// The objective function is:
/// <code>
/// J = Σ f(Xij) * (wi · wj + bi + bj - log(Xij))²
/// </code>
/// Where Xij is the co-occurrence count and f is a weighting function.
/// </para>
/// <para><b>For Beginners:</b> GloVe combines the best of both worlds:
/// - Uses global word co-occurrence statistics (like LSA)
/// - Produces dense vectors (like Word2Vec)
/// - Often produces high-quality embeddings with less training time
/// - The resulting word relationships are mathematically meaningful
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class GloVeVectorizer<T> : TextVectorizerBase<T>
{
    private readonly int _vectorSize;
    private readonly int _windowSize;
    private readonly int _minCount;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly double _xMax;
    private readonly double _alpha;
    private readonly Word2VecAggregation _aggregation;
    private readonly int? _randomState;

    private Dictionary<string, double[]>? _wordVectors;
    private Dictionary<string, int>? _wordCounts;

    /// <summary>
    /// Gets the learned word vectors.
    /// </summary>
    public Dictionary<string, double[]>? WordVectors => _wordVectors;

    /// <summary>
    /// Gets the vector dimensionality.
    /// </summary>
    public int VectorSize => _vectorSize;

    /// <summary>
    /// Creates a new instance of <see cref="GloVeVectorizer{T}"/>.
    /// </summary>
    /// <param name="vectorSize">Dimensionality of word vectors. Defaults to 100.</param>
    /// <param name="windowSize">Context window size for co-occurrence. Defaults to 10.</param>
    /// <param name="minCount">Minimum word frequency to include. Defaults to 5.</param>
    /// <param name="epochs">Number of training epochs. Defaults to 25.</param>
    /// <param name="learningRate">Learning rate for AdaGrad. Defaults to 0.05.</param>
    /// <param name="xMax">Maximum co-occurrence count for weighting. Defaults to 100.</param>
    /// <param name="alpha">Power for weighting function. Defaults to 0.75.</param>
    /// <param name="aggregation">How to combine word vectors into document vectors. Defaults to Mean.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="randomState">Random seed for reproducibility. Null for random.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">Optional ITokenizer for subword tokenization.</param>
    public GloVeVectorizer(
        int vectorSize = 100,
        int windowSize = 10,
        int minCount = 5,
        int epochs = 25,
        double learningRate = 0.05,
        double xMax = 100,
        double alpha = 0.75,
        Word2VecAggregation aggregation = Word2VecAggregation.Mean,
        bool lowercase = true,
        int? randomState = null,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
        : base(minCount, 1.0, null, (1, 1), lowercase, tokenizer, stopWords, advancedTokenizer)
    {
        if (vectorSize < 1)
            throw new ArgumentException("Vector size must be at least 1.", nameof(vectorSize));

        _vectorSize = vectorSize;
        _windowSize = windowSize;
        _minCount = minCount;
        _epochs = epochs;
        _learningRate = learningRate;
        _xMax = xMax;
        _alpha = alpha;
        _aggregation = aggregation;
        _randomState = randomState;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _wordVectors is not null;

    /// <inheritdoc/>
    public override int FeatureCount => _vectorSize;

    /// <summary>
    /// Trains GloVe embeddings on the corpus.
    /// </summary>
    /// <param name="documents">The text documents to train on.</param>
    public override void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        _nDocs = docList.Count;

        // Tokenize all documents
        var allTokens = new List<List<string>>();
        _wordCounts = new Dictionary<string, int>();

        foreach (string doc in docList)
        {
            var tokens = Tokenize(doc).ToList();
            allTokens.Add(tokens);

            foreach (string token in tokens)
            {
                _wordCounts.TryGetValue(token, out int count);
                _wordCounts[token] = count + 1;
            }
        }

        // Filter by minimum count
        var vocab = _wordCounts
            .Where(kvp => kvp.Value >= _minCount)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        var vocabArray = vocab.Keys.ToArray();
        int vocabSize = vocabArray.Length;

        _vocabulary = vocabArray.Select((k, i) => (k, i)).ToDictionary(x => x.k, x => x.i);
        _featureNames = Enumerable.Range(0, _vectorSize).Select(i => $"glove_dim_{i}").ToArray();

        if (vocabSize == 0)
        {
            _wordVectors = new Dictionary<string, double[]>();
            return;
        }

        // Build co-occurrence matrix
        var cooccurrence = new Dictionary<(int, int), double>();

        foreach (var tokens in allTokens)
        {
            var filteredTokens = tokens.Where(t => _vocabulary.ContainsKey(t)).ToList();

            for (int i = 0; i < filteredTokens.Count; i++)
            {
                int wordI = _vocabulary[filteredTokens[i]];

                for (int j = Math.Max(0, i - _windowSize); j < i; j++)
                {
                    int wordJ = _vocabulary[filteredTokens[j]];
                    double distance = i - j;
                    double weight = 1.0 / distance; // Distance-weighted co-occurrence

                    var key1 = (wordI, wordJ);
                    var key2 = (wordJ, wordI);

                    cooccurrence.TryGetValue(key1, out double val1);
                    cooccurrence[key1] = val1 + weight;

                    cooccurrence.TryGetValue(key2, out double val2);
                    cooccurrence[key2] = val2 + weight;
                }
            }
        }

        // Initialize vectors and biases
        var random = _randomState.HasValue ? new Random(_randomState.Value) : new Random();
        var W = new double[vocabSize, _vectorSize]; // Main word vectors
        var W_tilde = new double[vocabSize, _vectorSize]; // Context word vectors
        var b = new double[vocabSize]; // Main biases
        var b_tilde = new double[vocabSize]; // Context biases

        // AdaGrad accumulators
        var gradSqW = new double[vocabSize, _vectorSize];
        var gradSqW_tilde = new double[vocabSize, _vectorSize];
        var gradSqB = new double[vocabSize];
        var gradSqB_tilde = new double[vocabSize];

        double initRange = 0.5 / _vectorSize;
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < _vectorSize; j++)
            {
                W[i, j] = (random.NextDouble() - 0.5) * initRange;
                W_tilde[i, j] = (random.NextDouble() - 0.5) * initRange;
                gradSqW[i, j] = 1; // Initialize to 1 to avoid divide by zero
                gradSqW_tilde[i, j] = 1;
            }
            b[i] = 0;
            b_tilde[i] = 0;
            gradSqB[i] = 1;
            gradSqB_tilde[i] = 1;
        }

        // Convert co-occurrence to list for iteration
        var cooccurrenceList = cooccurrence.ToList();

        // Training with AdaGrad
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            // Shuffle for SGD
            for (int i = cooccurrenceList.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (cooccurrenceList[i], cooccurrenceList[j]) = (cooccurrenceList[j], cooccurrenceList[i]);
            }

            foreach (var kvp in cooccurrenceList)
            {
                int wordI = kvp.Key.Item1;
                int wordJ = kvp.Key.Item2;
                double Xij = kvp.Value;

                if (Xij == 0) continue;

                // Weighting function
                double weight = Xij < _xMax ? Math.Pow(Xij / _xMax, _alpha) : 1.0;

                // Compute inner product
                double innerProduct = 0;
                for (int k = 0; k < _vectorSize; k++)
                {
                    innerProduct += W[wordI, k] * W_tilde[wordJ, k];
                }
                innerProduct += b[wordI] + b_tilde[wordJ];

                // Cost and gradient
                double diff = innerProduct - Math.Log(Xij);
                double fdiff = weight * diff;

                // Update with AdaGrad
                for (int k = 0; k < _vectorSize; k++)
                {
                    double gradW = fdiff * W_tilde[wordJ, k];
                    double gradW_tilde = fdiff * W[wordI, k];

                    gradSqW[wordI, k] += gradW * gradW;
                    gradSqW_tilde[wordJ, k] += gradW_tilde * gradW_tilde;

                    W[wordI, k] -= _learningRate * gradW / Math.Sqrt(gradSqW[wordI, k]);
                    W_tilde[wordJ, k] -= _learningRate * gradW_tilde / Math.Sqrt(gradSqW_tilde[wordJ, k]);
                }

                // Update biases
                gradSqB[wordI] += fdiff * fdiff;
                gradSqB_tilde[wordJ] += fdiff * fdiff;
                b[wordI] -= _learningRate * fdiff / Math.Sqrt(gradSqB[wordI]);
                b_tilde[wordJ] -= _learningRate * fdiff / Math.Sqrt(gradSqB_tilde[wordJ]);
            }
        }

        // Final word vectors are sum of W and W_tilde
        _wordVectors = new Dictionary<string, double[]>();
        for (int i = 0; i < vocabSize; i++)
        {
            var vector = new double[_vectorSize];
            for (int j = 0; j < _vectorSize; j++)
            {
                vector[j] = W[i, j] + W_tilde[i, j];
            }
            _wordVectors[vocabArray[i]] = vector;
        }
    }

    /// <summary>
    /// Transforms documents to dense vectors by aggregating word vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's dense vector representation.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_wordVectors is null)
        {
            throw new InvalidOperationException("GloVeVectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        var result = new double[nDocs, _vectorSize];

        for (int d = 0; d < nDocs; d++)
        {
            var tokens = Tokenize(docList[d]).ToList();
            var validTokens = tokens.Where(t => _wordVectors.ContainsKey(t)).ToList();

            if (validTokens.Count == 0) continue;

            if (_aggregation == Word2VecAggregation.Mean)
            {
                foreach (string token in validTokens)
                {
                    var vec = _wordVectors[token];
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        result[d, i] += vec[i];
                    }
                }
                for (int i = 0; i < _vectorSize; i++)
                {
                    result[d, i] /= validTokens.Count;
                }
            }
            else if (_aggregation == Word2VecAggregation.Sum)
            {
                foreach (string token in validTokens)
                {
                    var vec = _wordVectors[token];
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        result[d, i] += vec[i];
                    }
                }
            }
            else if (_aggregation == Word2VecAggregation.Max)
            {
                for (int i = 0; i < _vectorSize; i++)
                {
                    result[d, i] = double.MinValue;
                }
                foreach (string token in validTokens)
                {
                    var vec = _wordVectors[token];
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        result[d, i] = Math.Max(result[d, i], vec[i]);
                    }
                }
            }
            else if (_aggregation == Word2VecAggregation.TfidfWeighted && _wordCounts is not null)
            {
                double totalDocs = _nDocs;
                double totalWeight = 0;

                foreach (string token in validTokens)
                {
                    double idf = Math.Log(totalDocs / (_wordCounts.GetValueOrDefault(token, 1) + 1));
                    var vec = _wordVectors[token];
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        result[d, i] += vec[i] * idf;
                    }
                    totalWeight += idf;
                }

                if (totalWeight > 0)
                {
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        result[d, i] /= totalWeight;
                    }
                }
            }
        }

        // Convert to output type
        var output = new T[nDocs, _vectorSize];
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < _vectorSize; j++)
            {
                output[i, j] = NumOps.FromDouble(result[i, j]);
            }
        }

        return new Matrix<T>(output);
    }

    /// <summary>
    /// Gets the vector for a specific word.
    /// </summary>
    /// <param name="word">The word to get the vector for.</param>
    /// <returns>The word's vector, or null if not in vocabulary.</returns>
    public double[]? GetWordVector(string word)
    {
        if (_wordVectors is null) return null;
        return _wordVectors.GetValueOrDefault(_lowercase ? word.ToLowerInvariant() : word);
    }

    /// <summary>
    /// Finds the most similar words to a given word.
    /// </summary>
    /// <param name="word">The target word.</param>
    /// <param name="topN">Number of similar words to return.</param>
    /// <returns>List of (word, similarity) pairs.</returns>
    public List<(string Word, double Similarity)> MostSimilar(string word, int topN = 10)
    {
        if (_wordVectors is null)
            throw new InvalidOperationException("GloVeVectorizer has not been fitted.");

        var targetVector = GetWordVector(word);
        if (targetVector is null)
            return new List<(string, double)>();

        var similarities = new List<(string Word, double Similarity)>();

        foreach (var kvp in _wordVectors)
        {
            if (kvp.Key == word) continue;

            double dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < _vectorSize; i++)
            {
                dot += targetVector[i] * kvp.Value[i];
                normA += targetVector[i] * targetVector[i];
                normB += kvp.Value[i] * kvp.Value[i];
            }

            double similarity = dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-10);
            similarities.Add((kvp.Key, similarity));
        }

        return similarities.OrderByDescending(s => s.Similarity).Take(topN).ToList();
    }
}
