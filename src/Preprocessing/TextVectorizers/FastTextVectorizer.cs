using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to dense vectors using FastText-style subword embeddings.
/// </summary>
/// <remarks>
/// <para>
/// FastText extends Word2Vec by representing each word as a bag of character n-grams.
/// This allows the model to:
/// - Handle out-of-vocabulary (OOV) words by composing subword vectors
/// - Capture morphological information (prefixes, suffixes, roots)
/// - Be more robust to spelling variations and typos
/// </para>
/// <para>
/// For example, "where" with n-grams (3,6) includes: "&lt;wh", "whe", "her", "ere", "re&gt;",
/// "&lt;whe", "wher", "here", "ere&gt;", "&lt;wher", "where", "here&gt;", "&lt;where", "where&gt;", "&lt;where&gt;"
/// </para>
/// <para><b>For Beginners:</b> FastText is Word2Vec that understands word parts:
/// - Can generate vectors for words it has never seen before
/// - "unhappiness" can be understood from "un-", "happy", "-ness" patterns
/// - Better for languages with rich morphology (German, Turkish, Finnish)
/// - Handles typos and spelling variations better than Word2Vec
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class FastTextVectorizer<T> : TextVectorizerBase<T>
{
    private readonly int _vectorSize;
    private readonly int _windowSize;
    private readonly int _minCount;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int _negativeSamples;
    private readonly (int Min, int Max) _subwordNGramRange;
    private readonly int _buckets;
    private readonly Word2VecAggregation _aggregation;
    private readonly int? _randomState;

    private Dictionary<string, double[]>? _wordVectors;
    private double[,]? _subwordVectors; // Hash bucket vectors
    private HashSet<string>? _knownWords;

    /// <summary>
    /// Gets the learned word vectors for known words.
    /// </summary>
    public Dictionary<string, double[]>? WordVectors => _wordVectors;

    /// <summary>
    /// Gets the vector dimensionality.
    /// </summary>
    public int VectorSize => _vectorSize;

    /// <summary>
    /// Creates a new instance of <see cref="FastTextVectorizer{T}"/>.
    /// </summary>
    /// <param name="vectorSize">Dimensionality of word vectors. Defaults to 100.</param>
    /// <param name="windowSize">Context window size. Defaults to 5.</param>
    /// <param name="minCount">Minimum word frequency to include. Defaults to 5.</param>
    /// <param name="epochs">Number of training epochs. Defaults to 5.</param>
    /// <param name="learningRate">Initial learning rate. Defaults to 0.05.</param>
    /// <param name="negativeSamples">Number of negative samples. Defaults to 5.</param>
    /// <param name="subwordNGramRange">Character n-gram range for subwords. Defaults to (3, 6).</param>
    /// <param name="buckets">Number of hash buckets for subwords. Defaults to 2000000.</param>
    /// <param name="aggregation">How to combine word vectors into document vectors. Defaults to Mean.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="randomState">Random seed for reproducibility. Null for random.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">Optional ITokenizer for subword tokenization.</param>
    public FastTextVectorizer(
        int vectorSize = 100,
        int windowSize = 5,
        int minCount = 5,
        int epochs = 5,
        double learningRate = 0.05,
        int negativeSamples = 5,
        (int Min, int Max)? subwordNGramRange = null,
        int buckets = 2000000,
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
        if (buckets < 1)
            throw new ArgumentOutOfRangeException(nameof(buckets), "Buckets must be at least 1.");

        _vectorSize = vectorSize;
        _windowSize = windowSize;
        _minCount = minCount;
        _epochs = epochs;
        _learningRate = learningRate;
        _negativeSamples = negativeSamples;
        _subwordNGramRange = subwordNGramRange ?? (3, 6);
        _buckets = buckets;
        if (aggregation == Word2VecAggregation.TfidfWeighted)
            throw new NotSupportedException("TfidfWeighted aggregation is not supported by FastTextVectorizer.");
        _aggregation = aggregation;
        _randomState = randomState;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _wordVectors is not null && _subwordVectors is not null;

    /// <inheritdoc/>
    public override int FeatureCount => _vectorSize;

    /// <summary>
    /// Gets the character n-grams for a word.
    /// </summary>
    private List<string> GetSubwords(string word)
    {
        var ngrams = new List<string>();
        string paddedWord = $"<{word}>";

        for (int n = _subwordNGramRange.Min; n <= _subwordNGramRange.Max; n++)
        {
            for (int i = 0; i <= paddedWord.Length - n; i++)
            {
                ngrams.Add(paddedWord.Substring(i, n));
            }
        }

        return ngrams;
    }

    /// <summary>
    /// Gets the hash bucket index for a subword.
    /// </summary>
    private int GetSubwordHash(string subword)
    {
        unchecked
        {
            uint hash = 2166136261;
            foreach (char c in subword)
            {
                hash ^= c;
                hash *= 16777619;
            }
            return (int)(hash % (uint)_buckets);
        }
    }

    /// <summary>
    /// Trains FastText embeddings on the corpus.
    /// </summary>
    /// <param name="documents">The text documents to train on.</param>
    public override void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        _nDocs = docList.Count;

        // Tokenize all documents
        var allTokens = new List<List<string>>();
        var wordCounts = new Dictionary<string, int>();

        foreach (string doc in docList)
        {
            var tokens = Tokenize(doc).ToList();
            allTokens.Add(tokens);

            foreach (string token in tokens)
            {
                wordCounts.TryGetValue(token, out int count);
                wordCounts[token] = count + 1;
            }
        }

        // Filter by minimum count
        var vocab = wordCounts
            .Where(kvp => kvp.Value >= _minCount)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        var vocabArray = vocab.Keys.ToArray();
        int vocabSize = vocabArray.Length;

        _vocabulary = vocabArray.Select((k, i) => (k, i)).ToDictionary(x => x.k, x => x.i);
        _featureNames = Enumerable.Range(0, _vectorSize).Select(i => $"fasttext_dim_{i}").ToArray();
        _knownWords = new HashSet<string>(vocabArray);

        if (vocabSize == 0)
        {
            _wordVectors = new Dictionary<string, double[]>();
            _subwordVectors = new double[_buckets, _vectorSize];
            return;
        }

        // Initialize vectors
        var random = _randomState.HasValue ? RandomHelper.CreateSeededRandom(_randomState.Value) : RandomHelper.CreateSecureRandom();
        _subwordVectors = new double[_buckets, _vectorSize];
        var outputVectors = new double[vocabSize, _vectorSize];

        double initRange = 0.5 / _vectorSize;
        for (int i = 0; i < _buckets; i++)
        {
            for (int j = 0; j < _vectorSize; j++)
            {
                _subwordVectors[i, j] = (random.NextDouble() - 0.5) * initRange;
            }
        }
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < _vectorSize; j++)
            {
                outputVectors[i, j] = 0;
            }
        }

        // Precompute subword hashes for each word
        var wordSubwords = new Dictionary<string, List<int>>();
        foreach (string word in vocabArray)
        {
            var subwords = GetSubwords(word);
            wordSubwords[word] = subwords.Select(GetSubwordHash).ToList();
        }

        // Build sampling table
        var samplingTable = BuildSamplingTable(vocab, vocabArray);

        // Training (Skip-gram style)
        double lr = _learningRate;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            foreach (var tokens in allTokens)
            {
                var filteredTokens = tokens.Where(t => _vocabulary.ContainsKey(t)).ToList();

                for (int pos = 0; pos < filteredTokens.Count; pos++)
                {
                    string targetWord = filteredTokens[pos];
                    int targetIdx = _vocabulary[targetWord];
                    var targetSubwords = wordSubwords[targetWord];

                    int windowStart = Math.Max(0, pos - _windowSize);
                    int windowEnd = Math.Min(filteredTokens.Count - 1, pos + _windowSize);

                    for (int ctxPos = windowStart; ctxPos <= windowEnd; ctxPos++)
                    {
                        if (ctxPos == pos) continue;

                        string contextWord = filteredTokens[ctxPos];
                        int contextIdx = _vocabulary[contextWord];

                        TrainSkipGram(targetSubwords, outputVectors, contextIdx,
                                     samplingTable, random, lr, vocabSize);
                    }
                }
            }

            lr = _learningRate * (1 - (double)(epoch + 1) / _epochs);
            lr = Math.Max(lr, _learningRate * 0.0001);
        }

        // Compute final word vectors from subword vectors
        _wordVectors = new Dictionary<string, double[]>();
        foreach (string word in vocabArray)
        {
            var subwordHashes = wordSubwords[word];
            var vector = new double[_vectorSize];

            foreach (int hash in subwordHashes)
            {
                for (int i = 0; i < _vectorSize; i++)
                {
                    vector[i] += _subwordVectors[hash, i];
                }
            }

            // Average
            if (subwordHashes.Count > 0)
            {
                for (int i = 0; i < _vectorSize; i++)
                {
                    vector[i] /= subwordHashes.Count;
                }
            }

            _wordVectors[word] = vector;
        }
    }

    private int[] BuildSamplingTable(Dictionary<string, int> vocab, string[] vocabArray)
    {
        const int tableSize = 100000;
        var table = new int[tableSize];

        double totalPow = vocab.Values.Sum(c => Math.Pow(c, 0.75));
        int idx = 0;
        double cumulative = Math.Pow(vocab[vocabArray[0]], 0.75) / totalPow;

        for (int i = 0; i < tableSize; i++)
        {
            table[i] = idx;
            if ((double)i / tableSize > cumulative && idx < vocabArray.Length - 1)
            {
                idx++;
                cumulative += Math.Pow(vocab[vocabArray[idx]], 0.75) / totalPow;
            }
        }

        return table;
    }

    private void TrainSkipGram(List<int> targetSubwords, double[,] outputVectors,
                              int contextIdx, int[] samplingTable, Random random,
                              double lr, int vocabSize)
    {
        if (_subwordVectors is null) return;

        // Compute input vector (sum of subword vectors)
        var inputVector = new double[_vectorSize];
        foreach (int hash in targetSubwords)
        {
            for (int i = 0; i < _vectorSize; i++)
            {
                inputVector[i] += _subwordVectors[hash, i];
            }
        }

        var gradInput = new double[_vectorSize];

        // Positive sample
        double dot = 0;
        for (int i = 0; i < _vectorSize; i++)
        {
            dot += inputVector[i] * outputVectors[contextIdx, i];
        }
        double sigmoid = 1.0 / (1.0 + Math.Exp(-dot));
        double grad = (1 - sigmoid) * lr;

        for (int i = 0; i < _vectorSize; i++)
        {
            gradInput[i] += grad * outputVectors[contextIdx, i];
            outputVectors[contextIdx, i] += grad * inputVector[i];
        }

        // Negative samples
        for (int n = 0; n < _negativeSamples; n++)
        {
            int negIdx = samplingTable[random.Next(samplingTable.Length)];
            if (negIdx == contextIdx) continue;

            dot = 0;
            for (int i = 0; i < _vectorSize; i++)
            {
                dot += inputVector[i] * outputVectors[negIdx, i];
            }
            sigmoid = 1.0 / (1.0 + Math.Exp(-dot));
            grad = -sigmoid * lr;

            for (int i = 0; i < _vectorSize; i++)
            {
                gradInput[i] += grad * outputVectors[negIdx, i];
                outputVectors[negIdx, i] += grad * inputVector[i];
            }
        }

        // Update subword vectors
        foreach (int hash in targetSubwords)
        {
            for (int i = 0; i < _vectorSize; i++)
            {
                _subwordVectors[hash, i] += gradInput[i] / targetSubwords.Count;
            }
        }
    }

    /// <summary>
    /// Gets the vector for a word, computing from subwords if unknown.
    /// </summary>
    public double[] GetWordVector(string word)
    {
        if (_wordVectors is null || _subwordVectors is null)
            throw new InvalidOperationException("FastTextVectorizer has not been fitted.");

        string processedWord = _lowercase ? word.ToLowerInvariant() : word;

        // Return cached vector if known
        if (_wordVectors.TryGetValue(processedWord, out var cachedVector))
        {
            return cachedVector;
        }

        // Compute from subwords for OOV words
        var subwordHashes = GetSubwords(processedWord).Select(GetSubwordHash).ToList();
        var vector = new double[_vectorSize];

        if (subwordHashes.Count == 0)
        {
            return vector;
        }

        foreach (int hash in subwordHashes)
        {
            for (int i = 0; i < _vectorSize; i++)
            {
                vector[i] += _subwordVectors[hash, i];
            }
        }

        for (int i = 0; i < _vectorSize; i++)
        {
            vector[i] /= subwordHashes.Count;
        }

        return vector;
    }

    /// <summary>
    /// Transforms documents to dense vectors by aggregating word vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's dense vector representation.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_wordVectors is null || _subwordVectors is null)
        {
            throw new InvalidOperationException("FastTextVectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        var result = new double[nDocs, _vectorSize];

        for (int d = 0; d < nDocs; d++)
        {
            var tokens = Tokenize(docList[d]).ToList();

            if (tokens.Count == 0) continue;

            var vectors = tokens.Select(GetWordVector).ToList();

            if (_aggregation == Word2VecAggregation.Mean)
            {
                foreach (var vec in vectors)
                {
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        result[d, i] += vec[i];
                    }
                }
                for (int i = 0; i < _vectorSize; i++)
                {
                    result[d, i] /= vectors.Count;
                }
            }
            else if (_aggregation == Word2VecAggregation.Sum)
            {
                foreach (var vec in vectors)
                {
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
                foreach (var vec in vectors)
                {
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        result[d, i] = Math.Max(result[d, i], vec[i]);
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
}
