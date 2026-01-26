using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to dense vectors using Word2Vec-style word embeddings.
/// </summary>
/// <remarks>
/// <para>
/// Word2Vec learns dense vector representations of words where semantically similar words
/// have similar vectors. This vectorizer trains word embeddings and represents documents
/// as the average (or weighted average) of their word vectors.
/// </para>
/// <para>
/// Two architectures are supported:
/// - CBOW (Continuous Bag of Words): Predicts target word from context
/// - Skip-gram: Predicts context words from target word
/// </para>
/// <para><b>For Beginners:</b> Word2Vec captures word meaning in numbers:
/// - "king" - "man" + "woman" ≈ "queen" (famous example)
/// - Similar words have similar vectors
/// - Documents become the average of their word vectors
/// - Much smaller dimensions than bag-of-words (typically 100-300)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class Word2VecVectorizer<T> : TextVectorizerBase<T>
{
    private readonly int _vectorSize;
    private readonly int _windowSize;
    private readonly int _minCount;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int _negativeSamples;
    private readonly Word2VecArchitecture _architecture;
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
    /// Creates a new instance of <see cref="Word2VecVectorizer{T}"/>.
    /// </summary>
    /// <param name="vectorSize">Dimensionality of word vectors. Defaults to 100.</param>
    /// <param name="windowSize">Context window size. Defaults to 5.</param>
    /// <param name="minCount">Minimum word frequency to include. Defaults to 5.</param>
    /// <param name="epochs">Number of training epochs. Defaults to 5.</param>
    /// <param name="learningRate">Initial learning rate. Defaults to 0.025.</param>
    /// <param name="negativeSamples">Number of negative samples. Defaults to 5.</param>
    /// <param name="architecture">Training architecture. Defaults to SkipGram.</param>
    /// <param name="aggregation">How to combine word vectors into document vectors. Defaults to Mean.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="randomState">Random seed for reproducibility. Null for random.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">Optional ITokenizer for subword tokenization.</param>
    public Word2VecVectorizer(
        int vectorSize = 100,
        int windowSize = 5,
        int minCount = 5,
        int epochs = 5,
        double learningRate = 0.025,
        int negativeSamples = 5,
        Word2VecArchitecture architecture = Word2VecArchitecture.SkipGram,
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
        if (windowSize < 1)
            throw new ArgumentException("Window size must be at least 1.", nameof(windowSize));

        _vectorSize = vectorSize;
        _windowSize = windowSize;
        _minCount = minCount;
        _epochs = epochs;
        _learningRate = learningRate;
        _negativeSamples = negativeSamples;
        _architecture = architecture;
        _aggregation = aggregation;
        _randomState = randomState;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _wordVectors is not null;

    /// <inheritdoc/>
    public override int FeatureCount => _vectorSize;

    /// <summary>
    /// Trains Word2Vec embeddings on the corpus.
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

        _vocabulary = vocab.Keys.Select((k, i) => (k, i)).ToDictionary(x => x.k, x => x.i);
        _featureNames = Enumerable.Range(0, _vectorSize).Select(i => $"w2v_dim_{i}").ToArray();

        // Build vocabulary and indices
        var vocabArray = vocab.Keys.ToArray();
        int vocabSize = vocabArray.Length;

        if (vocabSize == 0)
        {
            _wordVectors = new Dictionary<string, double[]>();
            return;
        }

        var wordToIndex = new Dictionary<string, int>();
        for (int i = 0; i < vocabSize; i++)
        {
            wordToIndex[vocabArray[i]] = i;
        }

        // Initialize vectors
        var random = _randomState.HasValue ? new Random(_randomState.Value) : new Random();
        var inputVectors = new double[vocabSize, _vectorSize];
        var outputVectors = new double[vocabSize, _vectorSize];

        double initRange = 0.5 / _vectorSize;
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < _vectorSize; j++)
            {
                inputVectors[i, j] = (random.NextDouble() - 0.5) * initRange;
                outputVectors[i, j] = 0;
            }
        }

        // Build sampling table for negative sampling
        var samplingTable = BuildSamplingTable(vocab, vocabArray);

        // Training
        double lr = _learningRate;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            foreach (var tokens in allTokens)
            {
                var filteredTokens = tokens.Where(t => wordToIndex.ContainsKey(t)).ToList();

                for (int pos = 0; pos < filteredTokens.Count; pos++)
                {
                    string targetWord = filteredTokens[pos];
                    int targetIdx = wordToIndex[targetWord];

                    // Define context window
                    int windowStart = Math.Max(0, pos - _windowSize);
                    int windowEnd = Math.Min(filteredTokens.Count - 1, pos + _windowSize);

                    for (int ctxPos = windowStart; ctxPos <= windowEnd; ctxPos++)
                    {
                        if (ctxPos == pos) continue;

                        string contextWord = filteredTokens[ctxPos];
                        int contextIdx = wordToIndex[contextWord];

                        if (_architecture == Word2VecArchitecture.SkipGram)
                        {
                            TrainSkipGram(inputVectors, outputVectors, targetIdx, contextIdx,
                                         samplingTable, random, lr, vocabSize);
                        }
                        else
                        {
                            TrainCBOW(inputVectors, outputVectors, filteredTokens, wordToIndex,
                                     pos, windowStart, windowEnd, samplingTable, random, lr, vocabSize);
                        }
                    }
                }
            }

            // Decay learning rate
            lr = _learningRate * (1 - (double)(epoch + 1) / _epochs);
            lr = Math.Max(lr, _learningRate * 0.0001);
        }

        // Store final vectors
        _wordVectors = new Dictionary<string, double[]>();
        for (int i = 0; i < vocabSize; i++)
        {
            var vector = new double[_vectorSize];
            for (int j = 0; j < _vectorSize; j++)
            {
                vector[j] = inputVectors[i, j];
            }
            _wordVectors[vocabArray[i]] = vector;
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

    private void TrainSkipGram(double[,] inputVectors, double[,] outputVectors,
                               int targetIdx, int contextIdx, int[] samplingTable,
                               Random random, double lr, int vocabSize)
    {
        var gradInput = new double[_vectorSize];

        // Positive sample
        double dot = 0;
        for (int i = 0; i < _vectorSize; i++)
        {
            dot += inputVectors[targetIdx, i] * outputVectors[contextIdx, i];
        }
        double sigmoid = 1.0 / (1.0 + Math.Exp(-dot));
        double grad = (1 - sigmoid) * lr;

        for (int i = 0; i < _vectorSize; i++)
        {
            gradInput[i] += grad * outputVectors[contextIdx, i];
            outputVectors[contextIdx, i] += grad * inputVectors[targetIdx, i];
        }

        // Negative samples
        for (int n = 0; n < _negativeSamples; n++)
        {
            int negIdx = samplingTable[random.Next(samplingTable.Length)];
            if (negIdx == contextIdx) continue;

            dot = 0;
            for (int i = 0; i < _vectorSize; i++)
            {
                dot += inputVectors[targetIdx, i] * outputVectors[negIdx, i];
            }
            sigmoid = 1.0 / (1.0 + Math.Exp(-dot));
            grad = -sigmoid * lr;

            for (int i = 0; i < _vectorSize; i++)
            {
                gradInput[i] += grad * outputVectors[negIdx, i];
                outputVectors[negIdx, i] += grad * inputVectors[targetIdx, i];
            }
        }

        // Update input vector
        for (int i = 0; i < _vectorSize; i++)
        {
            inputVectors[targetIdx, i] += gradInput[i];
        }
    }

    private void TrainCBOW(double[,] inputVectors, double[,] outputVectors,
                          List<string> tokens, Dictionary<string, int> wordToIndex,
                          int targetPos, int windowStart, int windowEnd,
                          int[] samplingTable, Random random, double lr, int vocabSize)
    {
        int targetIdx = wordToIndex[tokens[targetPos]];

        // Compute context vector (average of context words)
        var contextVector = new double[_vectorSize];
        int contextCount = 0;

        for (int ctxPos = windowStart; ctxPos <= windowEnd; ctxPos++)
        {
            if (ctxPos == targetPos) continue;
            int ctxIdx = wordToIndex[tokens[ctxPos]];
            for (int i = 0; i < _vectorSize; i++)
            {
                contextVector[i] += inputVectors[ctxIdx, i];
            }
            contextCount++;
        }

        if (contextCount == 0) return;

        for (int i = 0; i < _vectorSize; i++)
        {
            contextVector[i] /= contextCount;
        }

        var gradContext = new double[_vectorSize];

        // Positive sample
        double dot = 0;
        for (int i = 0; i < _vectorSize; i++)
        {
            dot += contextVector[i] * outputVectors[targetIdx, i];
        }
        double sigmoid = 1.0 / (1.0 + Math.Exp(-dot));
        double grad = (1 - sigmoid) * lr;

        for (int i = 0; i < _vectorSize; i++)
        {
            gradContext[i] += grad * outputVectors[targetIdx, i];
            outputVectors[targetIdx, i] += grad * contextVector[i];
        }

        // Negative samples
        for (int n = 0; n < _negativeSamples; n++)
        {
            int negIdx = samplingTable[random.Next(samplingTable.Length)];
            if (negIdx == targetIdx) continue;

            dot = 0;
            for (int i = 0; i < _vectorSize; i++)
            {
                dot += contextVector[i] * outputVectors[negIdx, i];
            }
            sigmoid = 1.0 / (1.0 + Math.Exp(-dot));
            grad = -sigmoid * lr;

            for (int i = 0; i < _vectorSize; i++)
            {
                gradContext[i] += grad * outputVectors[negIdx, i];
                outputVectors[negIdx, i] += grad * contextVector[i];
            }
        }

        // Update context word vectors
        for (int ctxPos = windowStart; ctxPos <= windowEnd; ctxPos++)
        {
            if (ctxPos == targetPos) continue;
            int ctxIdx = wordToIndex[tokens[ctxPos]];
            for (int i = 0; i < _vectorSize; i++)
            {
                inputVectors[ctxIdx, i] += gradContext[i] / contextCount;
            }
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
            throw new InvalidOperationException("Word2VecVectorizer has not been fitted. Call Fit() or FitTransform() first.");
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
            throw new InvalidOperationException("Word2VecVectorizer has not been fitted.");

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
