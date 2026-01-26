using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to dense vectors using Doc2Vec (Paragraph Vector).
/// </summary>
/// <remarks>
/// <para>
/// Doc2Vec extends Word2Vec to learn fixed-length representations for documents.
/// Two architectures are supported:
/// - PV-DM (Distributed Memory): Uses document vector with context words to predict target
/// - PV-DBOW (Distributed Bag of Words): Uses document vector alone to predict words
/// </para>
/// <para>
/// Unlike Word2Vec averaging, Doc2Vec learns document-specific vectors that capture
/// document-level semantics beyond just word averages.
/// </para>
/// <para><b>For Beginners:</b> Doc2Vec is Word2Vec for whole documents:
/// - Each document gets its own unique learned vector
/// - Captures document meaning, not just word averages
/// - Great for document similarity, classification, and clustering
/// - Better than averaging word vectors for longer documents
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class Doc2VecVectorizer<T> : TextVectorizerBase<T>
{
    private readonly int _vectorSize;
    private readonly int _windowSize;
    private readonly int _minCount;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int _negativeSamples;
    private readonly Doc2VecArchitecture _architecture;
    private readonly int? _randomState;

    private Dictionary<string, double[]>? _wordVectors;
    private Dictionary<int, double[]>? _docVectors;
    private string[]? _trainedDocIds;

    /// <summary>
    /// Gets the learned word vectors.
    /// </summary>
    public Dictionary<string, double[]>? WordVectors => _wordVectors;

    /// <summary>
    /// Gets the learned document vectors (indexed by training order).
    /// </summary>
    public Dictionary<int, double[]>? DocVectors => _docVectors;

    /// <summary>
    /// Gets the vector dimensionality.
    /// </summary>
    public int VectorSize => _vectorSize;

    /// <summary>
    /// Creates a new instance of <see cref="Doc2VecVectorizer{T}"/>.
    /// </summary>
    /// <param name="vectorSize">Dimensionality of document vectors. Defaults to 100.</param>
    /// <param name="windowSize">Context window size. Defaults to 5.</param>
    /// <param name="minCount">Minimum word frequency to include. Defaults to 5.</param>
    /// <param name="epochs">Number of training epochs. Defaults to 10.</param>
    /// <param name="learningRate">Initial learning rate. Defaults to 0.025.</param>
    /// <param name="negativeSamples">Number of negative samples. Defaults to 5.</param>
    /// <param name="architecture">Doc2Vec architecture. Defaults to PV_DM.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="randomState">Random seed for reproducibility. Null for random.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    public Doc2VecVectorizer(
        int vectorSize = 100,
        int windowSize = 5,
        int minCount = 5,
        int epochs = 10,
        double learningRate = 0.025,
        int negativeSamples = 5,
        Doc2VecArchitecture architecture = Doc2VecArchitecture.PV_DM,
        bool lowercase = true,
        int? randomState = null,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null)
        : base(minCount, 1.0, null, (1, 1), lowercase, tokenizer, stopWords)
    {
        if (vectorSize < 1)
            throw new ArgumentException("Vector size must be at least 1.", nameof(vectorSize));

        _vectorSize = vectorSize;
        _windowSize = windowSize;
        _minCount = minCount;
        _epochs = epochs;
        _learningRate = learningRate;
        _negativeSamples = negativeSamples;
        _architecture = architecture;
        _randomState = randomState;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _wordVectors is not null && _docVectors is not null;

    /// <inheritdoc/>
    public override int FeatureCount => _vectorSize;

    /// <summary>
    /// Trains Doc2Vec embeddings on the corpus.
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
        _featureNames = Enumerable.Range(0, _vectorSize).Select(i => $"doc2vec_dim_{i}").ToArray();

        if (vocabSize == 0)
        {
            _wordVectors = new Dictionary<string, double[]>();
            _docVectors = new Dictionary<int, double[]>();
            return;
        }

        var wordToIndex = _vocabulary;

        // Initialize vectors
        var random = _randomState.HasValue ? new Random(_randomState.Value) : new Random();
        var inputVectors = new double[vocabSize, _vectorSize];
        var outputVectors = new double[vocabSize, _vectorSize];
        var docVectorArray = new double[_nDocs, _vectorSize];

        double initRange = 0.5 / _vectorSize;
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < _vectorSize; j++)
            {
                inputVectors[i, j] = (random.NextDouble() - 0.5) * initRange;
                outputVectors[i, j] = 0;
            }
        }
        for (int i = 0; i < _nDocs; i++)
        {
            for (int j = 0; j < _vectorSize; j++)
            {
                docVectorArray[i, j] = (random.NextDouble() - 0.5) * initRange;
            }
        }

        // Build sampling table
        var samplingTable = BuildSamplingTable(vocab, vocabArray);

        // Training
        double lr = _learningRate;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            for (int docIdx = 0; docIdx < _nDocs; docIdx++)
            {
                var tokens = allTokens[docIdx].Where(t => wordToIndex.ContainsKey(t)).ToList();

                if (_architecture == Doc2VecArchitecture.PV_DBOW)
                {
                    TrainPVDBOW(docVectorArray, outputVectors, wordToIndex, tokens,
                               docIdx, samplingTable, random, lr, vocabSize);
                }
                else
                {
                    TrainPVDM(docVectorArray, inputVectors, outputVectors, wordToIndex, tokens,
                             docIdx, samplingTable, random, lr, vocabSize);
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

        _docVectors = new Dictionary<int, double[]>();
        for (int i = 0; i < _nDocs; i++)
        {
            var vector = new double[_vectorSize];
            for (int j = 0; j < _vectorSize; j++)
            {
                vector[j] = docVectorArray[i, j];
            }
            _docVectors[i] = vector;
        }

        _trainedDocIds = Enumerable.Range(0, _nDocs).Select(i => $"doc_{i}").ToArray();
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

    private void TrainPVDBOW(double[,] docVectors, double[,] outputVectors,
                            Dictionary<string, int> wordToIndex, List<string> tokens,
                            int docIdx, int[] samplingTable, Random random, double lr, int vocabSize)
    {
        // PV-DBOW: Use document vector to predict random words in document
        if (tokens.Count == 0) return;

        foreach (string word in tokens)
        {
            int wordIdx = wordToIndex[word];
            var gradDoc = new double[_vectorSize];

            // Positive sample
            double dot = 0;
            for (int i = 0; i < _vectorSize; i++)
            {
                dot += docVectors[docIdx, i] * outputVectors[wordIdx, i];
            }
            double sigmoid = 1.0 / (1.0 + Math.Exp(-dot));
            double grad = (1 - sigmoid) * lr;

            for (int i = 0; i < _vectorSize; i++)
            {
                gradDoc[i] += grad * outputVectors[wordIdx, i];
                outputVectors[wordIdx, i] += grad * docVectors[docIdx, i];
            }

            // Negative samples
            for (int n = 0; n < _negativeSamples; n++)
            {
                int negIdx = samplingTable[random.Next(samplingTable.Length)];
                if (negIdx == wordIdx) continue;

                dot = 0;
                for (int i = 0; i < _vectorSize; i++)
                {
                    dot += docVectors[docIdx, i] * outputVectors[negIdx, i];
                }
                sigmoid = 1.0 / (1.0 + Math.Exp(-dot));
                grad = -sigmoid * lr;

                for (int i = 0; i < _vectorSize; i++)
                {
                    gradDoc[i] += grad * outputVectors[negIdx, i];
                    outputVectors[negIdx, i] += grad * docVectors[docIdx, i];
                }
            }

            // Update doc vector
            for (int i = 0; i < _vectorSize; i++)
            {
                docVectors[docIdx, i] += gradDoc[i];
            }
        }
    }

    private void TrainPVDM(double[,] docVectors, double[,] inputVectors, double[,] outputVectors,
                          Dictionary<string, int> wordToIndex, List<string> tokens,
                          int docIdx, int[] samplingTable, Random random, double lr, int vocabSize)
    {
        // PV-DM: Use document vector + context words to predict target
        for (int pos = 0; pos < tokens.Count; pos++)
        {
            int targetIdx = wordToIndex[tokens[pos]];

            int windowStart = Math.Max(0, pos - _windowSize);
            int windowEnd = Math.Min(tokens.Count - 1, pos + _windowSize);

            // Compute context vector (average of doc vector + context word vectors)
            var contextVector = new double[_vectorSize];
            int contextCount = 1; // Start with 1 for doc vector

            // Add document vector
            for (int i = 0; i < _vectorSize; i++)
            {
                contextVector[i] = docVectors[docIdx, i];
            }

            // Add context word vectors
            for (int ctxPos = windowStart; ctxPos <= windowEnd; ctxPos++)
            {
                if (ctxPos == pos) continue;
                int ctxIdx = wordToIndex[tokens[ctxPos]];
                for (int i = 0; i < _vectorSize; i++)
                {
                    contextVector[i] += inputVectors[ctxIdx, i];
                }
                contextCount++;
            }

            // Average
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

            // Update doc vector
            for (int i = 0; i < _vectorSize; i++)
            {
                docVectors[docIdx, i] += gradContext[i] / contextCount;
            }

            // Update context word vectors
            for (int ctxPos = windowStart; ctxPos <= windowEnd; ctxPos++)
            {
                if (ctxPos == pos) continue;
                int ctxIdx = wordToIndex[tokens[ctxPos]];
                for (int i = 0; i < _vectorSize; i++)
                {
                    inputVectors[ctxIdx, i] += gradContext[i] / contextCount;
                }
            }
        }
    }

    /// <summary>
    /// Transforms documents to dense vectors by inferring document vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's inferred vector.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_wordVectors is null)
        {
            throw new InvalidOperationException("Doc2VecVectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        var result = new double[nDocs, _vectorSize];

        var random = _randomState.HasValue ? new Random(_randomState.Value + 1) : new Random();

        for (int d = 0; d < nDocs; d++)
        {
            var tokens = Tokenize(docList[d])
                .Where(t => _vocabulary?.ContainsKey(t) == true)
                .ToList();

            // Initialize document vector
            var docVector = new double[_vectorSize];
            for (int i = 0; i < _vectorSize; i++)
            {
                docVector[i] = (random.NextDouble() - 0.5) * 0.5 / _vectorSize;
            }

            if (tokens.Count == 0)
            {
                for (int i = 0; i < _vectorSize; i++)
                {
                    result[d, i] = docVector[i];
                }
                continue;
            }

            // Infer document vector through gradient descent
            double lr = _learningRate;
            for (int iter = 0; iter < 10; iter++)
            {
                foreach (string token in tokens)
                {
                    if (!_wordVectors.TryGetValue(token, out var wordVec)) continue;

                    // Simple inference: push doc vector towards word vector
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        docVector[i] += lr * (wordVec[i] - docVector[i]) * 0.1;
                    }
                }
                lr *= 0.9;
            }

            for (int i = 0; i < _vectorSize; i++)
            {
                result[d, i] = docVector[i];
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
