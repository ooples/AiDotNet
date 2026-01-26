using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to vectors using pre-trained word embeddings loaded from files.
/// </summary>
/// <remarks>
/// <para>
/// This vectorizer loads pre-trained word embeddings from standard file formats
/// (Word2Vec text, GloVe, FastText) and uses them to create document representations.
/// </para>
/// <para>
/// Supported file formats:
/// - Word2Vec text format: First line contains vocab_size and vector_size, followed by word vectors
/// - GloVe format: Each line contains word followed by vector values (no header)
/// - FastText format: Same as Word2Vec text format
/// </para>
/// <para><b>For Beginners:</b> Use this when you have pre-trained embeddings:
/// - Download pre-trained vectors (GloVe, Word2Vec, FastText) from the web
/// - Load them once and use for any text classification or similarity task
/// - Much faster than training your own embeddings
/// - Works great for most NLP tasks
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PretrainedEmbeddingsVectorizer<T> : TextVectorizerBase<T>
{
    private readonly PretrainedFormat _format;
    private readonly int? _maxVocabSize;
    private readonly Word2VecAggregation _aggregation;
    private readonly double _oovVector; // Value to use for OOV tokens

    private Dictionary<string, double[]>? _wordVectors;
    private int _vectorSize;

    /// <summary>
    /// Gets the loaded word vectors.
    /// </summary>
    public Dictionary<string, double[]>? WordVectors => _wordVectors;

    /// <summary>
    /// Gets the vector dimensionality.
    /// </summary>
    public int VectorSize => _vectorSize;

    /// <summary>
    /// Creates a new instance of <see cref="PretrainedEmbeddingsVectorizer{T}"/> by loading embeddings from a file.
    /// </summary>
    /// <param name="embeddingsPath">Path to the pre-trained embeddings file.</param>
    /// <param name="format">Format of the embeddings file. Defaults to Auto-detect.</param>
    /// <param name="maxVocabSize">Maximum number of words to load. Null for all.</param>
    /// <param name="aggregation">How to combine word vectors into document vectors. Defaults to Mean.</param>
    /// <param name="oovVector">Value to use for out-of-vocabulary tokens (0 = ignore OOV). Defaults to 0.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    public PretrainedEmbeddingsVectorizer(
        string embeddingsPath,
        PretrainedFormat format = PretrainedFormat.Auto,
        int? maxVocabSize = null,
        Word2VecAggregation aggregation = Word2VecAggregation.Mean,
        double oovVector = 0,
        bool lowercase = true,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null)
        : base(1, 1.0, null, (1, 1), lowercase, tokenizer, stopWords)
    {
        if (string.IsNullOrWhiteSpace(embeddingsPath))
            throw new ArgumentException("Embeddings path cannot be null or empty.", nameof(embeddingsPath));
        if (!File.Exists(embeddingsPath))
            throw new FileNotFoundException("Embeddings file not found.", embeddingsPath);

        _format = format;
        _maxVocabSize = maxVocabSize;
        _aggregation = aggregation;
        _oovVector = oovVector;

        LoadEmbeddings(embeddingsPath);
    }

    /// <summary>
    /// Creates a new instance from already-loaded word vectors.
    /// </summary>
    /// <param name="wordVectors">Dictionary of word to vector mappings.</param>
    /// <param name="aggregation">How to combine word vectors into document vectors.</param>
    /// <param name="oovVector">Value to use for out-of-vocabulary tokens.</param>
    /// <param name="lowercase">Convert all text to lowercase.</param>
    /// <param name="tokenizer">Custom tokenizer function.</param>
    /// <param name="stopWords">Words to exclude.</param>
    public PretrainedEmbeddingsVectorizer(
        Dictionary<string, double[]> wordVectors,
        Word2VecAggregation aggregation = Word2VecAggregation.Mean,
        double oovVector = 0,
        bool lowercase = true,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null)
        : base(1, 1.0, null, (1, 1), lowercase, tokenizer, stopWords)
    {
        if (wordVectors is null || wordVectors.Count == 0)
            throw new ArgumentException("Word vectors cannot be null or empty.", nameof(wordVectors));

        _format = PretrainedFormat.Dictionary;
        _aggregation = aggregation;
        _oovVector = oovVector;
        _wordVectors = wordVectors;
        _vectorSize = wordVectors.Values.First().Length;
        _vocabulary = wordVectors.Keys.Select((k, i) => (k, i)).ToDictionary(x => x.k, x => x.i);
        _featureNames = Enumerable.Range(0, _vectorSize).Select(i => $"emb_dim_{i}").ToArray();
    }

    private void LoadEmbeddings(string path)
    {
        _wordVectors = new Dictionary<string, double[]>();

        using var reader = new StreamReader(path);
        string? firstLine = reader.ReadLine();
        if (firstLine is null)
            throw new InvalidOperationException("Embeddings file is empty.");

        // Detect format
        var format = _format;
        if (format == PretrainedFormat.Auto)
        {
            var parts = firstLine.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            // Word2Vec/FastText format has two numbers on first line (vocab_size, dim)
            if (parts.Length == 2 && int.TryParse(parts[0], out _) && int.TryParse(parts[1], out _))
            {
                format = PretrainedFormat.Word2Vec;
            }
            else
            {
                format = PretrainedFormat.GloVe;
            }
        }

        // Parse first line based on format
        if (format == PretrainedFormat.Word2Vec || format == PretrainedFormat.FastText)
        {
            var header = firstLine.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            _vectorSize = int.Parse(header[1]);
        }
        else // GloVe format - first line is a word vector
        {
            var (word, vector) = ParseGloveLine(firstLine);
            _vectorSize = vector.Length;
            if (_lowercase) word = word.ToLowerInvariant();
            _wordVectors[word] = vector;
        }

        // Load word vectors
        int count = _wordVectors.Count;
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (_maxVocabSize.HasValue && count >= _maxVocabSize.Value)
                break;

            try
            {
                var (word, vector) = format == PretrainedFormat.GloVe
                    ? ParseGloveLine(line)
                    : ParseWord2VecLine(line);

                if (_lowercase) word = word.ToLowerInvariant();

                if (!_wordVectors.ContainsKey(word) && vector.Length == _vectorSize)
                {
                    _wordVectors[word] = vector;
                    count++;
                }
            }
            catch
            {
                // Skip malformed lines
            }
        }

        _vocabulary = _wordVectors.Keys.Select((k, i) => (k, i)).ToDictionary(x => x.k, x => x.i);
        _featureNames = Enumerable.Range(0, _vectorSize).Select(i => $"emb_dim_{i}").ToArray();
    }

    private (string word, double[] vector) ParseWord2VecLine(string line)
    {
        var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        string word = parts[0];
        var vector = new double[parts.Length - 1];
        for (int i = 1; i < parts.Length; i++)
        {
            vector[i - 1] = double.Parse(parts[i]);
        }
        return (word, vector);
    }

    private (string word, double[] vector) ParseGloveLine(string line)
    {
        // GloVe format: word val1 val2 ... valN
        int firstSpace = line.IndexOf(' ');
        string word = line.Substring(0, firstSpace);
        var vectorParts = line.Substring(firstSpace + 1).Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var vector = new double[vectorParts.Length];
        for (int i = 0; i < vectorParts.Length; i++)
        {
            vector[i] = double.Parse(vectorParts[i]);
        }
        return (word, vector);
    }

    /// <inheritdoc/>
    public override bool IsFitted => _wordVectors is not null;

    /// <inheritdoc/>
    public override int FeatureCount => _vectorSize;

    /// <summary>
    /// Fitting is not required for pre-trained embeddings vectorizer.
    /// </summary>
    /// <param name="documents">Documents (ignored - embeddings are pre-trained).</param>
    public override void Fit(IEnumerable<string> documents)
    {
        _nDocs = documents.Count();
        // No fitting needed - embeddings are already loaded
    }

    /// <summary>
    /// Transforms documents to dense vectors using pre-trained embeddings.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's embedding.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_wordVectors is null)
        {
            throw new InvalidOperationException("PretrainedEmbeddingsVectorizer embeddings are not loaded.");
        }

        var docList = documents.ToList();
        int nDocs = docList.Count;
        var result = new double[nDocs, _vectorSize];

        for (int d = 0; d < nDocs; d++)
        {
            var tokens = Tokenize(docList[d]).ToList();
            if (tokens.Count == 0)
            {
                // Fill with OOV vector
                if (_oovVector != 0)
                {
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        result[d, i] = _oovVector;
                    }
                }
                continue;
            }

            var vectors = new List<double[]>();
            foreach (string token in tokens)
            {
                if (_wordVectors.TryGetValue(token, out var vec))
                {
                    vectors.Add(vec);
                }
                else if (_oovVector != 0)
                {
                    // Create OOV vector
                    var oovVec = new double[_vectorSize];
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        oovVec[i] = _oovVector;
                    }
                    vectors.Add(oovVec);
                }
            }

            if (vectors.Count == 0) continue;

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
}
